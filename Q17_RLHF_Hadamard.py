#!/usr/bin/env python3
"""
Q17 RLHF — tehnika: Quantum RLHF preko interferometrijske (Hadamard) superpozicije
baseline i preference stanja (čisto kvantno, bez klasičnog PPO/reward modela i bez hibrida).

Koncept (Reinforcement Learning from Human Feedback u kvantnoj formi):
  - CSV se deli na:
      · baseline deo (prvih N - K redova) → „pre-RLHF model“;
      · preference deo (poslednjih K redova) → „human feedback“ (recentne preferencije).
  - |ψ_base⟩ = amplitude-encoding freq_vector-a baseline dela.
  - |ψ_pref⟩ = amplitude-encoding freq_vector-a preference dela.

Kolo (nq + 1 qubit-a):
  1) H na aux → |+⟩ = (|0⟩ + |1⟩)/√2.
  2) Kontrolisano StatePreparation(|ψ_base⟩) kad aux = 0.
  3) Kontrolisano StatePreparation(|ψ_pref⟩) kad aux = 1.
  4) H na aux (drugi put) — uvodi INTERFERENCIJU između dva režima.

Rezultat:
  |Ψ⟩ = ½ [ |0⟩⊗(|ψ_base⟩ + |ψ_pref⟩)  +  |1⟩⊗(|ψ_base⟩ - |ψ_pref⟩) ]

Post-selekcija:
  · aux = 0 (RLHF-aligned): state ∝ (|ψ_base⟩ + |ψ_pref⟩)  — konstruktivna
    interferencija; kombinacije potkrepljene i baseline-om i feedback-om.
  · aux = 1 (anti-RLHF): state ∝ (|ψ_base⟩ - |ψ_pref⟩)  — destruktivna
    interferencija; kombinacije gde se baseline i feedback razilaze.

Razlika od Q14 (Temperature) i Q15 (Hallucination):
  Q14/Q15 koriste ortogonalne aux kanale bez drugog Hadamard-a → linear mixture
  (p = cos² |ψ_0|² + sin² |ψ_1|²; nema interferencije).
  Q17 primenjuje drugi H na aux pre merenja → amplitude se SABIRAJU/ODUZIMAJU
  (prava kvantna interferencija), tipičan pattern Hadamard-test / Ramsey-šema.

Glavna predikcija: post-selekcija aux = 0 → bias_39 → NEXT (TOP-7).
Demo: post-selekcija aux = 1 pokazuje anti-RLHF NEXT.

Sve deterministički: seed=39; amp_base, amp_pref iz CELOG CSV-a (baseline + preference podela).
Deterministička grid-optimizacija (nq, K) po cos(bias_39(aux=0), freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_K = (100, 200, 500, 1000)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


def split_amps(H: np.ndarray, nq: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(H.shape[0])
    K = int(max(1, min(K, n - 1)))
    base = H[: n - K]
    pref = H[n - K :]
    amp_base = amp_from_freq(freq_vector(base), nq)
    amp_pref = amp_from_freq(freq_vector(pref), nq)
    return amp_base, amp_pref


# =========================
# Hadamard-interference kolo (RLHF aligned / anti)
# =========================
def build_rlhf_state(H: np.ndarray, nq: int, K: int) -> Statevector:
    amp_base, amp_pref = split_amps(H, nq, K)

    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(1, name="a")
    qc = QuantumCircuit(state, aux)

    qc.h(aux[0])

    sp_base = StatePreparation(amp_base.tolist()).control(num_ctrl_qubits=1, ctrl_state=0)
    qc.append(sp_base, [aux[0]] + list(state))

    sp_pref = StatePreparation(amp_pref.tolist()).control(num_ctrl_qubits=1, ctrl_state=1)
    qc.append(sp_pref, [aux[0]] + list(state))

    qc.h(aux[0])

    return Statevector(qc)


def rlhf_state_probs(H: np.ndarray, nq: int, K: int) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Vraća:
      p_aligned (nad state | aux=0),  P(aux=0),
      p_anti    (nad state | aux=1),  P(aux=1).
    """
    sv = build_rlhf_state(H, nq, K)
    p = np.abs(sv.data) ** 2
    dim_s = 2 ** nq
    mat = p.reshape(2, dim_s)

    p_aux0 = float(mat[0].sum())
    p_aux1 = float(mat[1].sum())

    p_aligned = mat[0] / p_aux0 if p_aux0 > 1e-18 else np.ones(dim_s) / dim_s
    p_anti = mat[1] / p_aux1 if p_aux1 > 1e-18 else np.ones(dim_s) / dim_s

    return p_aligned, p_aux0, p_anti, p_aux1


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, K) po cos(bias_aligned, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for K in GRID_K:
            try:
                p_al, p_a0, _p_an, _p_a1 = rlhf_state_probs(H, nq, int(K))
                bi = bias_39(p_al)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, -nq, -int(K))
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, K=int(K), P_aux0=float(p_a0), score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q17 RLHF (Hadamard interference baseline vs preference): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2

    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| K (preference redova):", best["K"],
        "| P(aux=0, aligned):", round(float(best["P_aux0"]), 6),
        "| cos(bias_aligned, freq_csv):", round(float(best["score"]), 6),
    )

    p_al, p_a0, p_an, p_a1 = rlhf_state_probs(H, best["nq"], best["K"])
    pred_aligned = pick_next_combination(p_al)
    pred_anti = pick_next_combination(p_an)

    print("--- RLHF aligned (aux = 0, konstruktivna interferencija) ---")
    print("P(aux=0):", round(p_a0, 6), "| NEXT:", pred_aligned)
    print("--- anti-RLHF (aux = 1, destruktivna interferencija) ---")
    print("P(aux=1):", round(p_a1, 6), "| NEXT:", pred_anti)

    print("--- glavna predikcija ---")
    print("predikcija NEXT:", pred_aligned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q17 RLHF (Hadamard interference baseline vs preference): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | K (preference redova): 1000 | P(aux=0, aligned): 0.998355 | cos(bias_aligned, freq_csv): 0.898408
--- RLHF aligned (aux = 0, konstruktivna interferencija) ---
P(aux=0): 0.998355 | NEXT: (7, 9, 19, 21, 24, 27, 28)
--- anti-RLHF (aux = 1, destruktivna interferencija) ---
P(aux=1): 0.001645 | NEXT: (8, 9, 14, 16, 18, 21, 28)
--- glavna predikcija ---
predikcija NEXT: (7, 9, 19, 21, 24, 27, 28)
"""



"""
Q17_RLHF_Hadamard.py — tehnika: Quantum RLHF preko Hadamard interferencije.

Podela CSV-a:
  baseline = prvih N - K redova ("pre-RLHF model").
  preference = poslednjih K redova ("human feedback" — recentne preferencije).

Kolo (nq + 1):
  H(aux); cSP(|ψ_base⟩, aux=0); cSP(|ψ_pref⟩, aux=1); H(aux).
  |Ψ⟩ = ½[ |0⟩(|ψ_base⟩ + |ψ_pref⟩) + |1⟩(|ψ_base⟩ - |ψ_pref⟩) ]

Post-selekcija:
  aux = 0 → aligned state ∝ |ψ_base⟩ + |ψ_pref⟩  (konstruktivno; RLHF smer).
  aux = 1 → anti state    ∝ |ψ_base⟩ - |ψ_pref⟩  (destruktivno; anti-RLHF).

Tehnike:
Hadamard test / Ramsey šema — drugi H na aux proizvodi interferenciju između
dva state-preparation granja.
Post-selekcija kao kvantno-koherentno izvlačenje „aligned“ komponente.
Deterministička grid-optimizacija (nq, K).

Prednosti:
Čisto kvantno — prava interferencija između baseline i feedback stanja
(razlika od Q14/Q15 gde je samo linear mixture preko ortogonalnih aux kanala).
Jedan zajednički RLHF korak u jednom kolu (nema RL iteracija / PPO / gradient-a).
Samo 1 dodatni qubit.

Nedostaci:
Jedinstveni RLHF korak — bez iterativne interakcije (jedan round feedback-a).
Post-selekcija na aux = 0 pretpostavlja P(aux = 0) > 0; za ortogonalne |ψ_base⟩⊥|ψ_pref⟩
grana bi bila disbalansirana (manje relevantno kad oba dolaze iz istog CSV-a).
mod-39 readout meša stanja (dim 2^nq ≠ 39).
Izbor K je grid-heuristika — pravi RLHF traži „human“ feedback koji ovde aproksimira
recentnost (tail CSV-a).
"""
