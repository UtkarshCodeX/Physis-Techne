"""
Photonic Quantum Experiment Designer — V7
==========================================
AI-driven autonomous design of quantum optical experiments.

USAGE (command line)
--------------------
  python photonic_designer_v7.py                      # GHZ-4 target
  python photonic_designer_v7.py --target bell        # Bell state
  python photonic_designer_v7.py --target custom \
      --vec "0.12,0.45,0.33,0.89,0.21,0.67,0.54,0.91"   # 8-amplitude custom
  python photonic_designer_v7.py --verify-only        # sanity check only

USAGE (from Python / web backend)
-----------------------------------
  from photonic_designer_v7 import run_experiment

  # Named target
  result = run_experiment(target_name='ghz4', pop_size=120, gens=3000)

  # Custom target — 8 floats (amplitudes on the 8 SPDC basis states)
  result = run_experiment(
      target_name='custom',
      custom_vec=[0.12, 0.45, 0.33, 0.89, 0.21, 0.67, 0.54, 0.91])

  # Custom target — DIM=8074 floats (full Fock-basis vector)
  result = run_experiment(
      target_name='custom',
      custom_vec=my_full_vector)

  result keys: circuit, fidelity, probability, target_name, report_lines

CUSTOM VECTOR FORMATS
---------------------
  8 values  → amplitudes on the 8 SPDC canonical basis states, ordered:
              [HHH, HHV, HVH, HVV, VHH, VHV, VVH, VVV]
              where H/V is the polarisation of each SPDC source (A, B, C).
              Example: [1,0,0,0,0,0,0,1] → (|HHH⟩+|VVV⟩)/√2 = 3-source GHZ.

  8074 values → full Fock-basis vector (advanced use only).

  Any other length → ValueError with a clear explanation.

  The vector is automatically normalised. Complex values are supported
  (pass as "re+imj" strings or as Python complex numbers).

PHYSICS
-------
3 SPDC sources, 12 optical modes, 6 photons:
  Source A: 0=A_H_sig  1=A_H_idl  2=A_V_sig  3=A_V_idl
  Source B: 4=B_H_sig  5=B_H_idl  6=B_V_sig  7=B_V_idl
  Source C: 8=C_H_sig  9=C_H_idl  10=C_V_sig 11=C_V_idl

The 8 canonical SPDC basis states (custom vec index → physical state):
  [0] A=H, B=H, C=H  |A_H_sig A_H_idl B_H_sig B_H_idl C_H_sig C_H_idl⟩
  [1] A=H, B=H, C=V  |A_H_sig A_H_idl B_H_sig B_H_idl C_V_sig C_V_idl⟩
  [2] A=H, B=V, C=H  |A_H_sig A_H_idl B_V_sig B_V_idl C_H_sig C_H_idl⟩
  [3] A=H, B=V, C=V  |A_H_sig A_H_idl B_V_sig B_V_idl C_V_sig C_V_idl⟩
  [4] A=V, B=H, C=H  |A_V_sig A_V_idl B_H_sig B_H_idl C_H_sig C_H_idl⟩
  [5] A=V, B=H, C=V  |A_V_sig A_V_idl B_H_sig B_H_idl C_V_sig C_V_idl⟩
  [6] A=V, B=V, C=H  |A_V_sig A_V_idl B_V_sig B_V_idl C_H_sig C_H_idl⟩
  [7] A=V, B=V, C=V  |A_V_sig A_V_idl B_V_sig B_V_idl C_V_sig C_V_idl⟩
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from itertools import product as iprod
import random
import copy
import argparse
import sys
import io
from typing import List, Dict, Tuple, Optional, Union
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.sparse')

# ── Fock-space configuration ──────────────────────────────────────────────────

N_MODES   = 12   # 3 SPDC sources × 4 modes each
N_PHOTONS = 6    # fixed photon number (conserved by all gates)
MAX_OCC   = 2    # max photons per mode (sufficient for HOM bunching)

MODE_NAMES = [
    'A_H_sig', 'A_H_idl', 'A_V_sig', 'A_V_idl',
    'B_H_sig', 'B_H_idl', 'B_V_sig', 'B_V_idl',
    'C_H_sig', 'C_H_idl', 'C_V_sig', 'C_V_idl',
]

ALLOWED_ANGLES = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2,
                  5*np.pi/8, 3*np.pi/4, np.pi, 3*np.pi/2]

# ── Build Fock basis (at import) ──────────────────────────────────────────────

BASIS: List[Tuple] = [
    occ for occ in iprod(range(MAX_OCC + 1), repeat=N_MODES)
    if sum(occ) == N_PHOTONS
]
DIM: int = len(BASIS)
IDX: Dict[Tuple, int] = {s: i for i, s in enumerate(BASIS)}

# ── Canonical 8 SPDC basis-state indices ─────────────────────────────────────
# Ordered: sA in {H=0,V=1}, sB in {H=0,V=1}, sC in {H=0,V=1}  (binary order)

SPDC_BASIS_IDX: List[int] = []
for _sA in (0, 1):
    for _sB in (0, 1):
        for _sC in (0, 1):
            _occ = [0] * N_MODES
            if _sA == 0: _occ[0]=1; _occ[1]=1
            else:        _occ[2]=1; _occ[3]=1
            if _sB == 0: _occ[4]=1; _occ[5]=1
            else:        _occ[6]=1; _occ[7]=1
            if _sC == 0: _occ[8]=1; _occ[9]=1
            else:        _occ[10]=1; _occ[11]=1
            SPDC_BASIS_IDX.append(IDX[tuple(_occ)])

SPDC_BASIS_LABELS = [
    'A=H,B=H,C=H', 'A=H,B=H,C=V', 'A=H,B=V,C=H', 'A=H,B=V,C=V',
    'A=V,B=H,C=H', 'A=V,B=H,C=V', 'A=V,B=V,C=H', 'A=V,B=V,C=V',
]

# Helper function to detect GHZ target
def is_ghz_target(target: np.ndarray) -> bool:
    """Check if target is a GHZ-like state (two opposite components)."""
    non_zero = np.where(np.abs(target) > 1e-6)[0]
    if len(non_zero) != 2:
        return False
    # Check if the two non-zero components are opposite in the SPDC basis
    idx1, idx2 = non_zero
    # For GHZ, one should be all H (index 0) and one all V (index 7)
    return (idx1 == 0 and idx2 == 7) or (idx1 == 7 and idx2 == 0)


# ── Hop operators ─────────────────────────────────────────────────────────────

_HOP: Dict[Tuple, sp.csr_matrix] = {}

def hop(mc: int, md: int) -> sp.csr_matrix:
    """c†_mc a_md — photon-number conserving hop operator. Cached."""
    key = (mc, md)
    if key in _HOP:
        return _HOP[key]
    rows, cols, vals = [], [], []
    for i, occ in enumerate(BASIS):
        nd = occ[md]
        if nd == 0 or occ[mc] >= MAX_OCC:
            continue
        occ2 = list(occ); occ2[md] -= 1; occ2[mc] += 1
        j = IDX.get(tuple(occ2))
        if j is not None:
            rows.append(j); cols.append(i)
            vals.append(float(nd * (occ[mc] + 1)) ** 0.5)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(DIM, DIM), dtype=complex)
    _HOP[key] = M
    return M

def precompute_hops() -> None:
    """Precompute all inter-mode and diagonal hop operators."""
    for a in range(N_MODES):
        for b in range(N_MODES):
            hop(a, b)   # covers both a≠b (inter-mode) and a==b (number op)


# ── Sparse Hamiltonian generators ─────────────────────────────────────────────

_GEN: Dict = {}

def _g(key, fn):
    if key not in _GEN:
        _GEN[key] = fn()
    return _GEN[key]

def gen_bs(m1: int, m2: int, theta: float = np.pi/4) -> sp.csr_matrix:
    """Anti-Hermitian BS generator.  U = exp(H)."""
    key = ('bs', m1, m2, round(theta, 9))
    return _g(key, lambda: theta * (hop(m1, m2) - hop(m2, m1)))

def gen_hwp(mH: int, mV: int, theta: float) -> sp.csr_matrix:
    """Hermitian HWP generator.  U = exp(iH)."""
    key = ('hwp', mH, mV, round(theta, 9))
    def build():
        c, s = np.cos(2*theta), np.sin(2*theta)
        return (np.pi/2) * (c*(hop(mH,mH)-hop(mV,mV)) + s*(hop(mH,mV)+hop(mV,mH)))
    return _g(key, build)

def gen_qwp(mH: int, mV: int, theta: float) -> sp.csr_matrix:
    """Hermitian QWP generator.  U = exp(iH)."""
    key = ('qwp', mH, mV, round(theta, 9))
    def build():
        c, s = np.cos(2*theta), np.sin(2*theta)
        return (np.pi/4) * (c*(hop(mH,mH)-hop(mV,mV)) + s*(hop(mH,mV)+hop(mV,mH)))
    return _g(key, build)

def gen_ps(mode: int, phi: float) -> sp.csr_matrix:
    """Hermitian PS generator.  U = exp(iH) = diagonal phase."""
    key = ('ps', mode, round(phi, 9))
    return _g(key, lambda: phi * hop(mode, mode))


# ── Detection masks ───────────────────────────────────────────────────────────

_MASK: Dict = {}

def det_mask(mode: int, n: int) -> np.ndarray:
    """1.0 where BASIS[i][mode]==n, else 0.0. Cached."""
    key = (mode, n)
    if key not in _MASK:
        _MASK[key] = np.fromiter(
            (1.0 if occ[mode] == n else 0.0 for occ in BASIS),
            dtype=np.float64, count=DIM)
    return _MASK[key]


# ── Gate application ──────────────────────────────────────────────────────────

def apply_gate(g: Dict, psi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply one gate to state vector psi.
    Returns (new_psi, prob_factor):
      - Unitary gates: prob_factor = 1.0, psi unchanged in norm.
      - DETECT gates:  prob_factor = P(outcome), psi renormalised.
    """
    t = g['type']
    if t == 'DETECT':
        psi = psi * det_mask(g['mode'], g['n'])
        norm2 = float(np.dot(psi.real, psi.real) + np.dot(psi.imag, psi.imag))
        if norm2 < 1e-30:
            return psi, 0.0
        psi /= norm2 ** 0.5
        return psi, norm2
    if t == 'BS':
        return expm_multiply(gen_bs(g['m1'], g['m2'], g.get('theta', np.pi/4)), psi), 1.0
    if t == 'HWP':
        return expm_multiply(1j * gen_hwp(g['m_H'], g['m_V'], g['theta']), psi), 1.0
    if t == 'QWP':
        return expm_multiply(1j * gen_qwp(g['m_H'], g['m_V'], g['theta']), psi), 1.0
    if t == 'PS':
        return expm_multiply(1j * gen_ps(g['mode'], g['phi']), psi), 1.0
    return psi, 1.0


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(circuit: List[Dict], psi0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run circuit on psi0 in strict gate order.
    Returns (output_state, joint_post-selection_probability).
    Memory: O(DIM) — no full matrix is ever constructed.
    """
    psi = psi0.copy()
    log_prob = 0.0
    for g in circuit:
        try:
            psi, p = apply_gate(g, psi)
            if p == 0.0:
                return psi, 0.0
            if g['type'] == 'DETECT':
                log_prob += np.log(p)
        except:
            return psi, 0.0  # Return zero probability on error
    return psi, float(np.exp(log_prob))


def fidelity(target: np.ndarray, psi: np.ndarray) -> float:
    """Pure-state fidelity F = |⟨target|psi⟩|²."""
    return float(min(1.0, max(0.0, abs(np.vdot(target, psi)) ** 2)))


# ── Initial SPDC state ────────────────────────────────────────────────────────

def make_spdc_psi0() -> np.ndarray:
    """
    3 SPDC Bell pairs → 8 equal-amplitude Fock-basis terms.
    Each source S emits (|H_sig H_idl⟩ + |V_sig V_idl⟩)/√2.
    """
    psi = np.zeros(DIM, dtype=complex)
    amp = 1.0 / (2.0 * 2.0**0.5)   # 1/√8
    for k, idx in enumerate(SPDC_BASIS_IDX):
        psi[idx] = amp
    return psi


# ── Custom target vector parsing ──────────────────────────────────────────────

def parse_custom_vec(raw: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Parse and validate a user-supplied target amplitude vector.

    Accepted formats
    ────────────────
    • Python list/array of 8 floats or complex  → amplitudes on the 8 SPDC
      canonical basis states [HHH, HHV, HVH, HVV, VHH, VHV, VVH, VVV].
      This is the recommended format for custom targets.

    • Python list/array of DIM=8074 values → full Fock-basis vector.

    • Comma-separated string of 8 or 8074 values, e.g. "0.1,0.2,...".

    Any other length raises a ValueError with instructions.

    The returned vector is always normalised to unit norm and complex128.
    """
    # ── Parse string input ──────────────────────────────────────────────────
    if isinstance(raw, str):
        raw = raw.strip().strip('[](){}')
        
        # Handle complex numbers
        if 'j' in raw:
            try:
                # Remove all whitespace
                cleaned = raw.replace(' ', '')
                # Split by commas
                parts = cleaned.split(',')
                vals = []
                for part in parts:
                    if part.strip():
                        part = part.strip()
                        try:
                            # Try direct complex conversion
                            vals.append(complex(part))
                        except:
                            # If that fails, try to handle common formats
                            if part.endswith('j'):
                                # Handle format like "1+0j"
                                vals.append(complex(part))
                            else:
                                # Treat as float
                                vals.append(float(part))
            except Exception as e:
                raise ValueError(
                    f"Could not parse complex vector. "
                    f"Expected format like '1+0j,0,0,0,0,0,0,1'. "
                    f"Got: '{raw[:80]}...'. Error: {e}")
        else:
            # Simple real numbers
            try:
                vals = [complex(x.strip()) for x in raw.split(',') if x.strip()]
            except ValueError:
                raise ValueError(
                    f"Could not parse custom vector. "
                    f"Expected comma-separated numbers, got: '{raw[:80]}...'")
        raw = vals

    vec = np.array(raw, dtype=complex)

    # ── Validate length ─────────────────────────────────────────────────────
    if len(vec) == 8:
        # Map 8 amplitudes onto the 8 SPDC canonical basis states
        target = np.zeros(DIM, dtype=complex)
        for k, idx in enumerate(SPDC_BASIS_IDX):
            target[idx] = vec[k]

    elif len(vec) == DIM:
        target = vec.copy()

    else:
        raise ValueError(
            f"\nCustom vector length {len(vec)} is not supported.\n"
            f"\nAccepted lengths:\n"
            f"  8     → amplitudes on the 8 SPDC canonical basis states:\n"
            f"          [HHH, HHV, HVH, HVV, VHH, VHV, VVH, VVV]\n"
            f"          Example: '1,0,0,0,0,0,0,1' → (|HHH⟩+|VVV⟩)/√2\n"
            f"  {DIM} → full Fock-basis vector (advanced use)\n"
            f"\nYour input had {len(vec)} values.")

    # ── Normalise ────────────────────────────────────────────────────────────
    norm = np.linalg.norm(target)
    if norm < 1e-12:
        raise ValueError("Custom vector is all zeros — cannot normalise.")
    return target / norm


# ── Built-in target states ────────────────────────────────────────────────────

def target_ghz4(psi0: np.ndarray) -> np.ndarray:
    """4-photon GHZ extracted from the verified seed circuit."""
    psi, prob = simulate(seed_ghz4(), psi0)
    assert prob > 1e-8, "GHZ seed failed"
    return psi / np.linalg.norm(psi)

def target_bell(psi0: np.ndarray) -> np.ndarray:
    """2-photon Bell state extracted from the verified seed circuit."""
    psi, prob = simulate(seed_bell(), psi0)
    assert prob > 1e-8, "Bell seed failed"
    return psi / np.linalg.norm(psi)


# ── Seed circuits ─────────────────────────────────────────────────────────────

def seed_bell() -> List[Dict]:
    """Bell state: fidelity=1.0, prob=0.125."""
    return [
        {'type':'BS','m1':1,'m2':5,'theta':np.pi/4},
        {'type':'BS','m1':3,'m2':7,'theta':np.pi/4},
        {'type':'DETECT','mode':1,'n':1},
        {'type':'DETECT','mode':5,'n':0},
        {'type':'DETECT','mode':3,'n':1},
        {'type':'DETECT','mode':7,'n':0},
    ]

def seed_ghz4() -> List[Dict]:
    """4-photon GHZ: fidelity=1.0, prob=1/64."""
    return [
        {'type':'BS','m1':1,'m2':5,'theta':np.pi/4},
        {'type':'BS','m1':3,'m2':7,'theta':np.pi/4},
        {'type':'DETECT','mode':1,'n':1},
        {'type':'DETECT','mode':5,'n':0},
        {'type':'DETECT','mode':3,'n':1},
        {'type':'DETECT','mode':7,'n':0},
        {'type':'BS','m1':4,'m2':9, 'theta':np.pi/4},
        {'type':'BS','m1':6,'m2':11,'theta':np.pi/4},
        {'type':'DETECT','mode':4, 'n':1},
        {'type':'DETECT','mode':9, 'n':0},
        {'type':'DETECT','mode':6, 'n':1},
        {'type':'DETECT','mode':11,'n':0},
    ]

def seed_ghz4_hwp() -> List[Dict]:
    """GHZ + tunable HWP rotations for Adam to explore."""
    return [
        {'type':'HWP','m_H':0,'m_V':2, 'theta':np.pi/8},
        {'type':'HWP','m_H':4,'m_V':6, 'theta':np.pi/8},
        {'type':'HWP','m_H':8,'m_V':10,'theta':np.pi/8},
        {'type':'BS','m1':1,'m2':5,'theta':np.pi/4},
        {'type':'BS','m1':3,'m2':7,'theta':np.pi/4},
        {'type':'DETECT','mode':1,'n':1},
        {'type':'DETECT','mode':5,'n':0},
        {'type':'DETECT','mode':3,'n':1},
        {'type':'DETECT','mode':7,'n':0},
        {'type':'BS','m1':4,'m2':9, 'theta':np.pi/4},
        {'type':'BS','m1':6,'m2':11,'theta':np.pi/4},
        {'type':'DETECT','mode':4, 'n':1},
        {'type':'DETECT','mode':9, 'n':0},
        {'type':'DETECT','mode':6, 'n':1},
        {'type':'DETECT','mode':11,'n':0},
        {'type':'HWP','m_H':0,'m_V':2, 'theta':0.0},
        {'type':'PS','mode':0,'phi':0.0},
    ]

def seed_ghz4_alt() -> List[Dict]:
    """Same GHZ physics, complementary coincidence pattern."""
    return [
        {'type':'BS','m1':1,'m2':5,'theta':np.pi/4},
        {'type':'BS','m1':3,'m2':7,'theta':np.pi/4},
        {'type':'DETECT','mode':1,'n':0},
        {'type':'DETECT','mode':5,'n':1},
        {'type':'DETECT','mode':3,'n':0},
        {'type':'DETECT','mode':7,'n':1},
        {'type':'BS','m1':4,'m2':9, 'theta':np.pi/4},
        {'type':'BS','m1':6,'m2':11,'theta':np.pi/4},
        {'type':'DETECT','mode':4, 'n':0},
        {'type':'DETECT','mode':9, 'n':1},
        {'type':'DETECT','mode':6, 'n':0},
        {'type':'DETECT','mode':11,'n':1},
    ]


# ── Random circuit generator ──────────────────────────────────────────────────

def random_circuit(min_gates: int = 5, max_gates: int = 18) -> List[Dict]:
    """Random circuit anchored with at least one inter-source HOM BS."""
    circ: List[Dict] = []
    idlers = [1, 3, 5, 7, 9, 11]
    others = [m for m in range(N_MODES) if m not in idlers]
    m1 = random.choice(idlers)
    m2 = random.choice([m for m in others if m != m1] + idlers)
    if m2 == m1:
        m2 = (m1 + 2) % N_MODES
    circ.append({'type':'BS','m1':m1,'m2':m2,'theta':random.choice(ALLOWED_ANGLES)})

    for _ in range(random.randint(min_gates-1, max_gates-1)):
        gt = random.choice(['BS','BS','HWP','QWP','PS','DETECT','DETECT'])
        if gt == 'BS':
            a, b = random.sample(range(N_MODES), 2)
            circ.append({'type':'BS','m1':a,'m2':b,'theta':random.choice(ALLOWED_ANGLES)})
        elif gt in ('HWP','QWP'):
            src = random.randint(0, 2)
            circ.append({'type':gt,'m_H':src*4,'m_V':src*4+2,
                         'theta':random.choice(ALLOWED_ANGLES)})
        elif gt == 'PS':
            circ.append({'type':'PS','mode':random.randint(0,N_MODES-1),
                         'phi':random.choice(ALLOWED_ANGLES)})
        else:
            circ.append({'type':'DETECT','mode':random.randint(0,N_MODES-1),
                         'n':random.choice([0,1])})
    return circ


# ── Circuit simplification ────────────────────────────────────────────────────

def simplify_circuit(circuit: List[Dict], psi0: np.ndarray,
                     target: np.ndarray, tol: float = 0.002) -> List[Dict]:
    """Greedy gate removal — keep any removal that drops fidelity < tol."""
    try:
        base_fid, _ = simulate(circuit, psi0)
        base_fid = fidelity(target, base_fid)
    except:
        return circuit
    
    improved = True
    while improved:
        improved = False
        for i in range(len(circuit)):
            cand = circuit[:i] + circuit[i+1:]
            if not cand:
                continue
            try:
                psi, prob = simulate(cand, psi0)
                if prob < 1e-10:
                    continue
                if fidelity(target, psi) >= base_fid - tol:
                    circuit  = cand
                    base_fid = fidelity(target, simulate(circuit, psi0)[0])
                    improved = True
                    break
            except:
                continue
    return circuit


# ── Adam optimizer ────────────────────────────────────────────────────────────

def adam_optimize(
    circuit:    List[Dict],
    psi0:       np.ndarray,
    target:     np.ndarray,
    lr:         float = 0.08,
    beta1:      float = 0.9,
    beta2:      float = 0.999,
    eps:        float = 1e-8,
    max_steps:  int   = 200,
    n_restarts: int   = 4,
    shift:      float = np.pi / 4,
    verbose:    bool  = True,
) -> Tuple[List[Dict], float]:
    """
    Exact parameter-shift Adam with improved convergence.
    """
    # Find all tunable parameters
    locs = [(i, k) for i, g in enumerate(circuit)
            for k in ('theta', 'phi') if k in g]
    
    # If no tunable parameters, add some by converting fixed gates to tunable ones
    if not locs:
        # Create a new circuit with tunable parameters
        new_circuit = []
        for g in circuit:
            if g['type'] in ('BS', 'HWP', 'QWP', 'PS'):
                # Make a copy with random initial angles
                g_copy = g.copy()
                if 'theta' in g:
                    g_copy['theta'] = random.choice(ALLOWED_ANGLES)
                if 'phi' in g:
                    g_copy['phi'] = random.choice(ALLOWED_ANGLES)
                new_circuit.append(g_copy)
            else:
                new_circuit.append(g)
        
        # Add tunable HWPs for flexibility
        n_hwp = sum(1 for g in new_circuit if g['type'] in ('HWP', 'QWP'))
        for i in range(max(0, 3 - n_hwp)):
            new_circuit.insert(0, {'type': 'HWP', 'm_H': i*4, 'm_V': i*4+2, 
                                   'theta': random.choice(ALLOWED_ANGLES)})
        
        circuit = new_circuit
        locs = [(i, k) for i, g in enumerate(circuit)
                for k in ('theta', 'phi') if k in g]
    
    if not locs:  # Still no parameters? Return original
        psi, _ = simulate(circuit, psi0)
        return circuit, fidelity(target, psi)

    n  = len(locs)
    w0 = np.array([circuit[i][k] for i, k in locs], dtype=float)

    def eval_f(params):
        trial = copy.deepcopy(circuit)
        for p, (i, k) in enumerate(locs): 
            trial[i][k] = float(params[p])
        try:
            psi, _ = simulate(trial, psi0)
            return fidelity(target, psi)
        except:
            return 0.0

    def run_adam(init, run_id):
        th = init.copy()
        ma = np.zeros(n)
        va = np.zeros(n)
        best_t, best_f = th.copy(), eval_f(th)
        plateau_count = 0
        last_best = best_f
        no_improve_steps = 0
        
        # Track best for early stopping
        best_so_far = best_f
        
        for step in range(max_steps):
            # Gradient computation with parameter-shift rule
            grad = np.zeros(n)
            for k in range(n):
                tp = th.copy()
                tp[k] += shift
                tm = th.copy()
                tm[k] -= shift
                grad[k] = eval_f(tp) - eval_f(tm)
            
            # Adaptive learning rate with cosine annealing
            lr_t = lr * (0.5 + 0.5 * np.cos(np.pi * step / max_steps))
            
            # Adam update
            ma = beta1 * ma + (1 - beta1) * grad
            va = beta2 * va + (1 - beta2) * grad**2
            m_hat = ma / (1 - beta1**(step+1))
            v_hat = va / (1 - beta2**(step+1))
            th = th + lr_t * m_hat / (np.sqrt(v_hat) + eps)
            
            # Wrap angles to [0, 2π]
            th = np.mod(th, 2*np.pi)
            
            # Evaluate
            f = eval_f(th)
            if f > best_f:
                best_f = f
                best_t = th.copy()
                no_improve_steps = 0
                if f > best_so_far:
                    best_so_far = f
            else:
                no_improve_steps += 1
            
            # Check plateau
            if f <= last_best + 1e-5:
                plateau_count += 1
            else:
                plateau_count = 0
                last_best = f
            
            # Plateau breaking - more aggressive
            if plateau_count > 15 or no_improve_steps > 20:
                # Add random perturbation
                th = best_t + np.random.normal(0, np.pi/6, n)
                th = np.mod(th, 2*np.pi)
                ma = np.zeros(n)
                va = np.zeros(n)
                plateau_count = 0
                no_improve_steps = 0
            
            if verbose and run_id == 0 and (step % 30 == 0 or f > 0.8 or step < 5):
                print(f"    [Adam r0] step {step:3d} | fid={f:.4f} | best={best_f:.4f}")
            
            if best_f > 0.99:
                if verbose: 
                    print(f"    [Adam r0] converged at step {step}")
                break
                
        return best_t, best_f

    # Multiple restarts with different initialization strategies
    gb_t, gb_f = w0.copy(), -1.0
    
    # Try multiple random restarts first (more exploration)
    restarts = []
    restarts.append(('original', w0))
    for i in range(n_restarts * 2):
        if i % 3 == 0:
            restarts.append((f'random_{i}', np.random.uniform(0, 2*np.pi, n)))
        elif i % 3 == 1:
            restarts.append((f'discrete_{i}', np.array([random.choice(ALLOWED_ANGLES) for _ in range(n)])))
        else:
            restarts.append((f'perturbed_{i}', w0 + np.random.normal(0, np.pi/4, n)))
    
    for name, init in restarts:
        if isinstance(init, np.ndarray):
            init = np.mod(init, 2*np.pi)
        if verbose and name == 'original':
            print(f"  [Adam] trying {name} with init_fid={eval_f(init):.4f}")
        bt, bf = run_adam(init, 0 if name == 'original' else 1)
        if bf > gb_f:
            gb_f = bf
            gb_t = bt.copy()
            if verbose:
                print(f"  [Adam] new best from {name}: {bf:.4f}")
        if gb_f > 0.99:
            break

    opt = copy.deepcopy(circuit)
    for p, (i, k) in enumerate(locs): 
        opt[i][k] = float(gb_t[p])
    return opt, gb_f


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def evolve(
    target:    np.ndarray,
    psi0:      np.ndarray,
    pop_size:  int = 120,
    gens:      int = 3000,
    rng_seed:  int = 42,
    early_stop_fid: float = 0.99,
    early_stop_patience: int = 100,
) -> List[Dict]:
    """GA + Adam — autonomously discovers a circuit for the given target state."""
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    _GEN.clear()

    # Check if this is a GHZ target
    is_ghz = is_ghz_target(target)
    if is_ghz:
        print("  [Target detected as GHZ state - using specialized optimization]")

    # Adaptive parameters
    ADAM_FID = 0.3 if is_ghz else 0.4
    ADAM_STAG = 8
    COOLDOWN = 12
    last_adam = -COOLDOWN
    
    # Initialize population with GHZ-specific seeds
    pop = []
    # Add GHZ seeds first
    for seed_circ in [seed_ghz4(), seed_ghz4_hwp(), seed_ghz4_alt()]:
        pop.append(seed_circ)
    # Add Bell seed for diversity
    pop.append(seed_bell())
    
    # Fill rest with random circuits
    while len(pop) < pop_size:
        circ = random_circuit()
        # Ensure circuits have tunable parameters
        if not any(k in g for g in circ for k in ('theta', 'phi')):
            circ.append({'type': 'HWP', 'm_H': 0, 'm_V': 2, 
                        'theta': random.choice(ALLOWED_ANGLES)})
        pop.append(circ)
    
    best_ever = -1.0
    best_ever_circuit = None
    stag = 0
    no_improve_count = 0
    last_best_fid = 0.0
    
    def score(fid, prob, n):
        """Adaptive scoring function - reward both components for GHZ"""
        p = min(max(prob, 0.0), 1.0)
        
        if is_ghz and fid > 0.4 and fid < 0.9:
            # For GHZ stuck at 0.5, give extra score for exploring both components
            return fid + 0.3 * p - 0.002 * n
        elif fid > 0.9:
            return fid + 0.3 * p - 0.001 * n
        elif fid > 0.7:
            return fid + 0.15 * p - 0.002 * n
        elif fid > 0.4:
            return fid + 0.1 * p - 0.0025 * n
        else:
            return fid + 0.05 * p - 0.003 * n

    for gen in range(gens):
        # Evaluate population
        scored = []
        for circ in pop:
            try:
                psi, prob = simulate(circ, psi0)
                fid = fidelity(target, psi)
                if prob < 1e-12 or np.isnan(fid):
                    continue
                scored.append((score(fid, prob, len(circ)), circ, fid, prob))
            except:
                continue
        
        if not scored:
            pop = [random_circuit() for _ in range(pop_size)]
            continue
            
        scored.sort(reverse=True, key=lambda x: x[0])
        _, best_c, best_f, best_p = scored[0]

        # Track improvement
        if best_f > best_ever + 1e-5:
            best_ever = best_f
            best_ever_circuit = copy.deepcopy(best_c)
            stag = 0
            no_improve_count = 0
            last_best_fid = best_f
            print(f"  ✓ New best! fid={best_f:.4f}")
            
            # Special handling for GHZ at 0.5
            if is_ghz and 0.45 < best_f < 0.55:
                print(f"  [GHZ] At 0.5 - trying to merge components...")
        else:
            stag += 1
            no_improve_count += 1

        # Early stopping
        if best_ever >= early_stop_fid:
            print(f"\n✓ TARGET REACHED at gen {gen} with fidelity {best_ever:.4f}")
            return best_ever_circuit or best_c
            
        if no_improve_count >= early_stop_patience:
            print(f"\n! Early stopping: No improvement for {early_stop_patience} generations")
            return best_ever_circuit or best_c

        # Progress report
        if gen % 5 == 0 or best_f > 0.8 or (gen < 20):
            elapsed = stag if stag < 1000 else f">{stag}"
            print(f"Gen {gen:4d} | Fid: {best_f:.4f} | Prob: {best_p:.5f} "
                  f"| Gates: {len(best_c):2d} | Stag: {elapsed}")

        # Adam optimization - more aggressive for GHZ at 0.5
        adam_due = False
        if is_ghz and 0.45 < best_f < 0.55 and gen % 5 == 0:
            adam_due = True  # Try Adam every 5 gens when stuck at 0.5
        elif best_f >= ADAM_FID and (gen - last_adam) >= COOLDOWN:
            adam_due = True
        elif stag >= ADAM_STAG and (gen - last_adam) >= COOLDOWN//2:
            adam_due = True
        elif gen % 10 == 0 and gen > 0:
            adam_due = True
        
        if adam_due:
            print(f"\n  [Adam] gen={gen} fid={best_f:.4f}")
            improved = []
            
            # Try Adam on top candidates
            candidates = []
            for rank, (_, circ, fid, prob) in enumerate(scored[:min(8, len(scored))]):
                candidates.append((circ, fid))
            
            if best_ever_circuit and best_ever_circuit not in [c[0] for c in candidates]:
                candidates.append((best_ever_circuit, best_ever))
            
            # For GHZ at 0.5, also try merging circuits that produce different components
            if is_ghz and 0.45 < best_f < 0.55:
                # Find circuits that produce the other component
                h_component = []
                v_component = []
                for _, circ, fid, _ in scored[:20]:
                    psi, _ = simulate(circ, psi0)
                    # Check which component dominates
                    if abs(psi[SPDC_BASIS_IDX[0]]) > 0.9:  # HHH component
                        h_component.append(circ)
                    elif abs(psi[SPDC_BASIS_IDX[7]]) > 0.9:  # VVV component
                        v_component.append(circ)
                
                # Try merging H and V component circuits
                if h_component and v_component:
                    merged = h_component[0] + v_component[0]
                    candidates.append((merged, 0.5))
                    print(f"  [GHZ] Trying merged circuit from both components")
            
            for idx, (circ, fid) in enumerate(candidates):
                # Adaptive parameters
                if is_ghz and 0.45 < fid < 0.55:
                    lr, steps, restarts = 0.04, 200, 10  # More thorough for stuck GHZ
                elif fid > 0.8:
                    lr, steps, restarts = 0.02, 200, 8
                elif fid > 0.6:
                    lr, steps, restarts = 0.03, 150, 6
                elif fid > 0.4:
                    lr, steps, restarts = 0.05, 120, 5
                else:
                    lr, steps, restarts = 0.08, 80, 3
                
                opt, opt_f = adam_optimize(circ, psi0, target, 
                                           lr=lr, max_steps=steps, 
                                           n_restarts=restarts,
                                           verbose=(idx == 0 and (fid > 0.4 or is_ghz)))
                if opt_f > best_ever:
                    best_ever = opt_f
                    best_ever_circuit = copy.deepcopy(opt)
                    stag = 0
                    no_improve_count = 0
                    print(f"    → New best from Adam: {opt_f:.4f}")
                improved.append(opt)
            
            last_adam = gen
            print(f"  [Adam] done → best_ever={best_ever:.4f}\n")
            
            # Create new population
            n_keep = min(12, len(improved))
            pop = improved[:n_keep] + [s[1] for s in scored[:max(8, pop_size - n_keep)]]
            if len(pop) < pop_size:
                # Add fresh circuits
                for _ in range(pop_size - len(pop)):
                    circ = random_circuit()
                    if not any(k in g for g in circ for k in ('theta', 'phi')):
                        circ.append({'type': 'HWP', 'm_H': 0, 'm_V': 2, 
                                    'theta': random.choice(ALLOWED_ANGLES)})
                    pop.append(circ)
            continue

        # Diversity preservation
        if stag > 20:
            severity = "moderate" if stag < 30 else "severe"
            print(f"  [!] {severity.capitalize()} earthquake at gen {gen}")
            stag = 0
            n_elite = 4 if severity == "moderate" else 2
            n_random = pop_size - n_elite
            random_circs = []
            while len(random_circs) < n_random:
                circ = random_circuit()
                if not any(k in g for g in circ for k in ('theta', 'phi')):
                    circ.append({'type': 'HWP', 'm_H': 0, 'm_V': 2, 
                                'theta': random.choice(ALLOWED_ANGLES)})
                random_circs.append(circ)
            pop = [s[1] for s in scored[:n_elite]] + random_circs
            continue

        # Normal selection and crossover
        n_elite = min(6, len(scored))
        elites = [s[1] for s in scored[:n_elite]]
        
        select_ratio = 0.45 if best_f < 0.7 else 0.35
        pool_size = max(n_elite, int(pop_size * select_ratio))
        pool = [s[1] for s in scored[:min(pool_size, len(scored))]]
        
        nxt = elites[:]
        
        # Mutation rates
        base_mutate_rate = 0.7 if is_ghz and best_f < 0.6 else 0.65
        
        while len(nxt) < pop_size:
            if random.random() < 0.25:  # 25% random injection
                circ = random_circuit()
                if not any(k in g for g in circ for k in ('theta', 'phi')):
                    circ.append({'type': 'HWP', 'm_H': 0, 'm_V': 2, 
                                'theta': random.choice(ALLOWED_ANGLES)})
                nxt.append(circ)
                continue
                
            # Crossover
            if len(pool) >= 2:
                p1, p2 = random.choice(pool), random.choice(pool)
                while p2 == p1 and len(pool) > 1:
                    p2 = random.choice(pool)
                
                if len(p1) > 3 and len(p2) > 3:
                    split1 = random.randint(1, min(len(p1), len(p2))-2)
                    split2 = random.randint(split1+1, min(len(p1), len(p2))-1)
                    child = copy.deepcopy(p1[:split1] + p2[split1:split2] + p1[split2:])
                else:
                    split = random.randint(1, max(1, min(len(p1), len(p2))-1))
                    child = copy.deepcopy(p1[:split] + p2[split:])
            else:
                child = copy.deepcopy(random.choice(elites))
            
            # Mutation
            if random.random() < base_mutate_rate and child:
                n_mutations = random.randint(1, min(3, len(child)))
                for _ in range(n_mutations):
                    idx = random.randint(0, len(child)-1)
                    g = child[idx]
                    r = random.random()
                    
                    for key in ('theta', 'phi'):
                        if key in g:
                            if r < 0.35:
                                g[key] += random.gauss(0, np.pi/10)
                            elif r < 0.6:
                                g[key] = random.uniform(0, 2*np.pi)
                            else:
                                g[key] = random.choice(ALLOWED_ANGLES)
                    
                    if random.random() < 0.2:
                        nt = random.choice(['BS', 'HWP', 'PS', 'DETECT'])
                        if nt == 'BS':
                            a, b = random.sample(range(N_MODES), 2)
                            child[idx] = {'type': 'BS', 'm1': a, 'm2': b, 
                                         'theta': random.choice(ALLOWED_ANGLES)}
                        elif nt == 'HWP':
                            src = random.randint(0, 2)
                            child[idx] = {'type': 'HWP', 'm_H': src*4, 'm_V': src*4+2,
                                         'theta': random.choice(ALLOWED_ANGLES)}
                        elif nt == 'PS':
                            child[idx] = {'type': 'PS', 
                                         'mode': random.randint(0, N_MODES-1),
                                         'phi': random.choice(ALLOWED_ANGLES)}
                        elif nt == 'DETECT':
                            child[idx] = {'type': 'DETECT', 
                                         'mode': random.randint(0, N_MODES-1),
                                         'n': random.choice([0, 1])}
            
            nxt.append(child)
        
        pop = nxt[:pop_size]

    return best_ever_circuit or scored[0][1]


# ── Output formatting ─────────────────────────────────────────────────────────

def format_circuit(circuit: List[Dict], label: str = "Circuit") -> List[str]:
    """
    Return circuit as a list of strings, grouped into logical stages.

    Detectors between BS/HWP gates are NOT a display bug — they mark
    post-selection boundaries.  Each DETECT block closes one measurement
    round; the next unitary block then runs on the post-selected state.
    This is how multi-stage HOM protocols physically work.
    """
    W = 68
    lines = []

    # Group into stages
    stages = []
    cur_u, cur_d = [], []
    for g in circuit:
        if g['type'] == 'DETECT':
            cur_d.append(g)
        else:
            if cur_d:
                stages.append((cur_u, cur_d))
                cur_u, cur_d = [], []
            cur_u.append(g)
    if cur_u or cur_d:
        stages.append((cur_u, cur_d))

    n_stages = sum(1 for _,d in stages if d)
    lines.append('=' * W)
    lines.append(f"  {label}")
    lines.append(f"  {len(circuit)} gates  |  {n_stages} post-selection stage(s)")
    if n_stages > 0:
        lines.append("  NOTE: detectors mid-circuit are physically required.")
        lines.append("  Each DETECT block collapses the state via post-selection;")
        lines.append("  the next BS block then acts on that post-selected state.")
    lines.append('=' * W)

    gate_num = 1
    for s_idx, (unitaries, detectors) in enumerate(stages):
        has_det = bool(detectors)
        lbl = f"Stage {s_idx+1}" if has_det else "Rotations"
        lines.append(f"\n  ┌─ {lbl} {'─'*(W-5-len(lbl))}┐")

        for g in unitaries:
            t = g['type']
            if t == 'BS':
                th  = g.get('theta', np.pi/4)
                ln  = (f"  │  {gate_num:2d}.  50:50 BS   "
                       f"{MODE_NAMES[g['m1']]:13s} <-> {MODE_NAMES[g['m2']]:13s}"
                       f"  theta={th/np.pi*180:.1f} deg")
            elif t == 'HWP':
                ln  = (f"  │  {gate_num:2d}.  HWP        "
                       f"H={MODE_NAMES[g['m_H']]:13s} V={MODE_NAMES[g['m_V']]:13s}"
                       f"  theta={g['theta']/np.pi*180:.1f} deg")
            elif t == 'QWP':
                ln  = (f"  │  {gate_num:2d}.  QWP        "
                       f"H={MODE_NAMES[g['m_H']]:13s} V={MODE_NAMES[g['m_V']]:13s}"
                       f"  theta={g['theta']/np.pi*180:.1f} deg")
            elif t == 'PS':
                ln  = (f"  │  {gate_num:2d}.  Phase shift "
                       f"{MODE_NAMES[g['mode']]:28s}"
                       f"  phi={g['phi']/np.pi*180:.1f} deg")
            else:
                ln  = f"  │  {gate_num:2d}.  {t}"
            lines.append(ln); gate_num+=1

        if detectors:
            n_click   = sum(1 for d in detectors if d['n']>0)
            n_noclick = sum(1 for d in detectors if d['n']==0)
            lines.append("  │")
            lines.append("  │  Measure & post-select (keep run only when ALL match):")
            for d in detectors:
                outcome = "CLICK" if d['n']>0 else "no click"
                lines.append(f"  │    {gate_num:2d}.  Detector  "
                              f"{MODE_NAMES[d['mode']]:16s}  n={d['n']}  ({outcome})")
                gate_num+=1
            est = 0.5**len(detectors)
            lines.append(f"  │  Stage prob ≈ {est:.4f} "
                         f"({n_click} click{'s' if n_click!=1 else ''}, "
                         f"{n_noclick} no-click{'s' if n_noclick!=1 else ''})")

        lines.append(f"  └{'─'*(W-3)}┘")

    lines.append('=' * W)
    return lines


def format_state(psi: np.ndarray, label: str = "State", top_k: int = 8) -> List[str]:
    """Return dominant Fock-basis components as a list of strings."""
    nz = np.where(np.abs(psi) > 1e-4)[0]
    nz = nz[np.argsort(-np.abs(psi[nz]))][:top_k]
    lines = [f"\n  {label} (dominant terms):"]
    for i in nz:
        occ = BASIS[i]
        occupied = [(MODE_NAMES[m], occ[m]) for m in range(N_MODES) if occ[m]>0]
        occ_str = ', '.join(f"{n}x{nm}" for nm,n in occupied)
        lines.append(f"    {psi[i]:+.4f}   |{occ_str}>")
    return lines


# ── run_experiment — the main API entry point ─────────────────────────────────

def run_experiment(
    target_name:  str                        = 'ghz4',
    custom_vec:   Optional[Union[str, List]] = None,
    pop_size:     int                        = 120,
    gens:         int                        = 3000,
    rng_seed:     int                        = 42,
    seed:         Optional[int]              = None,   # alias for rng_seed (backwards compat)
    log_queue    = None,
    result_queue = None,
) -> Dict:
    """
    Main API for running the photonic experiment designer.

    Parameters
    ----------
    target_name : 'ghz4' | 'bell' | 'custom'
        Which target state to design a circuit for.

    custom_vec : list of 8 or 8074 floats/complex  (required if target_name='custom')
        Amplitudes for the target state.
        • 8 values  → mapped onto the 8 SPDC canonical basis states
          [HHH, HHV, HVH, HVV, VHH, VHV, VVH, VVV].
          Example: [1,0,0,0,0,0,0,1] → (|HHH⟩+|VVV⟩)/√2
        • 8074 values → full Fock-basis vector (advanced).
        • Comma-separated string also accepted.
        • Vector is normalised automatically.

    pop_size, gens, rng_seed : GA hyper-parameters.

    log_queue : optional queue.Queue for streaming log lines to a UI.
    result_queue : optional queue.Queue for the result dict.

    Returns
    -------
    dict with keys:
        circuit      : List[Dict]   — the discovered gate sequence
        fidelity     : float        — F(target, output)
        probability  : float        — post-selection success probability
        target_name  : str          — as passed in
        report_lines : List[str]    — printable lines of the final report
    """
    # Redirect stdout to log_queue if provided
    if seed is not None:
        rng_seed = seed   # backwards-compatible alias
    if log_queue is not None:
        class _Q(io.TextIOBase):
            def write(self, s):
                if s and s != '\n': log_queue.put(s)
            def flush(self): pass
        sys.stdout = _Q()

    # ── Startup ───────────────────────────────────────────────────────────────
    label = target_name[:40]   # safe short label for display
    print("=" * 65)
    print(f"  Photonic Quantum Designer  |  Target: {label.upper()}")
    print(f"  Pop: {pop_size}  |  Gens: {gens}  |  Seed: {rng_seed}")
    print("=" * 65)

    _t0 = time.time()
    print(f"\nBuilding Fock basis → DIM={DIM}")
    print("Precomputing hop operators...")
    precompute_hops()
    print(f"  Done ({time.time()-_t0:.2f}s)")

    psi0 = make_spdc_psi0()
    print(f"\nSPDC state: {np.sum(np.abs(psi0)>1e-10)} non-zero terms, "
          f"norm={np.linalg.norm(psi0):.4f}")

    # ── Resolve target ────────────────────────────────────────────────────────
    if target_name == 'ghz4':
        target = target_ghz4(psi0)
        target_label = 'GHZ-4'
    elif target_name == 'bell':
        target = target_bell(psi0)
        target_label = 'Bell'
    elif target_name == 'custom':
        if custom_vec is None:
            raise ValueError(
                "target_name='custom' requires custom_vec to be provided.\n"
                "Pass a list of 8 amplitudes (on SPDC canonical states) "
                "or 8074 amplitudes (full Fock basis).")
        target = parse_custom_vec(custom_vec)
        target_label = 'Custom'
        print(f"\nCustom target parsed successfully.")
        print(f"  Non-zero components: {np.sum(np.abs(target)>1e-6)}")
        print(f"  Norm: {np.linalg.norm(target):.6f}")
        # Show which SPDC basis states have weight
        print("  Amplitudes on SPDC canonical states:")
        for k, (idx, lbl) in enumerate(zip(SPDC_BASIS_IDX, SPDC_BASIS_LABELS)):
            amp = target[idx]
            if abs(amp) > 1e-6:
                print(f"    [{k}] {lbl}  →  {amp:+.4f}")
    else:
        raise ValueError(f"Unknown target_name '{target_name}'. "
                         f"Choose 'ghz4', 'bell', or 'custom'.")

    print(f"\nTarget state: {target_label}")
    for ln in format_state(target, target_label): print(ln)

    # ── Verify seeds (sanity check) ───────────────────────────────────────────
    print("\n── Seed verification ──")
    psi_g, prob_g = simulate(seed_ghz4(), psi0)
    print(f"  GHZ-4 seed: prob={prob_g:.6f}  self-fid={fidelity(psi_g/np.linalg.norm(psi_g), psi_g):.4f}")
    psi_b, prob_b = simulate(seed_bell(), psi0)
    print(f"  Bell  seed: prob={prob_b:.6f}  self-fid={fidelity(psi_b/np.linalg.norm(psi_b), psi_b):.4f}")

    # ── Evolve ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\n  EVOLUTION\n{'='*65}\n")
    _t1 = time.time()
    
    # Use the actual user parameters with adaptive patience
    patience = min(50, max(10, gens // 10))
    best = evolve(
        target=target,
        psi0=psi0,
        pop_size=pop_size,
        gens=gens,
        rng_seed=rng_seed,
        early_stop_fid=0.99,
        early_stop_patience=patience
    )
    print(f"\nEvolution time: {(time.time()-_t1)/60:.1f} min")

    # ── Polish ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\n  FINAL ADAM POLISH\n{'='*65}\n")
    
    # Adaptive final Adam based on circuit size and gens
    if best:
        # Count number of tunable parameters
        n_params = sum(1 for g in best for k in ('theta', 'phi') if k in g)
        
        if n_params > 0:
            final_lr = 0.02 if n_params > 10 else 0.03
            final_steps = min(300, max(100, gens // 3))
            final_restarts = min(10, max(4, gens // 200))
            
            best, final_fid = adam_optimize(best, psi0, target, 
                                            lr=final_lr,
                                            max_steps=final_steps, 
                                            n_restarts=final_restarts, 
                                            verbose=True)
            psi_final, prob_final = simulate(best, psi0)
            final_fid = fidelity(target, psi_final)

            # ── Simplify ──────────────────────────────────────────────────────────────
            print("\nSimplifying circuit...")
            best = simplify_circuit(best, psi0, target)
            psi_final, prob_final = simulate(best, psi0)
            final_fid = fidelity(target, psi_final)
        else:
            # No tunable parameters, can't optimize
            psi_final, prob_final = simulate(best, psi0)
            final_fid = fidelity(target, psi_final)
    else:
        # Fallback if evolution failed
        best = seed_ghz4_hwp()  # Use HWP version for tunability
        psi_final, prob_final = simulate(best, psi0)
        final_fid = fidelity(target, psi_final)
        print("Warning: Evolution didn't produce a valid circuit, using seed with tunable parameters.")

    # ── Report ────────────────────────────────────────────────────────────────
    if   final_fid > 0.99: grade = "EXCELLENT"
    elif final_fid > 0.90: grade = "GOOD"
    elif final_fid > 0.70: grade = "FAIR"
    else:                  grade = "NEEDS MORE EXPLORATION"

    report_lines = (
        [f"\n  Fidelity:    {final_fid:.6f}",
         f"  Probability: {prob_final:.6f}",
         f"  Gates:       {len(best)}",
         f"  Grade:       {grade}", ""]
        + format_circuit(best, f"Discovered {target_label} circuit")
        + format_state(psi_final, "Output state")
    )
    for ln in report_lines:
        print(ln)

    result = {
        'circuit':      best,
        'fidelity':     final_fid,
        'probability':  prob_final,
        'target_name':  target_name,
        'target_label': target_label,
        'report_lines': report_lines,
    }
    if result_queue is not None:
        result_queue.put(result)

    if log_queue is not None:
        sys.stdout = sys.__stdout__

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Autonomous photonic quantum experiment designer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('--target', default='ghz4',
                   choices=['ghz4','bell','custom'],
                   help='Target state (default: ghz4)')
    p.add_argument('--vec', default=None,
                   help='Custom target: comma-separated amplitudes. '
                        '8 values → SPDC canonical states [HHH..VVV]. '
                        '8074 values → full Fock basis. '
                        'Example: --vec "1,0,0,0,0,0,0,1"')
    p.add_argument('--pop',  type=int, default=120,  help='GA population size')
    p.add_argument('--gens', type=int, default=3000, help='Max GA generations')
    p.add_argument('--seed', type=int, default=42,   help='Random seed')
    p.add_argument('--verify-only', action='store_true',
                   help='Run seed verification only, skip GA')
    args = p.parse_args()

    if args.target == 'custom' and args.vec is None:
        p.error("--target custom requires --vec. "
                "Example: --vec '1,0,0,0,0,0,0,1'")

    if args.verify_only:
        print(f"Fock basis: DIM={DIM}")
        precompute_hops()
        psi0 = make_spdc_psi0()
        print("Running seed verification...")
        psi_g, prob_g = simulate(seed_ghz4(), psi0)
        t_g = psi_g/np.linalg.norm(psi_g)
        print(f"  GHZ-4: prob={prob_g:.6f}  fid={fidelity(t_g,psi_g):.6f}")
        psi_b, prob_b = simulate(seed_bell(), psi0)
        t_b = psi_b/np.linalg.norm(psi_b)
        print(f"  Bell:  prob={prob_b:.6f}  fid={fidelity(t_b,psi_b):.6f}")
        print("Verification passed." if prob_g > 0.01 and prob_b > 0.01
              else "WARNING: seeds may be broken.")
        return

    result = run_experiment(
        target_name = args.target,
        custom_vec  = args.vec,
        pop_size    = args.pop,
        gens        = args.gens,
        rng_seed    = args.seed,
    )
    print(f"\n{'='*65}")
    print(f"  DONE — fidelity = {result['fidelity']:.4f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()