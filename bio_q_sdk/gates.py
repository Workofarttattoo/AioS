"""Common quantum gates and helpers implemented with NumPy.

These gates are intentionally explicit (no lazy broadcasting) to favor
readability. Multi-qubit expansion is performed by QuantumState.apply().
"""
from __future__ import annotations

import numpy as np
from typing import Callable

Complex = np.complex128

# Single-qubit Pauli and identity
I = np.array([[1, 0], [0, 1]], dtype=Complex)
X = np.array([[0, 1], [1, 0]], dtype=Complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=Complex)
Z = np.array([[1, 0], [0, -1]], dtype=Complex)

# Clifford / phase gates
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=Complex)
S = np.array([[1, 0], [0, 1j]], dtype=Complex)
SDG = np.conjugate(S.T)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=Complex)
TDG = np.conjugate(T.T)

def RX(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=Complex)

def RY(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=Complex)

def RZ(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2.0), 0],
                     [0, np.exp(1j * theta / 2.0)]], dtype=Complex)

def PHASE(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=Complex)

# Two-qubit gates
CNOT = np.array(
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,0,1],
     [0,0,1,0]], dtype=Complex
)

CZ = np.diag([1,1,1,-1]).astype(Complex)

SWAP = np.array(
    [[1,0,0,0],
     [0,0,1,0],
     [0,1,0,0],
     [0,0,0,1]], dtype=Complex
)


