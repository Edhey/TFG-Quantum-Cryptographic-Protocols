"""
Quantum primitives module for cryptographic algorithms.

This module contains educational implementations of the Quantum
Fourier Transform (QFT) and its inverse (IQFT). Designed for use in the
simulation notebooks for the final-year project.

! Notice that in this implementation, we are returning the QFT and IQFT as gates
! instead of circuits. This is to allow for modularity and reusability in larger
! quantum algorithms, such as Shor's algorithm, where the QFT and IQFT are often used
! as subroutines.

Author: Himar Edhey Hernández Alonso
Licence: MIT
"""

from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def qft(n_qubits: int) -> Gate:
    """Generates the QFT circuit for n_qubits qubits.
    Args:
        n_qubits (int): Number of qubits.
    Returns:
        Gate: The QFT gate for n_qubits qubits.
    """
    qft_circ = QuantumCircuit(n_qubits)

    # Iterative application of Hadamard and Controlled Rotations
    for i in range(n_qubits - 1, -1, -1):
        qft_circ.h(i)  # apply Hadamard gate to qubit i
        for j in range(i - 1, -1, -1):
            angle = pi / (2 ** (i - j))  # The rotation angle is halved at each step
            qft_circ.cp(angle, j, i)

    for i in range(n_qubits // 2):  # Final inversion of the qubit order using SWAPs
        qft_circ.swap(i, n_qubits - i - 1)

    qft_gate = qft_circ.to_gate(label="QFT")  # Circuit to a gate for modularity
    return qft_gate


def iqft(n_qubits: int) -> Gate:
    """Generates the Inverse QFT (IQFT) circuit for n_qubits qubits analytically.
    Args:
        n_qubits (int): Number of qubits.
    Returns:
        Gate: The IQFT gate for n_qubits qubits.
    """
    circ = QuantumCircuit(n_qubits)

    # 1. Reverse the SWAP gates
    for i in range(n_qubits // 2):
        circ.swap(i, n_qubits - i - 1)

    # 2. Reverse the Hadamard and Controlled Phase rotations
    for i in range(n_qubits):
        for j in range(i):
            # The angle is negated and gates are applied in reverse order
            angle = -pi / (2 ** (i - j))
            circ.cp(angle, j, i)
        circ.h(i)

    iqft_gate = circ.to_gate(label="IQFT")  # Circuit to a gate for modularity
    return iqft_gate
