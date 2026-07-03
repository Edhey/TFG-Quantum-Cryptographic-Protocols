"""
Quantum primitives module for cryptographic algorithms.

This module contains educational implementations of the Quantum
Phase Estimation (QPE). Designed for use in the simulation notebooks for the
final-year project.

! Notice that in this implementation, we are returning the QFT and IQFT as gates
! instead of circuits. This is to allow for modularity and reusability in larger
! quantum algorithms, such as Shor's algorithm, where the QFT and IQFT are often used
! as subroutines.

Author: Himar Edhey Hernández Alonso
Licence: MIT
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.algorithms.quantum_primitives.qft import iqft


def qpe(estimation_wires: int, operator_u: QuantumCircuit) -> Gate:
    """
    Generates the Quantum Phase Estimation (QPE) circuit for unitary operator provided.
    Args:
        estimation_wires: number of qubits in the estimation register
        operator_u: QC that represents the unitary operator U to be estimated.
    Returns:
        Gate: The complete QPE gate.
    """
    target_wires = operator_u.num_qubits
    total_qubits = estimation_wires + target_wires
    qpe_circ = QuantumCircuit(total_qubits)

    # Uniform superposition on the estimation register
    for qubit in range(estimation_wires):
        qpe_circ.h(qubit)

    # Phase kickback (controlled U^{2^j}))
    cu_gate = operator_u.control(1)

    for wire in range(estimation_wires):
        repetitions = 2**wire
        for _ in range(repetitions):
            qubits_involved = [wire] + list(range(estimation_wires, total_qubits))
            qpe_circ.append(cu_gate, qubits_involved)

    # Adding the IQFT modular to the circuit
    qpe_circ.compose(
        iqft(estimation_wires), qubits=range(estimation_wires), inplace=True
    )

    qpe_gate = qpe_circ.to_gate(label="QPE")  # Circuit to a gate for modularity
    return qpe_gate
