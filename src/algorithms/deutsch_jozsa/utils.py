"""
Module for implementing the Deutsch-Jozsa algorithm.

Contains functions to generate oracles (constant, balanced, and random)
and to build and evaluate the complete quantum circuit for the algorithm.

Author: Himar Edhey Hernández Alonso
License: MIT
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ==========================================
# 1. ORACLE GENERATORS
# ==========================================


def dj_random_oracle(num_qubits: int) -> QuantumCircuit:
    """
    Generates a random oracle that satisfies the Deutsch-Jozsa promise
    (it is strictly constant or strictly balanced).

    Args:
        num_qubits (int): Number of qubits in the input register.

    Returns:
        QuantumCircuit: The circuit implementing the oracle.
    """
    qc = QuantumCircuit(num_qubits + 1, name="Random Oracle")

    if np.random.randint(0, 2):
        # 50% chance of flipping the output qubit
        qc.x(num_qubits)
    if np.random.randint(0, 2):
        # 50% chance of returning a constant oracle
        return qc

    # Choose half of the possible input strings for the balanced oracle
    on_states = np.random.choice(
        range(2**num_qubits), 2**num_qubits // 2, replace=False
    )

    def add_cx(qc_inner, bit_string):
        for qubit, bit in enumerate(reversed(bit_string)):
            if bit == "1":
                qc_inner.x(qubit)
        return qc_inner

    for state in on_states:
        qc.barrier()
        qc = add_cx(qc, f"{state:0b}")
        qc.mcx(list(range(num_qubits)), num_qubits)
        qc = add_cx(qc, f"{state:0b}")

    qc.barrier()
    return qc


def dj_constant_1_oracle() -> QuantumCircuit:
    """
    Generates a didactic constant oracle (f(x) = 1) for 3 input qubits.
    """
    qc = QuantumCircuit(4, name="Constant Oracle (f=1)")
    qc.cx(0, 3)
    qc.x(0)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.x(1)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.x(2)
    qc.cx(2, 3)
    return qc


def dj_spanish_balanced_oracle() -> QuantumCircuit:
    """
    Generates a balanced oracle based on the parity of the length
    of the Spanish names of the numbers from 0 to 7 (0=even, 1=odd).
    """
    qc = QuantumCircuit(4, name="Spanish Balanced Oracle")
    qc.cx(2, 3)
    qc.cx(1, 3)
    qc.ccx(0, 1, 3)
    return qc


# ==========================================
# 2. ALGORITHM ASSEMBLY
# ==========================================


def compile_dj_circuit(oracle: QuantumCircuit) -> QuantumCircuit:
    """
    Assembles the complete Deutsch-Jozsa algorithm circuit.

    Args:
        oracle (QuantumCircuit): The oracle circuit to evaluate.

    Returns:
        QuantumCircuit: The compiled circuit ready to measure.
    """
    n = oracle.num_qubits - 1
    qc = QuantumCircuit(n + 1, n)

    # Initialization
    qc.x(n)
    qc.h(range(n + 1))

    # Apply the oracle
    qc.compose(oracle, inplace=True)

    # Final interference and measurement
    qc.h(range(n))
    qc.measure(range(n), range(n))

    return qc


def evaluate_dj_function(oracle: QuantumCircuit, shots: int = 1) -> str:
    """
    Runs the Deutsch-Jozsa algorithm on a simulator to determine
    whether the function is constant or balanced.

    Args:
        oracle (QuantumCircuit): The oracle circuit.
        shots (int): Number of runs (1 is theoretically sufficient).

    Returns:
        str: "constant" if it is constant, "balanced" if it is balanced.
    """
    qc = compile_dj_circuit(oracle)

    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots, memory=True).result()
    measurements = result.get_memory()

    # If there is any '1' in the input register measurement, it is balanced
    if "1" in measurements[0]:
        return "balanced"
    return "constant"
