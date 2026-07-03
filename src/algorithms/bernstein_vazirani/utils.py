"""
Module for the implementation of the Bernstein-Vazirani algorithm.

This module provides functions to generate the oracle for a hidden binary string,
compile the complete quantum circuit, and evaluate it using both local simulators
and real IBM Quantum hardware.

Author: Himar Edhey Hernández Alonso
License: MIT
"""

import json

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# ==========================================
# 1. ORACLE GENERATOR
# ==========================================


def bv_oracle(secret: str) -> QuantumCircuit:
    """
    Create a quantum circuit implementing a query gate for the
    Bernstein-Vazirani problem based on a hidden binary string.

    Args:
        secret (str): The hidden binary string to encode in the oracle.

    Returns:
        QuantumCircuit: The circuit implementing the oracle.
    """
    num_qubits = len(secret)
    qc = QuantumCircuit(num_qubits + 1, name=f"BV Oracle ({secret})")

    # Reverse the string to fit Qiskit's qubit ordering convention
    for index, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(index, num_qubits)
        elif bit == "0":
            qc.id(index)  # Identity gate applied for didactic purposes

    return qc


# ==========================================
# 2. CIRCUIT ASSEMBLER
# ==========================================


def compile_bv_circuit(oracle: QuantumCircuit) -> QuantumCircuit:
    """
    Compiles the complete circuit for use in the Bernstein-Vazirani algorithm.

    Args:
        oracle (QuantumCircuit): The query circuit (oracle) to compile.

    Returns:
        QuantumCircuit: The complete, compiled quantum circuit.
    """
    num_qubits = oracle.num_qubits - 1
    qc = QuantumCircuit(num_qubits + 1, num_qubits)

    # Initialization: put auxiliary in state |-> and inputs in superposition
    qc.x(num_qubits)
    qc.h(range(num_qubits + 1))
    qc.barrier()

    # Apply the oracle
    qc.compose(oracle, inplace=True)
    qc.barrier()

    # Apply Hadamard gates after querying the oracle and measure
    qc.h(range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    return qc


# ==========================================
# 3. EVALUATION FUNCTIONS
# ==========================================


def evaluate_bv_local(oracle: QuantumCircuit) -> str:
    """
    Executes the Bernstein-Vazirani algorithm using a local simulator.

    Args:
        oracle (QuantumCircuit): The oracle circuit to evaluate.

    Returns:
        str: The hidden binary string found by the algorithm.
    """
    qc = compile_bv_circuit(oracle)

    # Run using the local AerSimulator with a single shot
    result = AerSimulator().run(qc, shots=1, memory=True).result()
    return result.get_memory()[0]


def evaluate_bv_hardware(
    secret: str, backend, cache_file: str = "bernstein_vazirani_cache.json"
) -> dict:
    """
    Executes the Bernstein-Vazirani algorithm on real IBM Quantum hardware.

    Args:
        secret (str): The hidden binary string (used to generate the oracle).
        backend: The IBM Quantum backend to execute the circuit on.
        cache_file (str, optional): Path to save the results as a JSON file.

    Returns:
        dict: The execution counts showing the measurement distribution.
    """
    oracle = bv_oracle(secret)
    bv_circuit = compile_bv_circuit(oracle)

    # Transpile the circuit to an ISA compatible with the selected backend
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pass_manager.run(bv_circuit)

    # Execute using SamplerV2
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=1024)

    # Retrieve results and counts
    result = job.result()
    counts = result[0].data.c.get_counts()

    # Save to cache file if provided
    if cache_file:
        save_data = {"counts": counts, "secret": secret}
        with open(cache_file, "w") as f:
            json.dump(save_data, f)

    return counts
