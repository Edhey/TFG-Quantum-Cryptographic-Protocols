"""
Unit tests for the Grover's Algorithm implementation.
Validates optimal iteration calculation, parallel diffusion logic,
and end-to-end search convergence.
"""

import math

from qiskit import QuantumCircuit

from src.algorithms.grover.grover_algorithm import Grover


def test_grover_optimal_iterations():
    """Verify that Grover dynamically calculates the optimal iterations correctly."""
    # Search space N=16, 1 solution. Formula: floor((pi/4) * sqrt(16/1)) = floor(pi) = 3
    oracle = QuantumCircuit(4)
    grover = Grover(oracle, num_solutions=1, search_space_size=16)

    assert grover.optimal_iterations == 3


def test_grover_defaults_to_global_diffuser():
    """Verify that if no registers are provided, it defaults to a global diffuser."""
    oracle = QuantumCircuit(3)
    grover = Grover(oracle)

    # It should contain exactly one register covering all qubits
    assert grover.diffusion_registers == [[0, 1, 2]]


def test_grover_parallel_diffuser_structure():
    """Verify that the parallel diffuser builds independent diffusion blocks."""
    # Force two independent 1-qubit diffusers
    oracle = QuantumCircuit(2)
    grover = Grover(oracle, diffusion_registers=[[0], [1]])

    diffuser = grover._build_diffuser()

    # FIX: Use math.isclose to compare floating point numbers modulo 2*pi
    expected_phase = (math.pi * 2) % (2 * math.pi)
    assert math.isclose(
        diffuser.global_phase % (2 * math.pi), expected_phase, abs_tol=1e-9
    )

    # Check operations: 2 registers -> should involve 2 separate inversions
    ops = diffuser.count_ops()
    assert ops.get("z", 0) == 2


def test_grover_search_convergence():
    """Integration test: Verify Grover finds the solution |11> in 2 qubits."""
    # Create an oracle that marks state |11>
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)  # CZ marks the |11> state

    # Grover search for 1 solution in N=4
    # Expected iterations: floor(pi/4 * sqrt(4/1)) = floor(pi/2) = 1
    grover = Grover(oracle, num_solutions=1, search_space_size=4)

    results = grover.search(shots=1024)

    # Check that '11' is the most frequent result
    most_frequent = max(results, key=results.get)
    assert most_frequent == "11"
    assert results["11"] > 800  # Should be very high probability


def test_grover_builds_expected_circuit_and_iterations():
    """
    Verify Grover builds the expected circuit and uses the optimal iteration count.
    """
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)

    grover = Grover(oracle, num_solutions=1)

    assert grover.optimal_iterations == 1
    assert grover.get_grover_operator().num_qubits == 2

    circuit = grover.build_circuit()
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.count_ops().get("measure", 0) == 2


def test_grover_execute_delegates_to_search(monkeypatch):
    """Verify the class-level execute helper instantiates Grover and delegates to
    search."""
    calls = {}

    def fake_search(self, shots=1024, debug=False):
        calls["shots"] = shots
        calls["debug"] = debug
        return {"11": shots}

    monkeypatch.setattr(Grover, "search", fake_search)

    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)

    result = Grover.execute(oracle, num_solutions=1, shots=256, debug=True)

    assert result == {"11": 256}
    assert calls == {"shots": 256, "debug": True}
