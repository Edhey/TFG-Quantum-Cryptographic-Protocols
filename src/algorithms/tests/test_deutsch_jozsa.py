"""
Unit tests for the Deutsch-Jozsa algorithm implementation.
Validates the classification logic for constant and balanced quantum oracles.
"""

import numpy as np
from qiskit.quantum_info import Operator

from src.algorithms.deutsch_jozsa import utils as dj_utils
from src.algorithms.deutsch_jozsa.utils import (
    compile_dj_circuit,
    dj_constant_1_oracle,
    dj_spanish_balanced_oracle,
    evaluate_dj_function,
)


def test_dj_constant_1_oracle_builds_expected_circuit():
    """Verify that the didactic constant oracle is built with the expected gates."""
    oracle = dj_constant_1_oracle()

    assert oracle.num_qubits == 4
    assert oracle.name == "Constant Oracle (f=1)"

    ops = oracle.count_ops()
    assert ops.get("cx", 0) == 6
    assert ops.get("x", 0) == 3


def test_dj_spanish_balanced_oracle_builds_expected_circuit():
    """Verify that the Spanish parity oracle is built as the documented balanced
    oracle."""
    oracle = dj_spanish_balanced_oracle()

    assert oracle.num_qubits == 4
    assert oracle.name == "Spanish Balanced Oracle"

    ops = oracle.count_ops()
    assert ops.get("cx", 0) == 2
    assert ops.get("ccx", 0) == 1


def test_compile_dj_circuit_adds_measurements():
    """Verify that Deutsch-Jozsa circuit assembly adds the expected classical
    readout."""
    oracle = dj_constant_1_oracle()
    circuit = compile_dj_circuit(oracle)

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 3
    assert circuit.count_ops().get("measure", 0) == 3


def test_dj_random_oracle_can_build_constant_oracle(monkeypatch):
    """Verify the random oracle returns a constant circuit when the constant branch is
    chosen."""

    values = iter([0, 1])

    def fake_randint(low, high):
        return next(values)

    monkeypatch.setattr(dj_utils.np.random, "randint", fake_randint)

    oracle = dj_utils.dj_random_oracle(3)

    assert oracle.num_qubits == 4
    assert oracle.count_ops().get("x", 0) == 0
    assert oracle.count_ops().get("mcx", 0) == 0


def test_dj_random_oracle_can_build_balanced_oracle(monkeypatch):
    """Verify the random oracle can also build the balanced branch deterministically."""

    randint_values = iter([0, 0])

    def fake_randint(low, high):
        return next(randint_values)

    def fake_choice(values, size, replace=False):
        return np.array([1, 2, 3, 4])[:size]

    monkeypatch.setattr(dj_utils.np.random, "randint", fake_randint)
    monkeypatch.setattr(dj_utils.np.random, "choice", fake_choice)

    oracle = dj_utils.dj_random_oracle(3)

    assert oracle.num_qubits == 4
    assert oracle.count_ops().get("mcx", 0) == 4


def test_evaluate_dj_function_classifies_constant_and_balanced_oracles():
    """
    Verifies that the Deutsch-Jozsa algorithm correctly identifies the nature
    of an oracle.

    This test runs the algorithm using Qiskit's AerSimulator to ensure:
    1. A constant function (f(x)=1) is classified as 'constant'.
    2. The Spanish language parity oracle (designed as 'balanced') is
       correctly classified as 'balanced'.

    The test confirms the phase kickback mechanism operates as expected
    within the quantum circuit.
    """
    # Verify constant oracle classification
    assert evaluate_dj_function(dj_constant_1_oracle()) == "constant", (
        "Failed: Constant oracle was not classified as 'constant'."
    )

    # Verify balanced oracle classification
    assert evaluate_dj_function(dj_spanish_balanced_oracle()) == "balanced", (
        "Failed: Spanish parity oracle was not classified as 'balanced'."
    )


def test_oracles_are_unitary():
    """Verify that the core oracles are mathematically unitary."""
    oracles = [dj_constant_1_oracle(), dj_spanish_balanced_oracle()]
    for oracle in oracles:
        op = Operator(oracle)
        # Identidad = op * op_adjoint
        assert np.allclose(op.dot(op.adjoint()), np.eye(2**oracle.num_qubits))
