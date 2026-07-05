"""
Unit tests for Shor's Algorithm implementation.
Validates the construction of the unitary oracle, the cryptographic
post-processing logic, and the end-to-end factorization process.
"""

import numpy as np
import pytest
from qiskit.circuit.library import UnitaryGate

from src.algorithms.shor.shor_algorithm import Shor


def test_shor_initialization_constraints():
    """Verify that the Shor class correctly validates the input modulus N."""
    with pytest.raises(ValueError, match="N must be greater than or equal to 2"):
        Shor(1)._check_N()

    with pytest.raises(ValueError, match="N must be an odd integer"):
        Shor(14)._check_N()

    with pytest.raises(ValueError, match="N must be an odd integer"):
        Shor(16)._check_N()


def test_unitary_oracle_unitarity():
    """Verify that the modular exponentiation oracle generates a valid unitary
    matrix."""
    solver = Shor(15)
    # Test for a=2, power=2, N=15 => (2^2 mod 15) = 4
    oracle = solver._create_unitary_oracle(a=2, power=2)

    assert isinstance(oracle, UnitaryGate)
    # Unitary condition: U * U_dagger = Identity
    matrix = oracle.to_matrix()
    identity = np.eye(len(matrix))
    assert np.allclose(matrix @ matrix.conj().T, identity)


def test_shor_cryptographic_logic():
    """
    Validates the core period-to-factor logic defined in the _cryptographic_validation
    method.
    Using N=15, a=2, period r=4. Factors should be (3, 5).
    """
    solver = Shor(15)

    # Case: Even period, not a trivial root -> Success
    assert solver._cryptographic_validation(a=2, r_candidate=4) == (3, 5)

    # Case: Odd period -> Invalid
    assert solver._cryptographic_validation(a=2, r_candidate=3) == (None, None)

    # Case: Trivial root (a^(r/2) == -1 mod N) -> Invalid
    # For N=15, if a=4, r=2. 4^(2/2) = 4 mod 15. Wait, 4^1=4, not -1.
    # We test the validation condition logic itself:
    assert solver._cryptographic_validation(a=14, r_candidate=2) == (None, None)


def test_shor_factorizes_15_deterministically(monkeypatch):
    """
    Integration test: Runs the full factorization process for N=15.
    We mock the random base 'a' to be 7, which guarantees a successful path.
    """
    # Force the random base 'a' to be 7
    monkeypatch.setattr("numpy.random.randint", lambda low, high: 7)

    solver = Shor(15)
    f1, f2 = solver.factorize(debug=True)

    # Sort results to ensure order doesn't matter (3*5 = 5*3)
    factors = sorted([f1, f2])
    assert factors == [3, 5]


def test_shor_validation_helpers():
    """Verify the Shor helper validations and the non-trivial factor checks."""
    solver = Shor(15)

    assert solver._cryptographic_validation(2, 4) == (3, 5)
    assert solver._cryptographic_validation(2, 3) == (None, None)


def test_shor_execute_delegates_to_factorize(monkeypatch):
    """Verify the class-level execute helper instantiates Shor and delegates to
    factorize."""
    calls = {}

    def fake_factorize(self, debug=False):
        calls["debug"] = debug
        return (3, 5)

    monkeypatch.setattr(Shor, "factorize", fake_factorize)

    assert Shor.execute(15, max_attempts=2, debug=True) == (3, 5)
    assert calls == {"debug": True}
