import pytest

from src.algorithms.bernstein_vazirani.utils import (
    bv_oracle,
    compile_bv_circuit,
    evaluate_bv_local,
)


def test_bv_oracle_encodes_secret():
    """Verify that the Bernstein-Vazirani oracle encodes the hidden string."""
    secret = "101"
    oracle = bv_oracle(secret)

    assert oracle.num_qubits == len(secret) + 1
    assert oracle.name == f"BV Oracle ({secret})"

    ops = oracle.count_ops()
    assert ops.get("cx", 0) == 2
    assert ops.get("id", 0) == 1


def test_compile_bv_circuit_adds_measurements():
    """Verify that the assembled Bernstein-Vazirani circuit measures the inputs."""
    oracle = bv_oracle("101")
    circuit = compile_bv_circuit(oracle)

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 3
    assert circuit.count_ops().get("measure", 0) == 3


def test_evaluate_bv_local_recovers_secret():
    """Verify that the local simulator recovers the hidden Bernstein-Vazirani string."""
    assert evaluate_bv_local(bv_oracle("101")) == "101"


@pytest.mark.parametrize("secret", ["1", "101010", "11111111"])
def test_bv_recovers_variable_length_secrets(secret):
    """Verify that the algorithm correctly scales for various secret string lengths."""
    oracle = bv_oracle(secret)
    assert evaluate_bv_local(oracle) == secret


def test_bv_oracle_gate_count():
    """Ensure the oracle creates exactly the number of CX gates as there are 1s in the
    secret."""
    secret = "11001"  # 3 ones
    oracle = bv_oracle(secret)
    ops = oracle.count_ops()
    assert ops.get("cx", 0) == secret.count("1")
    assert ops.get("id", 0) == secret.count("0")


@pytest.mark.parametrize("secret", ["000", "1111"])
def test_bv_handles_edge_case_secrets(secret):
    """Verify that the local simulator recovers all-zero and all-one secrets."""
    assert evaluate_bv_local(bv_oracle(secret)) == secret

