import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from src.algorithms.quantum_primitives.qft import iqft, qft
from src.algorithms.quantum_primitives.qpe import qpe


def test_qft_and_iqft_are_gates_with_expected_size():
    """Verify that QFT and IQFT are returned as labeled gates with the expected
    width."""
    qft_gate = qft(3)
    iqft_gate = iqft(3)

    assert qft_gate.num_qubits == 3
    assert iqft_gate.num_qubits == 3
    assert qft_gate.label == "QFT"
    assert iqft_gate.label == "IQFT"


def test_qft_followed_by_iqft_behaves_like_identity():
    """Verify that QFT followed by IQFT composes to the identity operator."""
    circuit = QuantumCircuit(3)
    circuit.append(qft(3), range(3))
    circuit.append(iqft(3), range(3))

    operator = Operator(circuit).data
    assert np.allclose(operator, np.eye(2**3))


def test_qpe_builds_expected_circuit_and_preserves_zero_eigenstate():
    """Verify that QPE builds the expected circuit and keeps a trivial eigenstate
    unchanged."""
    operator = QuantumCircuit(1)
    operator.z(0)

    circuit = qpe(2, operator)

    assert circuit.num_qubits == 3
    assert circuit.name == "QPE"

    final_state = Statevector.from_label("000").evolve(circuit)
    expected_state = Statevector.from_label("000")

    assert np.allclose(final_state.data, expected_state.data)


def test_qft_maps_zero_to_uniform_superposition():
    """Verify that QFT|000> results in |+++> (uniform superposition)."""
    n = 3
    qc = QuantumCircuit(n)
    qc.append(qft(n), range(n))

    final_state = Statevector.from_instruction(qc)

    expected_amplitude = 1 / np.sqrt(2**n)
    assert np.allclose(final_state.data, expected_amplitude)


def test_qpe_estimates_non_trivial_phase():
    """Verify QPE correctly estimates the phase of a P-gate (phase shift)."""
    phase_gate = QuantumCircuit(1)
    phase_gate.p(np.pi / 2, 0)

    n_estimation = 3
    qpe_circuit = qpe(n_estimation, phase_gate)

    total_qubits = n_estimation + 1
    qc = QuantumCircuit(total_qubits)
    qc.x(n_estimation)
    qc.compose(qpe_circuit, inplace=True)

    probabilities = Statevector.from_instruction(qc).probabilities_dict()

    most_likely_state = max(probabilities, key=probabilities.get)
    assert most_likely_state == "1010"
    assert probabilities[most_likely_state] > 0.99
