"""
Module for the implementation of Shor's Algorithm.

This module provides an object-oriented implementation of Shor's algorithm,
capable of factoring generic integers by combining Quantum Phase Estimation
(using generalized unitary matrices) with classical continued fractions analysis.

Author: Himar Edhey Hernández Alonso
License: MIT
"""

import math
from fractions import Fraction

import numpy as np
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


class Shor:
    def __init__(self, N: int, max_attempts: int = 5, simulator=AerSimulator()):
        self.N = N
        self.n = N.bit_length()  # Nº qubits for the target register
        self.m = 2 * self.n  # Nº qubits for the estimation register (m >= 2 * log2(N))
        self.max_attempts = max_attempts
        self.simulator = simulator

    def _check_N(self):
        if self.N < 2:
            raise ValueError("N must be greater than or equal to 2.")
        if self.N % 2 == 0:
            raise ValueError("N must be an odd integer.")
        if self.N < 15:
            raise ValueError("N must be greater than or equal to 15.")

    def _create_unitary_oracle(self, a, power):
        """
        Generates the unitary matrix for the oracle U|y> = |(a^power)*y mod N>.
        Args:
            a (int): The base in the modular exponentiation.
            power (int): The exponent in the modular exponentiation.
            N (int): The modulus.
        Returns:
            UnitaryGate: The unitary gate representing the oracle.
        Raises:
            ValueError: If the generated matrix is not unitary.
        """
        dim = 2**self.n  # Dimension of the unitary matrix
        U_matrix = np.zeros((dim, dim), dtype=int)
        multiplier = pow(a, power, self.N)

        for y in range(dim):
            if y < self.N:
                target = (multiplier * y) % self.N  # Apply the modular multiplication
            else:
                target = y  # Out of range values map to themselves (identity)
            U_matrix[target, y] = 1

        is_unitary = np.allclose(np.eye(dim), U_matrix @ U_matrix.T)
        if not is_unitary:
            raise ValueError(
                "The generated matrix is not unitary. Check if 'a' and 'N' are coprime."
            )

        return UnitaryGate(U_matrix, label=f"{a}^{power}mod{self.N}")

    def _iqft(self, n: int) -> Gate:
        """Generates the Inverse QFT (IQFT) circuit for n qubits analytically.
        Args:
            n (int): Number of qubits for the IQFT.
        Returns:
            Gate: The IQFT gate for n qubits.
        """
        circ = QuantumCircuit(n)

        # 1. Reverse the SWAP gates
        for i in range(n // 2):
            circ.swap(i, n - i - 1)

        # 2. Reverse the Hadamard and Controlled Phase rotations
        for i in range(n):
            for j in range(i):
                # The angle is negated and gates are applied in reverse order
                angle = -np.pi / (2 ** (i - j))
                circ.cp(angle, j, i)
            circ.h(i)

        iqft_gate = circ.to_gate()
        iqft_gate.name = f"IQFT_{n}"
        return iqft_gate

    def _quantum_period_finding_circ(
        self, a: int, draw_circ: bool = False
    ) -> QuantumCircuit:
        """
        Builds the Quantum Phase Estimation (QPE) circuit for period finding in
        Shor's algorithm.

        Args:
            a (int): The base for the modular exponentiation in the oracle.
            draw_circ (bool): Whether to draw the circuit or not. Defaults to False.

        Returns:
            QuantumCircuit: The complete QPE circuit for period finding.
        """
        # Definition of the architecture of registers
        qr_control = QuantumRegister(self.m, name="control")
        qr_target = QuantumRegister(self.n, name="target")
        classical_registers = ClassicalRegister(self.m, name="measure")
        qc = QuantumCircuit(qr_control, qr_target, classical_registers)

        # Initialize states
        qc.h(qr_control)  # Superposition on the control register
        qc.x(qr_target[0])  # Initialize the target register to |1> (which is a^0 mod N)
        qc.barrier()

        # Oracle: Controlled Modular Exponentiation
        for i in range(self.m):
            power = 2**i
            u_cgate = self._create_unitary_oracle(a, power).control(1)
            qubits_involed = [qr_control[i]] + list(qr_target)
            qc.append(u_cgate, qubits_involed)
        qc.barrier()

        qc.append(self._iqft(self.m), qr_control)
        qc.barrier()

        # Measure the control register to extract the phase
        qc.measure(qr_control, classical_registers)

        if draw_circ:
            qc.draw("mpl", style="iqx", fold=-1, idle_wires=False, justify="left")

        return qc

    def _classical_post_processing(
        self, counts: dict, a: int, debug: bool = False
    ) -> tuple:
        """
        Uses the Continued Fractions algorithm to extract the period r.
        Delegates the mathematical checks to the cryptographic validation method.
        """
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        total_shots = sum(counts.values())

        if debug:
            print("\n--- Period Extraction and Factorization ---")

        for state_bitstring, occurrences in sorted_counts:
            decimal = int(state_bitstring, 2)
            measured_phase = decimal / (2**self.m)

            # Discard trivial collapse (phase 0)
            if measured_phase == 0:
                continue

            # Continued fractions algorithm bounded by maximum denominator N
            fraction = Fraction(measured_phase).limit_denominator(self.N)
            r_candidate = fraction.denominator

            # Only analyze statistically significant peaks (> 1% of shots)
            is_significant_peak = occurrences > (total_shots * 0.01)

            if is_significant_peak:
                if debug:
                    print(
                        f"Detected peak ({occurrences} shots) "
                        f"| Phase: {measured_phase:.5f} -> Fraction: {fraction}"
                    )

                # Delegate responsibility to the cryptographic checker
                f1, f2 = self._cryptographic_validation(a, r_candidate, debug)

                # If valid factors are returned, bubble them up and exit
                if f1 is not None and f2 is not None:
                    return (f1, f2)

        # If we exhausted all valid peaks and found nothing
        return (None, None)

    def _cryptographic_validation(
        self, a: int, r_candidate: int, debug: bool = False
    ) -> tuple:
        """
        Validates a period candidate 'r' against Shor's cryptographic conditions
        and computes the prime factors if valid.
        """
        # 1. Condition A: The period 'r' must be even
        if r_candidate % 2 != 0:
            if debug:
                print(f"   -> Period r = {r_candidate} (Odd period, invalid)")
            return (None, None)

        # 2. Condition B: a^(r/2) is not congruent to -1 mod N
        if pow(a, r_candidate // 2, self.N) == self.N - 1:
            if debug:
                print(
                    f"   -> Period r = {r_candidate} (Trivial condition met: "
                    f"a^(r/2) == -1 mod N)"
                )
            return (None, None)

        # 3. Compute the prime factors
        factor_1 = math.gcd(pow(a, r_candidate // 2) - 1, self.N)
        factor_2 = math.gcd(pow(a, r_candidate // 2) + 1, self.N)

        # 4. Verify that the factors are non-trivial
        if factor_1 not in [1, self.N] and factor_2 not in [1, self.N]:
            if debug:
                print(
                    f"   -> [!] SUCCESS: Non-trivial factors of {self.N} found: "
                    f"{factor_1} and {factor_2}"
                )
            return (factor_1, factor_2)
        else:
            if debug:
                print(f"   -> Period r = {r_candidate} (Yields trivial factors)")
            return (None, None)

    def factorize(self, debug: bool = False) -> tuple:
        """
        The main public orchestrator method. Executes the classical-quantum
        hybrid loop of Shor's algorithm.
        """
        print(f"Starting Shor's Algorithm to factorize N = {self.N}")

        for attempt in range(1, self.max_attempts + 1):
            if debug:
                print(f"\n--- Attempt {attempt}/{self.max_attempts} ---")

            a = np.random.randint(2, self.N)  # Choose a random base
            if debug:
                print(f"Randomly chosen base a = {a}")

            gcd_a_N = math.gcd(a, self.N)  # Check if 'a' is already a factor of N
            if gcd_a_N > 1:
                if debug:
                    print(f"Lucky classical guess! Found factor: {gcd_a_N}")
                return (gcd_a_N, self.N // gcd_a_N)

            # Quantum circuit
            if debug:
                print("Building and transpiling quantum circuit...")
            qc = self._quantum_period_finding_circ(a, draw_circ=(attempt == 1))
            compiled_circuit = transpile(qc, self.simulator)

            if debug:
                print("Executing on simulator...")
            job = self.simulator.run(compiled_circuit, shots=1024)
            counts = job.result().get_counts()

            # Post-processing
            f1, f2 = self._classical_post_processing(counts, a, debug)
            if f1 is not None and f2 is not None:
                return (f1, f2)
            else:
                if debug:
                    print(
                        "Failed to find valid factors from this quantum execution. "
                        "Trying new base..."
                    )

        if debug:
            print("\nMax attempts reached. Factorization failed.")
        return (None, None)

    @classmethod
    def execute(cls, N: int, max_attempts: int = 5, debug: bool = False) -> tuple:
        """
        Static convenience method to run the algorithm without manually
        instantiating the class.
        """
        # Automatically creates the instance and runs it
        solver_instance = cls(N=N, max_attempts=max_attempts)
        return solver_instance.factorize(debug=debug)
