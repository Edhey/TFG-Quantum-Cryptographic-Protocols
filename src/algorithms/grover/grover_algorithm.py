"""
Module for the implementation of Grover's Algorithm.

This module provides an object-oriented implementation of Grover's algorithm,
capable of performing the search for marked states in an unstructured database,
handling both standard Grover's search and amplitude amplification with
custom heuristics. It also includes a quantum counting feature to estimate the
number of solutions when unknown.

Author: Himar Edhey Hernández Alonso
License: MIT
"""

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

from src.algorithms.quantum_primitives.qpe import qpe


class Grover:
    def __init__(
        self,
        oracle: QuantumCircuit,
        num_solutions: int = None,
        state_preparation: QuantumCircuit = None,
        search_space_size: int = None,
        diffusion_registers: list[list[int]] = None,
        simulator=AerSimulator(),
    ):
        """
        Initializes the Grover search engine.

        Args:
            oracle (QuantumCircuit): The quantum circuit implementing the oracle.
            num_solutions (int): The expected number of valid solutions.
                If None, it will be estimated using Quantum Counting.
            state_preparation (QuantumCircuit, optional): The heuristic circuit A.
                If None, standard H gates are applied.
            search_space_size (int, optional): The reduced search space N if a
                heuristic is used. Defaults to 2^n.
            diffusion_registers (list[list[int]], optional): The list of qubit registers
            to be used for the diffusion operator. Defaults to all qubits.
            simulator: The Qiskit backend/simulator to execute the circuit.
        """
        self.oracle = oracle
        self.num_qubits = oracle.num_qubits
        self.num_solutions = num_solutions
        self.simulator = simulator
        self.diffusion_registers = diffusion_registers

        # Default to global diffusion if no specific registers are provided
        if self.diffusion_registers is None:
            self.diffusion_registers = [list(range(self.num_qubits))]

        # Handle State Preparation (Standard Grover vs Amplitude Amplification)
        if state_preparation is None:
            # Standard Grover: H on all qubits
            qc = QuantumCircuit(self.num_qubits, name="Standard_H")
            qc.h(range(self.num_qubits))
            self.state_preparation = qc
            self.search_space_size = 2**self.num_qubits
        else:
            # Amplitude Amplification: Custom heuristic A
            self.state_preparation = state_preparation
            # Use provided reduced space size, or default to max
            self.search_space_size = search_space_size or (2**self.num_qubits)

        if self.num_solutions is not None:
            self.optimal_iterations = self._calculate_optimal_iterations()
        else:
            self.optimal_iterations = 0  # Will be updated after quantum counting

    def _calculate_optimal_iterations(self) -> int:
        """
        Dynamically calculates the optimal number of iterations to maximize
        the probability of measuring the target state.
        """
        N = self.search_space_size

        # Prevent math errors if solutions are >= to half the search space
        if self.num_solutions >= N / 2:
            return 0

        # Formula: floor((pi/4) * sqrt(N/M))
        theta = math.asin(math.sqrt(self.num_solutions / N))
        iterations = math.floor(math.pi / (4 * theta))

        return iterations

    def _build_diffuser(self) -> QuantumCircuit:
        """
        Constructs the generalized diffusion operator: A * (2|0><0| - I) * A_inv.
        Supports parallel amplitude amplification via multiple independent diffusers.
        If state_preparation is standard H, this naturally reduces to standard Grover.
        """
        qc = QuantumCircuit(self.num_qubits, name="Diffuser")

        qc.compose(self.state_preparation.inverse(), inplace=True)  # Apply A^(-1)

        for reg in self.diffusion_registers:
            qc.x(reg)
            # Apply multi-controlled Z gate only to qubits in the current diffusion reg
            num_controls = len(reg) - 1
            if num_controls > 0:
                qc.compose(MCMTGate(ZGate(), num_controls, 1), qubits=reg, inplace=True)
            else:
                qc.z(reg[0])  # Case of 1 single qubit

            qc.x(reg)
        qc.compose(self.state_preparation, inplace=True)  # Re-apply A

        # Add global phase of π per independent diffusion block for correct reflection
        qc.global_phase = math.pi * len(self.diffusion_registers)

        return qc

    def count_solutions(
        self, estimation_wires: int = 6, shots: int = 2048, debug: bool = False
    ) -> int:
        """
        Uses Quantum Phase Estimation to blindly estimate the number of solutions M.
        Updates the internal state (num_solutions and optimal_iterations) automatically.
        """
        if debug:
            print("\n--- QUANTUM COUNTING PHASE ---")
            print(f"Estimating solutions using {estimation_wires} precision qubits...")

        target_wires = self.num_qubits
        total_qubits = estimation_wires + target_wires
        counting_circ = QuantumCircuit(total_qubits, estimation_wires)

        # State Preparation: Apply heuristic A (or H) to the target register
        state_prep = QuantumCircuit(total_qubits)
        state_prep.compose(
            self.state_preparation,
            qubits=range(estimation_wires, total_qubits),
            inplace=True,
        )

        # Extract Grover Operator and inject into QPE
        grover_op = self.get_grover_operator()
        qpe_engine = qpe(estimation_wires, operator_u=grover_op)

        # Assemble and Measure
        counting_circ.compose(state_prep, inplace=True)
        counting_circ.compose(qpe_engine, inplace=True)
        counting_circ.measure(range(estimation_wires), range(estimation_wires))

        # Execute on Simulator
        compiled_circ = transpile(counting_circ, self.simulator)
        results = self.simulator.run(compiled_circ, shots=shots).result().get_counts()

        # Classical Post-Processing
        most_frequent_state = max(results, key=results.get)
        measured_integer = int(most_frequent_state, 2)

        N = self.search_space_size
        phase_fraction = measured_integer / (2**estimation_wires)
        theta = math.pi * phase_fraction

        M_raw = N * (math.sin(theta) ** 2)
        estimated_M = int(round(M_raw))

        if debug:
            print(f"Measured phase fraction: {phase_fraction:.4f}")
            print(f"Estimated solutions (M'): {estimated_M}")

        # Update internal class state
        self.num_solutions = estimated_M
        self.optimal_iterations = self._calculate_optimal_iterations()

        return estimated_M

    def get_grover_operator(self) -> QuantumCircuit:
        """
        Returns the combined Grover operator (Oracle + Diffuser) for a single iteration.
        This is particularly useful to export the operator for Quantum Counting.
        """
        grover_op = QuantumCircuit(self.num_qubits, name="Grover_Op")
        grover_op.compose(self.oracle, inplace=True)
        grover_op.compose(self._build_diffuser(), inplace=True)
        return grover_op

    def build_circuit(self, draw_circ: bool = False) -> QuantumCircuit:
        """
        Assembles the complete Grover's algorithm quantum circuit.
        """
        qr = QuantumRegister(self.num_qubits, name="q")
        cr = ClassicalRegister(self.num_qubits, name="c")
        qc = QuantumCircuit(qr, cr)

        qc.compose(self.state_preparation, inplace=True)  # Apply heuristic A or H

        # Apply the Grover operator the optimal number of times
        diffuser = self._build_diffuser()
        for _ in range(self.optimal_iterations):
            qc.compose(self.oracle, inplace=True)
            qc.compose(diffuser, inplace=True)
            qc.barrier()

        qc.measure(qr, cr)

        if draw_circ:
            qc.draw("mpl", style="iqp", fold=-1, idle_wires=False)

        return qc

    def search(self, shots: int = 1024, debug: bool = False) -> dict:
        """
        The main orchestrator method. Compiles and runs the circuit on the simulator.

        Returns:
            dict: The measurement counts.
        """
        if self.num_solutions is None:
            if debug:
                print("Number of solutions unknown. Running Quantum Counting...")
            self.count_solutions(debug=debug)

        if debug:
            print(f"Starting Grover's Algorithm for {self.num_qubits} qubits.")
            print(f"Expected solutions: {self.num_solutions}")
            print(f"Applying {self.optimal_iterations} optimal iterations...")

        qc = self.build_circuit(draw_circ=debug)

        if debug:
            print("Transpiling and executing...")

        compiled_circuit = transpile(qc, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=shots)
        counts = job.result().get_counts()

        if debug:  # Sort and display the most frequent results
            sorted_counts = sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
            print("\n--- Top Measurement Results ---")
            for state, count in sorted_counts[:5]:
                print(f"State: |{state}> -> {count} occurrences ({count / shots:.1%})")

        return counts

    @classmethod
    def execute(
        cls,
        oracle: QuantumCircuit,
        num_solutions: int = 1,
        shots: int = 1024,
        debug: bool = False,
    ) -> dict:
        """
        A convenience method to execute Grover's algorithm without explicitly creating
        an instance.

        Args:
            oracle (QuantumCircuit): The quantum circuit implementing the oracle.
            num_solutions (int): The expected number of valid solutions (marked states).
            shots (int): Number of shots for the simulation.
            debug (bool): If True, prints detailed debug information.

        Returns:
            dict: The measurement counts.
        """
        grover_instance = cls(oracle, num_solutions)
        return grover_instance.search(shots=shots, debug=debug)
