# TFG: Study of Quantum Cryptological Algorithms

A repository containing the implementations, simulations and notebooks developed for the final-year project on quantum computing, quantum cryptography and related quantum algorithms: *Study of quantum cryptological algorithms*.

The project combines introductory material, demonstrations in Qiskit and Python utilities for studying algorithms such as Deutsch–Jozsa, Bernstein–Vazirani, Grover, QFT, QPE and Shor.

## Contents

- `notebooks/`: explanatory and experimental notebooks organised by topic.
- `src/algorithms/`: reusable Python implementations for the project’s algorithms and utilities.

### Notebooks

- `0_introduction/`: an introduction to Qiskit and the basics.
- `1_deutsch-jozsa/`: implementation of the Deutsch-Jozsa algorithm.
- `2_bernstein-vazirani/`: implementation of the Bernstein-Vazirani algorithm and cached results.
- `3_quantum_primitives/`: quantum gates and subroutines such as QFT and QPE.
- `4_shor/`: implementation of Shor’s algorithm.
- `5_grover/`: Grover’s algorithm and applications.

## Requirements

- Python 3.11
- Poetry
- Qiskit and associated scientific dependencies

The main dependencies are defined in [pyproject.toml](pyproject.toml).

## Installation

```bash
poetry install
```

If you want to open the notebooks with the project environment:

```bash
poetry run python -m ipykernel install --user --name tfg-quantum-cryptographic-protocols
```

## Usage

1. Install the dependencies.
2. Open the notebooks from the [notebooks/](notebooks/) folder.
3. Run the cells in the recommended order for each notebook.
4. To reuse logic from Python code, import from [src/algorithms/](src/algorithms/).

Example of using Poetry to launch a Python session:

```bash
poetry run python
```

## Project Structure

```text
.
├── docs/
├── notebooks/
├── src/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Objective of the Repository

This repository is designed as support for the final-year project, prioritizing didactic clarity, traceability of implementations, and reproducibility of experiments conducted with Qiskit.

## License

This project is distributed under the MIT license. See the [LICENSE](LICENSE) file for more details.
