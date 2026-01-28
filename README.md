# Entangle: Quantum CNF-XOR SAT Solver

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-61DAFB?style=for-the-badge&logo=qiskit&logoColor=black)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-orange?style=for-the-badge)]()

> **A robust algorithmic synthesis framework for solving arbitary mixed CNF-XOR satisfiability problems using the scalable boolean oracle and Controlled Diffusion (CUs) operator.**

---

##  Overview

**Entangle** is a quantum logic synthesis  designed to solve complex combinatorial satisfiability (SAT) problems found in cryptanalysis. Unlike standard Grover implementations that struggle with phase instability in mixed constraints, Entangle utilizes a **Hybrid Boolean Oracle** and **Controlled Diffusion** to maintain unitary coherence.

###  Key Features
* **Automated Synthesis:** Compiles high-level constraint strings (e.g., `XOR(0,1,2)`) directly into linear-depth quantum circuits.
* **Phase Robustness:** Bypasses "Phase Kickback" errors in high-arity gates (like 5-input ORs) using Al-Bayaty’s $CU_s$ operator.
* **Stack-Based Uncomputation:** Automatically manages ancilla qubits to ensure zero residual entanglement ("garbage") in the final state.
* **Scalable:** Supports arbitrary clause lengths and complex topologies (Bridged, Interwoven, Cyclic).

---

## 🛠️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/anushkasaini39/QuantumCNF-XOR-SATSolver.git](https://github.com/anushkasaini39/QuantumCNF-XOR-SATSolver.git)
    cd QuantumCNF-XOR-SATSolver
    ```

2.  **Install Dependencies:**
    You need Python 3.8+ and the Qiskit SDK.
    ```bash
    pip install qiskit qiskit-aer matplotlib numpy
    ```

---

## Usage

To use the solver, simply initialize the `QuantumSATSolver` and pass a list of constraint strings. The solver handles all circuit construction, simulation, and plotting automatically.

### Quick Start Example

```python
# step 1- start
# Mode 1 = Default mode (solves for 3 node graph coloring)
# Mode 2 = Manual mode(to give input by yourself)
-----------------------------------------------------
eg.- Select Mode (1/2): 2
-----------------------------------------------------

# step 2- Example: 5 Qubits
enter number of variables : eg.5
# 1. Define the Problem Topology (Number of Variables)
-----------------------------------------------------
eg. Number of Variables: 4
-----------------------------------------------------
#step 3-  Define Constraints
# Format: "GATE(qubit_indices)"
constraints = [
    "OR(0, 1, 2, 3, 4)",       # High-Arity OR Gate
    "XOR(0, 1, 2, 3, 4)",      # Global Parity Check
    "AND(0, 4)"                # Boundary Condition
]
-----------------------------------------------------
eg. Enter Formula (e.g. XOR(0,1,2) OR(0,1) NOT(1,2)):
> XOR(0,1,2) XOR(1,2,3) XOR(0,3) OR(0,1,2,3)
----------------------------------------------------

![Histogram Result](images/histogram_f1.png)
![Circuit Diagram](images/circuit_f1.png)

