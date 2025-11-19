# QuantumXORSolver 
Quantum SAT solving is a challenging task because Boolean satisfiability (SAT) belongs to the class of NP‑Complete problems. These problems are computationally hard: classical solvers often struggle to efficiently find solutions when constraints grow large, especially in cryptographic contexts. SAT in Conjunctive Normal Form (CNF) or with XOR constraints is widely used to model cryptographic primitives, making it directly relevant to stream cipher cryptanalysis.
Traditional SAT solvers can handle many cases but fail to exploit quantum parallelism. Our project builds on the theoretical foundation of Grover’s operator, which provides quadratic speed‑up for unstructured search, and extends it to hybrid Boolean oracles (AND/OR/XOR/NOT, ESOP, parity systems). By encoding SAT constraints into quantum circuits, we amplify valid solutions while suppressing non‑solutions, making the solver more reliable for solving CNF-XOR SAT .


