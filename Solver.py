import numpy as np
import matplotlib.pyplot as plt
import re
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import warnings

# --- CONFIGURATION ---
%matplotlib inline
warnings.filterwarnings('ignore')

class ResearchSuite:
    def __init__(self):
        self.backend = AerSimulator()

    # 1. CLASSICAL SOLVER (Ground Truth)
    def get_ground_truth(self, num_vars, constraints):
        solutions = []
        for i in range(2**num_vars):
            bin_str = format(i, f'0{num_vars}b')
            satisfied = True
            for g_type, qubits in constraints:
                # Get values for all involved qubits (reading from right-to-left index)
                vals = [int(bin_str[-(q+1)]) for q in qubits]

                if g_type == 'XOR':
                    # Universal XOR: Parity check (odd number of 1s = True)
                    if sum(vals) % 2 == 0: satisfied = False
                elif g_type == 'OR':
                    # Universal OR: True if ANY input is 1
                    if not any(vals): satisfied = False
                elif g_type == 'AND':
                    # Universal AND: True only if ALL inputs are 1
                    if not all(vals): satisfied = False
                elif g_type == 'NOT':
                    # Implies Inequality (A != B) for 2 inputs, or Negation for 1 input
                    if len(vals) == 2:
                        if vals[0] == vals[1]: satisfied = False
                    elif len(vals) == 1:
                        if vals[0] == 1: satisfied = False # NOT(A) means A must be 0

            if satisfied:
                solutions.append(bin_str)
        return solutions

    # 2. HYBRID BOOLEAN ORACLE (UNIVERSAL BUILDER)
    def build_universal_oracle(self, num_vars, constraints):
        var_reg = QuantumRegister(num_vars, 'var')
        clause_reg = QuantumRegister(len(constraints), 'clause')
        fqubit = QuantumRegister(1, 'fqubit')
        qc = QuantumCircuit(var_reg, clause_reg, fqubit, name="Boolean_Oracle")

        # --- HELPER: LOGIC GATE APPLIER ---
        def apply_logic(g_type, qubits, target):
            if g_type == 'XOR':
                # XOR is CNOT from every input to target
                for q in qubits:
                    qc.cx(var_reg[q], target)

            elif g_type == 'OR':
                # OR Logic: De Morgan's Law (NOT all zeros)
                # 1. Flip all inputs
                for q in qubits: qc.x(var_reg[q])
                # 2. MCX checks if all are now 1 (meaning original were 0)
                qc.mcx([var_reg[q] for q in qubits], target)
                # 3. Flip target (Result is 1 if inputs were NOT all 0)
                qc.x(target)
                # 4. Restore inputs
                for q in qubits: qc.x(var_reg[q])

            elif g_type == 'AND':
                # AND is standard Multi-Controlled X
                qc.mcx([var_reg[q] for q in qubits], target)

            elif g_type == 'NOT':
                # If 2 inputs: NOT(A,B) -> A != B (Same as XOR)
                if len(qubits) == 2:
                    qc.cx(var_reg[qubits[0]], target)
                    qc.cx(var_reg[qubits[1]], target)
                # If 1 input: NOT(A) -> A must be 0
                elif len(qubits) == 1:
                    qc.x(var_reg[qubits[0]]) # Flip A
                    qc.cx(var_reg[qubits[0]], target) # Copy to target
                    qc.x(var_reg[qubits[0]]) # Restore A

        # A. Compute Logic
        for i, (g_type, qubits) in enumerate(constraints):
            apply_logic(g_type, qubits, clause_reg[i])

        # B. Aggregate (Boolean Bit Flip)
        qc.mcx(clause_reg, fqubit[0])

        # C. Uncompute (Strictly Reversed)
        for i, (g_type, qubits) in reversed(list(enumerate(constraints))):
            apply_logic(g_type, qubits, clause_reg[i])

        return qc

    # 3. CONTROLLED DIFFUSION OPERATOR (CUs)
    def build_cus_diffuser(self, num_vars):
        var_reg = QuantumRegister(num_vars, 'var')
        fqubit = QuantumRegister(1, 'fqubit')
        qc = QuantumCircuit(var_reg, fqubit, name="CUs_Diffuser")

        # Standard H and X on inputs
        qc.h(var_reg)
        qc.x(var_reg)

        # --- THE NOVELTY: CONTROLLED REFLECTION ---
        qc.x(fqubit)
        qc.h(fqubit)
        qc.mcx(var_reg, fqubit) # Multi-controlled X using fqubit as target/control logic
        qc.h(fqubit)
        qc.x(fqubit)
        # ------------------------------------------

        qc.x(var_reg)
        qc.h(var_reg)
        return qc

    # 4. VISUALIZATION
    def plot_schematic(self, num_vars, oracle_circ, cus_circ):
        print("\n[FIG 1] Algorithm Schematic (CUs Architecture)...")
        plt.close('all')
        v = QuantumRegister(num_vars, 'var')
        c = QuantumRegister(oracle_circ.num_qubits - num_vars - 1, 'clause')
        f = QuantumRegister(1, 'fq')
        m = ClassicalRegister(num_vars, 'm')
        vis = QuantumCircuit(v, c, f, m)

        vis.h(v) # Initialization
        vis.barrier()
        vis.append(oracle_circ, list(v)+list(c)+list(f)) # Boolean Oracle
        vis.barrier()
        vis.append(cus_circ, list(v)+list(f)) # CUs Operator
        vis.measure(v, m)

        display(vis.decompose().draw(output='mpl', fold=40, style='clifford', scale=0.8))

    # 5. SIMULATION
    def run_cus_simulation(self, num_vars, constraints, n_iters):
        oracle = self.build_universal_oracle(num_vars, constraints)
        diffuser = self.build_cus_diffuser(num_vars)

        all_regs = oracle.qregs
        cbits = ClassicalRegister(num_vars, 'meas')
        qc = QuantumCircuit(*all_regs, cbits)

        # 1. Superposition
        qc.h(qc.qregs[0])

        # 2. Boolean Target Initialization
        pass

        # 3. Iteration
        for _ in range(n_iters):
            qc.append(oracle, qc.qubits[:oracle.num_qubits])
            qc.append(diffuser, list(qc.qregs[0]) + list(qc.qregs[2]))

        qc.measure(qc.qregs[0], cbits)
        return self.backend.run(transpile(qc, self.backend), shots=4096).result().get_counts()

    # 6. ANALYSIS
    def analyze_results(self, counts, true_solutions):
        total = sum(counts.values())
        keys = sorted(counts.keys())
        probs = [counts.get(k, 0) / total for k in keys]

        # Determine Status
        if not true_solutions:
            status = "(NO SOLUTION)"
            # colors = ['#FF3333' for _ in keys] # Red for noise
            colors = ['#1F77B4' for _ in keys]
        else:
            status = f"SOLVED ({len(true_solutions)} found)"
            # colors = ['#00CC66' if k in true_solutions else '#D3D3D3' for k in keys]
            colors = ['#1F77B4' if k in true_solutions else '#1F77B4' for k in keys]

        # Calculate SNR
        if true_solutions:
            signal = min([counts.get(s,0) for s in true_solutions])
            noise = max([v for k,v in counts.items() if k not in true_solutions] + [0.1])
            snr = 10 * np.log10((signal + 1e-9)/(noise + 1e-9))
        else:
            snr = 0.0

        print("\n===========================================================")
        print(f"   STATUS: {status}")
        print(f"   SOLUTIONS: {true_solutions if true_solutions else 'None'}")
        print("===========================================================")
        print(f"| Metric                      | Proposed CUs (Boolean Oracle)  |")
        print(f"|-----------------------------|--------------------------------|")
        print(f"| Total Probability Mass      | {sum(counts.get(s,0) for s in true_solutions)/total*100:6.2f}%                        |")
        print("|  on solution subspace       |                           |")
        print(f"| Selectivity (Contrast Ratio)| {snr:6.2f} dB                      |")
        print("===========================================================")

        # Plotting
        print("\n[FIG 2] Results Histogram...")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(keys, probs, color=colors)
        ax.set_title(f"Measurement of input qubits for Grover's algorithm\n({status})")
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        max_h = max(probs) if probs else 0.1
        ax.set_ylim(0, max_h * 1.25)
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, h, label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.show()

    # 7. MAIN FLOW
    def run_full_suite(self, num_vars, constraints):
        solutions = self.get_ground_truth(num_vars, constraints)
        M = len(solutions)
        N = 2**num_vars

        if M > 0: optimal_iters = max(1, int(np.pi/4 * np.sqrt(N/M)))
        else: optimal_iters = max(1, int(np.pi/4 * np.sqrt(N)))

        print(f"\n[INFO] Found {M} solutions. Running {optimal_iters} iterations.")
        self.plot_schematic(num_vars, self.build_universal_oracle(num_vars, constraints), self.build_cus_diffuser(num_vars))
        cus_counts = self.run_cus_simulation(num_vars, constraints, optimal_iters)
        self.analyze_results(cus_counts, solutions)

# 8. INPUT PARSER
def main():
    suite = ResearchSuite()
    print("--- SCALABLE CNF-XOR SAT SOLVER ---")
    print("1. 3-Node Graph Coloring (Default)")
    print("2. Manual Input")

    mode = input("\nSelect Mode (1/2): ").strip()

    if mode == '1':
        constrs = [('XOR', [0,1]), ('XOR', [1,2])]
        suite.run_full_suite(3, constrs)
    elif mode == '2':
        try:
            n_in = input("Number of Variables: ")
            if not n_in.isdigit(): raise ValueError("Vars must be an integer.")
            n = int(n_in)
            print("Enter Formula (e.g. XOR(0,1,2) OR(0,1) NOT(1,2)):")
            raw = input("> ").upper()
            constrs = []
            matches = re.findall(r"([A-Z]+)\(([\d,\s]+)\)", raw)
            for g, args in matches:
                clean_args = args.replace(" ", "")
                qubits = [int(x) for x in clean_args.split(',')]
                constrs.append((g, qubits))
            if not constrs: return
            print(f"Parsed: {constrs}")
            suite.run_full_suite(n, constrs)
        except Exception as e: print(f"Input Error: {e}")

if __name__ == "__main__":
    main() 
