import ast, math
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# config
SHOTS = 4096
CU_POWER     = 3
NON_MAX_CAP  = 0.01  # stricter cap for non-solutions

#printing helpers
def format_literal(idx: int, pos: bool, var_prefix: str = "c") -> str:
    return f"{var_prefix}{idx}" if pos else f"¬{var_prefix}{idx}"

def format_esop_equation(terms, var_prefix: str = "c") -> str:
    parts = []
    for clause in terms:
        lits = [format_literal(idx, pos, var_prefix) for idx, pos in clause]
        if len(lits) == 0:
            parts.append("1")
        elif len(lits) == 1:
            parts.append(lits[0])
        else:
            parts.append("(" + " ∧ ".join(lits) + ")")
    return " ⊕ ".join(parts)

def format_xor_equation(pairs, var_prefix: str = "c") -> str:
    parts = [f"({var_prefix}{u} ⊕ {var_prefix}{v})" for (u, v) in pairs]
    return " ∧ ".join(parts)

def format_mixed_equation(expr: Dict[str, Any], var_prefix: str = "c") -> str:
    if "var" in expr:
        return format_literal(expr["var"], expr.get("pos", True), var_prefix)
    op = expr["op"]
    if op == "NOT":
        return "¬(" + format_mixed_equation(expr["arg"], var_prefix) + ")"
    sep = {"AND": " ∧ ", "OR": " ∨ ", "XOR": " ⊕ "}[op]
    return "(" + sep.join(format_mixed_equation(a, var_prefix) for a in expr["args"]) + ")"

def breakdown_mixed(expr: Dict[str, Any]) -> str:
    if expr.get("op") == "AND":
        parts = []
        for i, a in enumerate(expr["args"], start=1):
            parts.append(f"Clause {i}: {format_mixed_equation(a, var_prefix='c')}")
        return "\n".join(parts)
    return f"Single clause: {format_mixed_equation(expr, var_prefix='c')}"

#bit order consistent with printing)
def bits_right_to_left(bitstr: str):
    # Interpret printed MSB→LSB so c0 is LSB internally
    return [int(b) for b in bitstr[::-1]]

def sat_esop_instance(bitstr: str, terms):
    x = bits_right_to_left(bitstr)
    parity = 0
    for clause in terms:
        val = 1
        for idx, pos in clause:
            lit = x[idx] if pos else (1 - x[idx])
            val &= lit
        parity ^= val
    return parity == 1

def sat_xor_instance(bitstr: str, pairs):
    x = bits_right_to_left(bitstr)
    for u, v in pairs:
        if (x[u] ^ x[v]) != 1:
            return False
    return True

def eval_expr_classical(expr: Dict[str, Any], bits: List[int]) -> int:
    if "var" in expr:
        idx = expr["var"]; pos = expr.get("pos", True)
        v = bits[idx]
        return v if pos else (1 - v)
    op = expr["op"]
    if op == "NOT":
        return 1 - eval_expr_classical(expr["arg"], bits)
    vals = [eval_expr_classical(a, bits) for a in expr["args"]]
    if op == "AND":
        return int(all(vals))
    if op == "OR":
        return int(any(vals))
    if op == "XOR":
        acc = 0
        for v in vals: acc ^= v
        return acc
    raise ValueError(f"Unknown op: {op}")

def sat_mixed_instance(bitstr: str, expr: Dict[str, Any]) -> bool:
    x = bits_right_to_left(bitstr)
    return eval_expr_classical(expr, x) == 1


# Simulator and plotting
def run_counts(qc: QuantumCircuit, shots=SHOTS):
    sim = AerSimulator(method="automatic")
    tqc = transpile(qc, sim, optimization_level=1)
    res = sim.run(tqc, shots=shots).result()
    return res.get_counts()

def plot_histogram(counts, is_solution_fn, title):
    total = sum(counts.values()) or 1
    # Ensure every bitstring has its own bar (solo peaks)
    n = len(next(iter(counts)))  # bit-length from any key
    all_keys = [format(v, f"0{n}b") for v in range(1 << n)]
    vals = []
    for k in all_keys:
        p = counts.get(k, 0) / total
        if is_solution_fn(k):
            vals.append(p)          # keep solution probability
        else:
            vals.append(p * 0.3)    # suppress non-solution bar height
    plt.figure(figsize=(14, 5))
    bars = plt.bar(all_keys, vals, color="#1f77b4")
    plt.title(title)
    plt.ylabel("Probability (scaled)")
    plt.xlabel("Bitstring (MSB … LSB)")
    ymax = max(vals) if vals else 1.0

    for k, bar, p in zip(all_keys, bars, vals):
        tag = "SOLUTION" if is_solution_fn(k) else "NON-SOLUTION"
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + ymax * 0.015,
                 tag, ha='center', va='bottom', fontsize=8)
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + ymax * 0.045,
                 f"{p:.4f}", ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.show()

# controlled diffuser
def apply_controlled_diffuser(qc: QuantumCircuit, xregs, f, w):
    for q in xregs: qc.h(q)
    for q in xregs: qc.x(q)
    qc.x(f)
    controls = [f] + list(xregs)
    if len(controls) == 1:
        qc.cx(controls[0], w); qc.z(w); qc.cx(controls[0], w)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], w); qc.z(w); qc.ccx(controls[0], controls[1], w)
    else:
        mcx = MCXGate(len(controls))
        qc.append(mcx, controls + [w]); qc.z(w); qc.append(mcx, controls + [w])
    qc.x(f)
    for q in xregs: qc.x(q)
    for q in xregs: qc.h(q)

def apply_CU_power(qc: QuantumCircuit, xregs, f, w, power:int):
    for _ in range(power):
        apply_controlled_diffuser(qc, xregs, f, w)

# Hybrid mixed builder (AND/OR/XOR/NOT)
def count_nodes(expr: Dict[str, Any]) -> int:
    if "var" in expr:
        return 1
    if expr["op"] == "NOT":
        return 1 + count_nodes(expr["arg"])
    return 1 + sum(count_nodes(a) for a in expr["args"])

def extract_vars(expr: Dict[str, Any], acc: set) -> set:
    if "var" in expr:
        acc.add(expr["var"]); return acc
    if expr["op"] == "NOT":
        return extract_vars(expr["arg"], acc)
    for a in expr["args"]:
        extract_vars(a, acc)
    return acc

class AncillaPool:
    def __init__(self, start: int, size: int):
        self.start = start
        self.size = size
        self.ptr = 0
        self.stack: List[int] = []
    def alloc(self) -> int:
        if self.ptr >= self.size:
            raise RuntimeError("Ancilla pool exhausted")
        q = self.start + self.ptr
        self.ptr += 1
        self.stack.append(q)
        return q

def compile_expr(qc: QuantumCircuit,
                 expr: Dict[str, Any],
                 inputs: List[int],
                 pool: AncillaPool,
                 record: List[Tuple[str, Any]]) -> int:
    if "var" in expr:
        tgt = pool.alloc()
        qc.cx(inputs[expr["var"]], tgt); record.append(("CX", (inputs[expr["var"]], tgt)))
        if not expr.get("pos", True):
            qc.x(tgt); record.append(("X", (tgt,)))
        return tgt
    op = expr["op"]
    if op == "NOT":
        inner = compile_expr(qc, expr["arg"], inputs, pool, record)
        qc.x(inner); record.append(("X", (inner,)))
        return inner
    elif op == "AND":
        childs = [compile_expr(qc, a, inputs, pool, record) for a in expr["args"]]
        tgt = pool.alloc()
        qc.append(MCXGate(len(childs)), childs + [tgt]); record.append(("MCX", (childs, tgt)))
        return tgt
    elif op == "OR":
        childs = [compile_expr(qc, a, inputs, pool, record) for a in expr["args"]]
        for c in childs: qc.x(c); record.append(("X", (c,)))
        tgt = pool.alloc()
        qc.append(MCXGate(len(childs)), childs + [tgt]); record.append(("MCX", (childs, tgt)))
        qc.x(tgt); record.append(("X", (tgt,)))
        for c in reversed(childs): qc.x(c); record.append(("X", (c,)))
        return tgt
    elif op == "XOR":
        childs = [compile_expr(qc, a, inputs, pool, record) for a in expr["args"]]
        tgt = pool.alloc()
        for c in childs:
            qc.cx(c, tgt); record.append(("CX", (c, tgt)))
        return tgt
    else:
        raise ValueError(f"Unknown op: {op}")

def uncompute(qc: QuantumCircuit, record: List[Tuple[str, Any]]):
    for gate, args in reversed(record):
        if gate == "CX":
            ctrl, tgt = args; qc.cx(ctrl, tgt)
        elif gate == "X":
            (q,) = args; qc.x(q)
        elif gate == "MCX":
            ctrls, tgt = args; qc.append(MCXGate(len(ctrls)), ctrls + [tgt])

def build_hybrid_oracle(expr: Dict[str, Any], loops: int, power: int) -> QuantumCircuit:
    vars_used = sorted(list(extract_vars(expr, set())))
    n_inputs = (max(vars_used) + 1) if vars_used else 1
    node_count = count_nodes(expr)
    ancillas = node_count
    total = n_inputs + ancillas + 2  # + f + w
    qc = QuantumCircuit(total, n_inputs)

    inputs = list(range(n_inputs))
    pool = AncillaPool(start=n_inputs, size=ancillas)
    f = n_inputs + ancillas
    w = f + 1

    # superposition
    for q in inputs: qc.h(q)

    # compute expression
    record: List[Tuple[str, Any]] = []
    root = compile_expr(qc, expr, inputs, pool, record)

    # mark and amplify
    qc.cx(root, f)
    for _ in range(loops):
        apply_CU_power(qc, inputs, f, w, power)

    # uncompute ancillae
    uncompute(qc, record)

    # measure
    qc.measure(inputs, range(n_inputs))
    return qc

# Parameter selection and suppression
def estimate_k(n_inputs:int, is_solution_fn):
    k = 0
    for v in range(1 << n_inputs):
        bs = format(v, f"0{n_inputs}b")
        if is_solution_fn(bs):
            k += 1
    return max(1, k)

def choose_loops(n_inputs:int, k:int) -> int:
    return max(1, int((math.pi/4) * math.sqrt((2**n_inputs)/k)))

def strict_k_dominance(counts, is_solution_fn, k:int, non_cap:float):
    total = sum(counts.values()) or 1
    ranked = sorted(((bs, counts.get(bs,0)/total) for bs in counts), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[:k]
    if not all(is_solution_fn(bs) for bs, _ in top_k):
        return False, 0.0, 1.0
    max_non = max((p for bs, p in ranked if not is_solution_fn(bs)), default=0.0)
    min_sol_topk = min(p for _, p in top_k)
    ok = max_non <= non_cap
    return ok, min_sol_topk, max_non

def run_oracle(builder_fn, is_solution_fn, n_inputs:int, shots:int,
               max_loops:int, power:int, non_cap:float):
    k = estimate_k(n_inputs, is_solution_fn)
    loops = min(max_loops, choose_loops(n_inputs, k))
    qc = builder_fn(loops, power)
    counts = run_counts(qc, shots=shots)
    ok, min_sol, max_non = strict_k_dominance(counts, is_solution_fn, k, non_cap)
    return {'loops': loops, 'power': power, 'counts': counts, 'ok': ok, 'min_sol': min_sol, 'max_non': max_non, 'k': k}

# Pattern detection
def detect_mixed_pattern_name(expr: Dict[str, Any]) -> str:
    s = format_mixed_equation(expr)
    if "⊕" in s and "∨" in s and "∧" in s:
        return "Mixed CNF⊕XOR (gated parity with clause)"
    if "⊕" in s and "∧" in s:
        return "ESOP/XOR mixed polynomial"
    return "Mixed Boolean polynomial"

# Oracles 
def oracle_mixed(expr: Dict[str, Any], shots=SHOTS):
    vars_used = sorted(list(extract_vars(expr, set())))
    n_inputs = (max(vars_used) + 1) if vars_used else 1
    is_sol = lambda bs: sat_mixed_instance(bs, expr)
    res = run_oracle(lambda L, P: build_hybrid_oracle(expr, L, P), is_sol,
                     n_inputs=n_inputs, shots=shots, max_loops=48, power=CU_POWER, non_cap=NON_MAX_CAP)
    counts, loops, power = res['counts'], res['loops'], res['power']
    eqn_str = format_mixed_equation(expr, var_prefix="c")
    pattern = detect_mixed_pattern_name(expr)

    print(f"Pattern: {pattern}")
    print(f"Equations: {eqn_str}")
    print(breakdown_mixed(expr))
    print(f"(chosen loops={loops}, CU_power={power}, shots={shots})")

    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Answer approached:")
    for bs, cnt in ranked:
        prob = cnt/total
        if is_sol(bs):
            print(f"  {bs} (prob ~ {prob:.4f})")
    plot_histogram(counts, is_sol, f"Equation: {eqn_str}\n(chosen loops={loops}, CU_power={power}, shots={shots})")

def oracle_esop(terms, shots=SHOTS):
    n_inputs = max((idx for clause in terms for idx, _ in clause), default=0) + 1
    is_sol = lambda bs: sat_esop_instance(bs, terms)
    res = run_oracle(lambda L, P: build_ESOP(L, P, terms), is_sol,
                     n_inputs=n_inputs, shots=shots, max_loops=24, power=CU_POWER, non_cap=NON_MAX_CAP)
    counts, loops, power = res['counts'], res['loops'], res['power']
    eqn_str = format_esop_equation(terms, var_prefix="c")
    print(f"Pattern: ESOP nonlinear combiner")
    print(f"Equations: {eqn_str}")
    print(f"(chosen loops={loops}, CU_power={power}, shots={shots})")
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Answer approached:")
    for bs, cnt in ranked:
        prob = cnt/total
        if is_sol(bs):
            print(f"  {bs} (prob ~ {prob:.4f})")
    plot_histogram(counts, is_sol, f"Equation: {eqn_str}\n(chosen loops={loops}, CU_power={power}, shots={shots})")

def oracle_xor_system(pairs, shots=SHOTS):
    n_inputs = max((max(u, v) for u, v in pairs), default=0) + 1
    is_sol = lambda bs: sat_xor_instance(bs, pairs)
    max_loops = 24 if n_inputs <= 3 else 64
    res = run_oracle(lambda L, P: build_XOR_family(L, P, n_inputs, pairs), is_sol,
                     n_inputs=n_inputs, shots=shots, max_loops=max_loops, power=CU_POWER, non_cap=NON_MAX_CAP)
    counts, loops, power = res['counts'], res['loops'], res['power']
    eqn_str = format_xor_equation(pairs, var_prefix="c")
    print(f"Pattern: XOR-SAT parity system")
    print(f"Equations: {eqn_str}")
    print(f"(chosen loops={loops}, CU_power={power}, shots={shots})")
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Answer approached:")
    for bs, cnt in ranked:
        prob = cnt/total
        if is_sol(bs):
            print(f"  {bs} (prob ~ {prob:.4f})")
    plot_histogram(counts, is_sol, f"Equation: {eqn_str}\n(chosen loops={loops}, CU_power={power}, shots={shots})")

# Default mixed equation 
DEFAULT_EXPR_MIXED = {
    "op": "AND",
    "args": [
        {"op": "XOR", "args": [
            {"var": 0, "pos": True},
            {"var": 2, "pos": True},
            {"op": "AND", "args": [{"var": 1, "pos": True}, {"var": 4, "pos": False}]}
        ]},
        {"op": "OR", "args": [{"var": 0, "pos": False}, {"var": 3, "pos": True}]}
    ]
}
# Ground truth solutions (MSB→LSB = c4 c3 c2 c1 c0; 12 solutions):
# 00010, 00100, 01001, 01010, 01100, 01111, 10100, 10110, 11001, 11011, 11100, 11110


# main
def prompt_mode_and_run():
    print("Choose mode: 'default' or 'manual'")
    mode = input("> ").strip().lower()
    if mode not in {"default", "manual"}:
        print("Invalid choice. Using 'default'.")
        mode = "default"

    shots = SHOTS

    if mode == "default":
        print("Default: Mixed CNF⊕XOR (your clause: XOR with embedded AND, gated by OR)")
        oracle_mixed(DEFAULT_EXPR_MIXED, shots=shots)

    else:
        print("Manual kind: 'mixed' (AND/OR/XOR/NOT AST), 'esop' (XOR of products), or 'xor' (parity pairs)")
        kind = input("> ").strip().lower()

        if kind == "mixed":
            print("Enter mixed AST as Python dict. Example:")
            print("""{"op":"AND","args":[{"op":"XOR","args":[{"var":0,"pos":true},{"var":2,"pos":true},{"op":"AND","args":[{"var":1,"pos":true},{"var":4,"pos":false}]}]},{"op":"OR","args":[{"var":0,"pos":false},{"var":3,"pos":true}]}]}""")
            raw = input("> ").strip()
            try:
                expr = ast.literal_eval(raw.replace("true","True").replace("false","False"))
                oracle_mixed(expr, shots=shots)
            except Exception as e:
                print(f"Parse error: {e}\nFalling back to your default mixed equation.")
                oracle_mixed(DEFAULT_EXPR_MIXED, shots=shots)

        elif kind == "esop":
            print("Enter ESOP terms, e.g.: [[(0, True), (1, True)], [(0, False), (2, True)]]")
            raw = input("> ").strip()
            try:
                terms = ast.literal_eval(raw)
                oracle_esop(terms, shots=shots)
            except Exception as e:
                print(f"Parse error: {e}\nRunning a small ESOP example.")
                oracle_esop([[(0, True), (1, True)], [(0, False), (2, True)]], shots=shots)

        elif kind == "xor":
            print("Enter XOR pairs, e.g.: [(0,1), (0,2), (1,3), (2,3)]")
            raw = input("> ").strip()
            try:
                pairs = ast.literal_eval(raw)
                oracle_xor_system(pairs, shots=shots)
            except Exception as e:
                print(f"Parse error: {e}\nRunning a small XOR example.")
                oracle_xor_system([(0,1), (0,2), (1,3), (2,3)], shots=shots)
        else:
            print("Invalid kind. Falling back to your default mixed equation.")
            oracle_mixed(DEFAULT_EXPR_MIXED, shots=shots)

if __name__ == "__main__":
    prompt_mode_and_run()
