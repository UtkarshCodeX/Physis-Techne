import numpy as np
import ast
import random
from itertools import product as iproduct

# ═══════════════════════════════════════════════════════
# SECTION 1 — UTILITY
# ═══════════════════════════════════════════════════════

def normalize(v):
    v = np.array(v, dtype=complex)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def fidelity(target, output):
    
    return float(abs(np.dot(target.conj(), output)) ** 2)

def memory_mb(n_qubits):
  
    dim = 2 ** n_qubits
    return dim * dim * 16 / 1e6   

# ═══════════════════════════════════════════════════════
# SECTION 2 — SINGLE QUBIT GATES  (2x2 matrices)
# ═══════════════════════════════════════════════════════

def HWP(theta):
  
    c, s = np.cos(2*theta), np.sin(2*theta)
    return np.array([[c, s], [s, -c]], dtype=complex)

def QWP(theta):
    
    c, s = np.cos(2*theta), np.sin(2*theta)
    return np.array([
        [c + 1j*s,  s - 1j*c],
        [s - 1j*c, -c - 1j*s]
    ], dtype=complex) / np.sqrt(2)

def PHASE(phi):

    return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)

# ═══════════════════════════════════════════════════════
# SECTION 3 — TWO QUBIT GATES  (4x4 matrices)
# ═══════════════════════════════════════════════════════



def BS_gate():

    return (1/np.sqrt(2)) * np.array([
        [1,  0,  0, 1j],
        [0,  1, 1j,  0],
        [0, 1j,  1,  0],
        [1j, 0,  0,  1]
    ], dtype=complex)


SINGLE_QUBIT_GATES = ["HWP", "QWP", "PHASE"]
TWO_QUBIT_GATES    = ["BS"]
ALL_GATES          = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES

# ═══════════════════════════════════════════════════════
# SECTION 4 — MEMORY-SAFE GATE APPLICATION
# ═══════════════════════════════════════════════════════

def apply_single_gate(state, gate_2x2, target, n_qubits):

    new_state = state.copy()
    dim = 2 ** n_qubits

   
    stride = 2 ** (n_qubits - 1 - target)

    for i in range(dim):
        
        if (i >> (n_qubits - 1 - target)) & 1 == 0:
            j = i | stride           
            a0 = state[i]            
            a1 = state[j]         
            new_state[i] = gate_2x2[0,0]*a0 + gate_2x2[0,1]*a1
            new_state[j] = gate_2x2[1,0]*a0 + gate_2x2[1,1]*a1

    return new_state


def apply_two_gate(state, gate_4x4, qa, qb, n_qubits):

    new_state = state.copy()
    dim = 2 ** n_qubits
    bit_a = n_qubits - 1 - qa
    bit_b = n_qubits - 1 - qb

    visited = set()

    for i in range(dim):
        
        base = i & ~(1 << bit_a) & ~(1 << bit_b)   

        if base in visited:
            continue
        visited.add(base)


        i00 = base
        i01 = base | (1 << bit_b)
        i10 = base | (1 << bit_a)
        i11 = base | (1 << bit_a) | (1 << bit_b)

        amps = np.array([state[i00], state[i01], state[i10], state[i11]])
        out  = gate_4x4 @ amps

        new_state[i00] = out[0]
        new_state[i01] = out[1]
        new_state[i10] = out[2]
        new_state[i11] = out[3]

    return new_state


def simulate(circuit, input_state):

    state     = input_state.copy().astype(complex)
    n_qubits  = int(np.log2(len(state)))

    GATE_FN = {
        "HWP":  lambda p: HWP(p),
        "QWP":  lambda p: QWP(p),
        "PHASE":lambda p: PHASE(p),
        
        "BS":   lambda p: BS_gate(),
    }

    for gate_name, param, qa, qb in circuit:
        if gate_name not in GATE_FN:
            continue
        g = GATE_FN[gate_name](param)

        if gate_name in SINGLE_QUBIT_GATES:
            state = apply_single_gate(state, g, qa, n_qubits)
        else:
            state = apply_two_gate(state, g, qa, qb, n_qubits)

    return normalize(state)

# ═══════════════════════════════════════════════════════
# SECTION 5 — SPDC SOURCES
# ═══════════════════════════════════════════════════════

def spdc_single():
 
    s = np.zeros(4, dtype=complex)
    s[0] = 1.0   # |00>
    s[3] = 1.0   # |11>
    
    return normalize(s)

def spdc_two_sources():

    bell = spdc_single()
    return normalize(np.kron(bell, bell))

def spdc_three_sources():

    bell = spdc_single()
    return normalize(np.kron(np.kron(bell, bell), bell))

def vacuum_state(n_qubits):

    s = np.zeros(2**n_qubits, dtype=complex)
    s[0] = 1.0
    return s

# ═══════════════════════════════════════════════════════
# SECTION 6 — DETECTORS AND POST-SELECTION
#================================================
def threshold_detector(state, qubit, n_qubits):

    dim = 2 ** n_qubits
    bit_pos = n_qubits - 1 - qubit

    
    prob_click = sum(
        abs(state[i])**2
        for i in range(dim)
        if (i >> bit_pos) & 1 == 1
    )

    prob_no_click = 1.0 - prob_click

    if prob_click < 1e-12:
        return 0, state, prob_no_click

    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        if (i >> bit_pos) & 1 == 1:
            new_state[i] = state[i]

    new_state = normalize(new_state)
    return 1, new_state, prob_click


def photon_number_detector(state, qubit, n_qubits):
  
    return threshold_detector(state, qubit, n_qubits)


def herald(state, herald_qubit, expected_outcome, n_qubits):

    outcome, collapsed, prob = threshold_detector(state, herald_qubit, n_qubits)

    if outcome != expected_outcome:
        return False, state, 0.0

  
    remaining_qubits = [q for q in range(n_qubits) if q != herald_qubit]
    new_dim = 2 ** len(remaining_qubits)
    new_state = np.zeros(new_dim, dtype=complex)

    bit_pos = n_qubits - 1 - herald_qubit

    new_idx = 0
    for i in range(2**n_qubits):
        herald_bit = (i >> bit_pos) & 1
        if herald_bit == expected_outcome:
            # Map remaining bits to new index
            bits = format(i, f'0{n_qubits}b')
            remaining_bits = ''.join(
                bits[q] for q in range(n_qubits) if q != herald_qubit
            )
            j = int(remaining_bits, 2)
            new_state[j] += collapsed[i]

    return True, normalize(new_state), prob


def post_select(state, measurements, n_qubits):

    current_state   = state.copy()
    current_nqubits = n_qubits
    total_prob      = 1.0
    removed         = 0

    for qubit, expected in sorted(measurements.items()):
        adjusted_qubit = qubit - removed
        success, current_state, prob = herald(
            current_state, adjusted_qubit,
            expected, current_nqubits
        )
        if not success:
            return False, state, 0.0
        total_prob      *= prob
        current_nqubits -= 1
        removed         += 1

    return True, current_state, total_prob

# ═══════════════════════════════════════════════════════
# SECTION 7 — RANDOM CIRCUIT GENERATOR
# ═══════════════════════════════════════════════════════

def random_circuit(length=10, n_qubits=4):
    circuit = []
    qubits  = list(range(n_qubits))

    for _ in range(length):
        gate = random.choice(ALL_GATES)

        if gate in SINGLE_QUBIT_GATES:
            qa    = random.choice(qubits)
            qb    = None
            param = np.random.uniform(0, np.pi)
        else:
            qa, qb = random.sample(qubits, 2)
            param  = None

        circuit.append((gate, param, qa, qb))

    return circuit

# ═══════════════════════════════════════════════════════
# SECTION 8 — ADAM OPTIMIZER
# ═══════════════════════════════════════════════════════

class Adam:
    def __init__(self, lr=0.08):
        self.lr = lr
        self.m  = {}
        self.v  = {}
        self.t  = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            if i not in self.m:
                self.m[i] = 0.0
                self.v[i] = 0.0
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*grads[i]
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grads[i]**2)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            params[i] += self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params

# ═══════════════════════════════════════════════════════
# SECTION 9 — PARAMETER REFINEMENT
# ═══════════════════════════════════════════════════════

def refine_parameters(circuit, target, input_state, steps=80, web_steps=20):
    
    param_indices = [
        (i, param)
        for i, (g, param, qa, qb) in enumerate(circuit)
        if param is not None
    ]

    if not param_indices:
        return circuit

    params     = [p for _, p in param_indices]
    opt        = Adam(lr=0.08)
    best_p     = params.copy()
    best_f     = 0.0
    no_improve = 0
    actual_steps = min(steps, web_steps)

    for _ in range(actual_steps):
        cur = list(circuit)
        for k, (ci, _) in enumerate(param_indices):
            g, _, qa, qb = circuit[ci]
            cur[ci] = (g, params[k], qa, qb)

        F = fidelity(target, simulate(cur, input_state))

        if F > best_f:
            best_f = F
            best_p = params.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > 15:
            break

        eps   = 1e-4  
        grads = []
        for k in range(len(params)):
            params[k] += eps
            cur2 = list(circuit)
            for j, (ci, _) in enumerate(param_indices):
                g, _, qa, qb = circuit[ci]
                cur2[ci] = (g, params[j], qa, qb)
            f2 = fidelity(target, simulate(cur2, input_state))
            grads.append((f2 - F) / eps)
            params[k] -= eps

        params = opt.step(params, grads)

    final = list(circuit)
    for k, (ci, _) in enumerate(param_indices):
        g, _, qa, qb = circuit[ci]
        final[ci] = (g, best_p[k], qa, qb)
    return final

# ═══════════════════════════════════════════════════════
# SECTION 10 — MUTATION AND CROSSOVER
# ═══════════════════════════════════════════════════════

def mutate(circuit, n_qubits, mutation_rate=0.2):
    new_circuit = []
    for gate, param, qa, qb in circuit:
        if random.random() < mutation_rate:
            new_gate = random.choice(ALL_GATES)
            if new_gate in SINGLE_QUBIT_GATES:
                new_circuit.append((
                    new_gate,
                    np.random.uniform(0, np.pi),
                    random.randint(0, n_qubits-1),
                    None
                ))
            else:
                new_circuit.append((
                    new_gate, None,
                    *random.sample(range(n_qubits), 2)
                ))
        elif param is not None and random.random() < 0.3:
            # Small angle tweak
            new_circuit.append((gate, param + np.random.normal(0, 0.2), qa, qb))
        else:
            new_circuit.append((gate, param, qa, qb))
    return new_circuit

def crossover(c1, c2):
    if len(c1) < 2:
        return c1
    point = random.randint(1, len(c1)-1)
    return c1[:point] + c2[point:]

# ═══════════════════════════════════════════════════════
# SECTION 11 — GENETIC SEARCH
# ═══════════════════════════════════════════════════════

def genetic_search(target, input_state,
                   population=80,
                   generations=100,
                   circuit_length=12,
                   log_queue=None,
                   web_mode=False):

    n_qubits = int(np.log2(len(input_state)))

    # Log helper — sends to queue if web mode, else prints normally
    def _log(msg):
        if log_queue is not None:
            log_queue.put(str(msg))
        else:
            print(msg)

    mem = memory_mb(n_qubits)
    if mem > 500:
        _log(f"WARNING: {n_qubits} qubits would need {mem:.0f}MB for full matrix.")
        _log("Using memory-safe gate application instead. Safe!")

    _log(f"Searching...")
    _log(f"Qubits: {n_qubits} | Pop: {population} | Gens: {generations} | Circuit length: {circuit_length}")
    _log("-" * 55)

    pop        = [random_circuit(circuit_length, n_qubits) for _ in range(population)]
    best_f     = 0.0
    best_circ  = None
    stagnation = 0

    for gen in range(generations):

        scored = []
        for circ in pop:
            circ  = refine_parameters(circ, target, input_state, web_steps=20 if web_mode else 80)
            out   = simulate(circ, input_state)
            F     = fidelity(target, out)
            scored.append((F, circ))

        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best = scored[0][0]

        _log(f"Gen {gen:3d} | Fidelity: {gen_best:.4f} | Best ever: {best_f:.4f}")

        if gen_best > best_f:
            best_f    = gen_best
            best_circ = scored[0][1]
            stagnation = 0
        else:
            stagnation += 1

        if best_f > 0.999:
            _log("Perfect circuit found!")
            break

        if stagnation > 15:
            _log(f"  Stuck — injecting fresh circuits...")
            stagnation = 0
            fresh = [random_circuit(circuit_length, n_qubits)
                     for _ in range(population // 3)]
            pop = [c for _, c in scored[:population//2]] + fresh
            continue

        elites  = [c for _, c in scored[:5]]
        parents = [c for _, c in scored[:population//2]]
        new_pop = elites.copy()

        while len(new_pop) < population:
            r = random.random()
            if r < 0.5:
                child = crossover(random.choice(parents), random.choice(parents))
                child = mutate(child, n_qubits, 0.15)
            elif r < 0.8:
                child = mutate(random.choice(parents), n_qubits, 0.25)
            else:
                child = random_circuit(circuit_length, n_qubits)
            new_pop.append(child)

        pop = new_pop

    return best_circ, best_f

# ═══════════════════════════════════════════════════════
# SECTION 12 — TARGET STATES
# ═══════════════════════════════════════════════════════

def ghz_4():
  
    s = np.zeros(16, dtype=complex); s[0]=1; s[15]=10; s[12]=6
    return normalize(s)

def w_state_4():

    s = np.zeros(16, dtype=complex); s[1]=1; s[2]=1; s[4]=1; s[8]=1
    return normalize(s)

def cluster_4():
    
    s = np.zeros(16, dtype=complex); s[0]=1; s[3]=1; s[12]=1; s[15]=-1
    return normalize(s)

def bell_state(s):
    return normalize(s)

# ═══════════════════════════════════════════════════════
# SECTION 13 — PRETTY PRINT CIRCUIT
# ═══════════════════════════════════════════════════════

def print_circuit(circuit):
    print("\nBest circuit found:")
    print("-" * 50)
    for i, (gate, param, qa, qb) in enumerate(circuit):
        if param is not None:
            print(f"  Step {i+1:2d}: {gate:6s} on qubit {qa}"
                  f"  angle={param:.3f} rad ({np.degrees(param):.1f}°)")
        else:
            print(f"  Step {i+1:2d}: {gate:6s} on qubits {qa} → {qb}")
    print("-" * 50)

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  QUANTUM OPTICAL EXPERIMENT DESIGNER")
    print("         BY MLX3 TEAM")
    print("=" * 55)


    print("\n── SPDC Sources ──")
    bell = spdc_single()
    print(f"1 SPDC source  → Bell state: {np.round(bell, 3)}")

    two_bell = spdc_two_sources()
    print(f"2 SPDC sources → Bell×Bell:  {np.round(two_bell[:4], 3)}...")

    
    print("\n── Detector Demo ──")
    test_state = normalize(np.array([1,0,0,1], dtype=complex))  # Bell state
    outcome, collapsed, prob = threshold_detector(test_state, qubit=0, n_qubits=2)
    print(f"Bell state measured on qubit 0:")
    print(f"  Outcome: {outcome} | Probability: {prob:.3f}")
    print(f"  Collapsed state: {np.round(collapsed, 3)}")

  
    print("\n── Heralding Demo ──")
    two_bell = spdc_two_sources()
    success, heralded, prob = herald(two_bell, herald_qubit=0,
                                     expected_outcome=1, n_qubits=4)
    print(f"Herald qubit 0 of 2xSPDC state:")
    print(f"  Success: {success} | Probability: {prob:.3f}")
    print(f"  Remaining state (3 qubits): {np.round(heralded[:4], 3)}...")

 
    print("\n── Memory Safety Check ──")
    for n in [4, 8, 12, 16, 20]:
        mb = memory_mb(n)
        safe = "SAFE (vector only)" if True else f"DANGER: {mb:.0f}MB matrix"
        print(f"  {n:2d} qubits: full matrix = {mb:10.1f} MB  → {safe}")

  
    print("\nStarting from 2 SPDC sources (Bell × Bell)")

    #alok input
    user_permission = input("Would you like to provide a custom input state? (Y/N):")
    user_input = None
    input_list=None
    if(user_permission=='Y' or user_permission=='y'):
        user_input = input('Enter your input list:')
        input_list = ast.literal_eval(user_input)
    else:
        pass
    user_target = input('Enter your target array:')
    
    target_list = ast.literal_eval(user_target)

    if(len(target_list)==4):
        alok_qubit=2
       
        if(user_permission=='Y' or user_permission=='y'):
             input_state = normalize(np.array(input_list, dtype=complex))
        else:
            input_state = spdc_single()
        

        target = bell_state(target_list)
        



    #aaaaaaaa============================

    elif(len(target_list)==16):
        alok_qubit=4
        if(user_permission=='Y' or user_permission=='y'):
            input_state = normalize(np.array(input_list, dtype=complex))
   
             
        else:
            input_state = spdc_two_sources() 

  
        target = normalize(np.array(target_list, dtype=complex))

 
    elif(len(target_list)==8):
        alok_qubit = int(np.log2(len(target_list)))
        if(user_permission=='Y' or user_permission=='y'):
            input_state = normalize(np.array(input_list, dtype=complex))
        else:
            input_state = vacuum_state(alok_qubit)
       
        target = normalize(np.array(target_list, dtype=complex))
    else:
        if(user_permission=='Y' or user_permission=='y'):
            input_state= normalize(np.array(input_list, dtype=complex))
        else:
            print("Please give input to the function for this.")
            exit()
        target = normalize(np.array(target_list, dtype=complex))



    print("Target state loaded from user input")

    circuit, F = genetic_search(
        target       = target,
        input_state  = input_state,
        population   = 80,
        generations  = 100,
        circuit_length = 12
    )

    print_circuit(circuit)
    print(f"\nFinal Fidelity: {F:.6f}")

    if   F > 0.999: print("PERFECT — experiment found!")
    elif F > 0.99:  print("EXCELLENT — nearly perfect!")
    elif F > 0.90:  print("GOOD — try more generations")
    else:           print("NEEDS IMPROVEMENT — try more population/generations")

 
    print("\n── Post-Selection Example ──")
    out_state = simulate(circuit, input_state)
    print("Applying post-selection: require qubit 0 = 1 AND qubit 1 = 0")
    success, final_state, prob = post_select(
        out_state,
        measurements={0: 1, 1: 0},

        n_qubits=alok_qubit
    )
    print(f"Post-selection success: {success}")
    print(f"Success probability:    {prob:.4f}")
    if success:
        print(f"Final state (2 qubits): {np.round(final_state, 3)}")