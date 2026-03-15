# Photonic Quantum Experiment Designer

> Bridging the gap between theoretical quantum information and practical optical implementations.

This project implements an AI-driven system for designing quantum optical experiments capable of generating entangled photonic states. The framework simulates multi-photon interference in a realistic linear-optical setup and searches for circuits that produce target quantum states with high fidelity and experimentally viable success probabilities.

---

## 🔬 The Simulation Engine

 * **Construction :** Models three Spontaneous Parametric Down-Conversion (SPDC) sources producing random six photons across twelve optical modes to be subjected to random tools provided on hand.
 * **State Space :** Operates within an 8074-dimensional "Fock space" restricted to a fixed photon number i.e. Each Mode n may have m no. of photons.
 * **Hardware Set :** Optical elements such as (BS)beam splitters, (HWPs)half-wave plates, (QWPs)quarter-wave plates, (PS)phase shifters, and photon detectors are used for such applied changes.
 * **Efficient :** State evolution is computed using `scipy.sparse.linalg.expm_multiply`, it stores the non zero terms and their locations in a 0 term heavy matrix hence saving computation space for efficiency.
---

## 🧠 Hybridized Model 

The system combines the global exploration capabilities of evolutionary search with the precision of gradient optimization to navigate the vast design space of multi-photon interference.

* **Genetic Algorithm (GA):** Performs global exploration over possible circuits. Circuits are randomly generated, recombined, and mutated while being evaluated against a target quantum state using pure-state fidelity calculation as mentioned in problem statement.
* **Balanced Fitness Function:** Evaluates circuits by balancing fidelity, success probability, and circuit complexity. This encourages experimentally meaningful and compact solutions rather than large impractical designs.
* **Gradient-Based Tuning (Adam):** Once promising circuit structures emerge, the system switches to Adam optimization. It applies the **parameter-shift rule** to compute exact gradients for continuous gate parameters (e.g., angles for our beam-splitters and phase shifts).
* **Exploration Diversity:** To prevent premature convergence, the algorithm includes mechanisms such as elite preservation, stochastic mutations, periodic population resets ("earthquakes"), and the delayed injection of known high-fidelity seed circuits.

---

## 🎯 Supported Targets & Capabilities
* **Entanglement Generation:** Currently supports the discovery of photonic circuits for generating **Bell states** and **four-photon GHZ states**, starting from entangled photon pairs.
* **Laboratory Viability:** The system evaluates circuits based on both quantum state fidelity and heralding probability, ensuring that the resulting designs remain physically meaningful for actual laboratory implementation.

---

## 📦 Installation

This project requires Python 3.8+ and relies on standard scientific computing libraries. No heavy quantum frameworks are required.
```bash
# Clone the repository
git clone [https://github.com/yourusername/photonic-quantum-designer.git](https://github.com/yourusername/photonic-quantum-designer.git)
cd photonic-quantum-designer

# Install dependencies
pip install numpy scipy
```

## 💻 Usage                                                                                                                          
```
# Generate a 4-photon GHZ state blueprint (Default)
python main.py --target ghz4

# Generate a 2-photon Bell state blueprint
python main.py --target bell

# Run with custom parameters for deeper exploration
python main.py --target ghz4 --pop 150 --gens 3000 --seed 42
```

## 🎯 | Copy & Paste
EmojiDB
https://emojidb.org › output-emojisOutput Interpretation
When a viable circuit is discovered, the engine will output the final hardware blueprint. This represents the physical optical table layout, including:

BS: Beam Splitters (for Hong-Ou-Mandel interference)

HWP/QWP: Half and Quarter-Wave Plates (for polarization rotation)

PS: Phase Shifters

DETECT: Photon number resolving detectors (for heralding and post-selection)
