# ProSpec: Value-Guided Prospective Action Filtering for Data-Efficient Model-Free Reinforcement Learning


**Prospective Thinking (PT)**—the human ability to envision future scenarios and plan accordingly—is key to efficient decisions. Conventional model-free RL lacks this foresight, leading to data inefficiency and “dead-end” traps. **ProSpec** injects human-like PT into model-free agents, enabling them to **plan ahead** and then **execute**.

<p align="center">
  <img src="res/prospec.png" alt="ProSpec Logo" height="480"/>
</p>


---

## 📖 Framework Overview

<p align="center">
  <img src="res/framework.png" alt="ProSpec Framework" width="800"/>
</p>

---

## 🚀 Key Contributions

1. **Flow-based Reversible Dynamics Model**  
   - Learns a bijective mapping between (state, action) and next state.  
   - Rolls out _n_ candidate trajectories from the current policy for rich future anticipation.

2. **Prospective Planning Mechanism**  
   - Model Predictive Control (MPC) with a **value consistency** constraint.  
   - “Plan ahead, then execute”: pick the action whose predicted return matches the value network.  
   - Prunes high-risk “dead-end” trajectories to avoid catastrophic failures.

3. **Cyclical Consistency Constraint**  
   - Enforces reversibility: rolling predicted future states back recovers the original.  
   - Generates abundant, accurate virtual trajectories—boosting data efficiency and stability.

---

## 📊 Experimental Results

<p align="center">
  <img src="res/dmc.png" alt="DMControl Results" width="700"/>
</p>
<p align="center">
  <img src="res/atari.jpg" alt="Atari Results" width="700"/>
</p>

| Benchmark     | Tasks | ProSpec Wins | SPR Wins | PlayVirtual Wins |RLASTIC Wins |
|--------------:|:-----:|:------------:|:---------:|:---------:|:---------:|
| **DMControl** | 6     | **4**        | 0        |1        |2        |
| **Atari**     | 26    | **8**        | 0       |1        |8        |

> Under limited interactions, ProSpec outperforms state-of-the-art on 4/6 DMControl tasks and 8/26 Atari games.

---

## 💻 Installation

```bash
# Clone the repository
[git clone https://github.com/anonymous/ProSpec.git](https://anonymous.4open.science/r/ProSpec-35B8)
cd ProSpec

# Create and activate a Conda environment
conda create -n ProSpec python=3.8 -y
conda activate ProSpec

# Install dependencies for your target domain:
# DMControl
pip install -r DMControl/requirements.txt

# Atari
pip install -r Atari/requirements.txt
