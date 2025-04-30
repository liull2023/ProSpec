# ProSpec: Plan Ahead, Then Execute

**Prospective thinking (PT)**—the human capacity to imagine future scenarios and plan accordingly—is central to efficient decision making. Traditional model-free reinforcement learning (RL) methods lack this foresight, often suffering from data inefficiency and “dead-end” state traps. **ProSpec** is the first RL framework to embed human-like prospective thinking into model-free agents, guiding them to **plan ahead** before **executing** actions.

---
## framwork 

![ProSpec](/res/prospec.png)

![ProSpec framework](/res/framework.png)

---

## 🚀 Key Contributions

1. **Flow-based Reversible Dynamics Model**  
   - Learns a reversible mapping from state–action pairs to next states.  
   - Generates _n_ candidate future trajectories from the current state and policy, enabling rich scenario anticipation.

2. **Prospective Planning Mechanism**  
   - Uses Model Predictive Control (MPC) with a **value consistency constraint**.  
   - Ensures the agent “plans ahead, then executes” by selecting actions whose predicted returns align with learned value estimates.  
   - Avoids high-risk “dead ends” by ruling out trajectories with low or inconsistent value.

3. **Cyclical Consistency Constraint**  
   - Enforces reversibility: rolling a predicted future state back to the original state recovers the start.  
   - Augments real data with accurate, reversible virtual trajectories—boosting data efficiency and stability.
<<<<<<< HEAD

---

## Results
![DMControl](https://raw.githubusercontent.com/liull2023/ProSpec/tree/master/res/dmc.jpg)

![Atari](https://raw.githubusercontent.com/liull2023/ProSpec/tree/master/res/atari.jpg)
=======
---

## Results
![DMControl](/res/dmc.png)

![Atari](/res/atari.png)

---

## 📈 Experimental Results

We benchmark ProSpec under limited environment interactions on both DMControl and Atari:

| Domain       | # Environments | ProSpec Wins | SOTA Wins |
|--------------|----------------|--------------|-----------|
| **DMControl**| 6              | **4**        | 2         |
| **Atari**    | 26             | **8**        | 18        |

ProSpec outperforms state-of-the-art baselines in 4/6 DMControl tasks and 8/26 Atari games, demonstrating its ability to plan ahead effectively under resource constraints.

---

## 🔧 Installation

```bash
# 1) Clone the repository
git clone https://github.com/anonymous/ProSpec.git
cd ProSpec

# 2) (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
