

#  Snake Game AI with Q-Learning

A classic Snake game where the snake learns to play using **Reinforcement Learning** via **Q-Learning**. This project visualizes the training and inference of an AI agent as it learns to collect food and avoid walls or self-collisions.

---
![Snake RL Demo](snke-rl.gif)
##  Features

- Game visualization using **Pygame**
- Reinforcement Learning with **Q-Learning**
- Custom environment with simple state representation
- Live visualization of agent's learned policy

---

##  Reinforcement Learning Setup

| Element      | Description                                |
|--------------|--------------------------------------------|
| **State**    | Relative direction to food, dangers around, current direction |
| **Actions**  | Up, Down, Left, Right                      |
| **Reward**   | +10 for eating food, -10 for dying, -0.1 per move |
| **Q-table**  | Dictionary mapping states to action values |
| **Policy**   | ε-greedy (exploration vs exploitation)     |

---

## 🛠 Installation

Make sure Python 3 is installed, then install the dependencies:

```bash
pip install pygame numpy


