# Dungeon Crawler RL Environment

A custom Gymnasium environment implementing a grid-based dungeon crawler game designed for tabular Reinforcement Learning algorithms.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gymnasium](https://img.shields.io/badge/gymnasium-0.29.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎮 Overview

This project implements a simplified dungeon crawler environment where an RL agent learns to navigate a 16×16 grid, avoid randomly moving enemies, and reach an exit door. The environment features **global vision** (full observability) and uses **tabular RL algorithms** (Q-Learning, SARSA, Expected SARSA) for training.

### Key Features

- **16×16 grid** with only border walls (no interior obstacles)
- **Global vision**: Agent observes the entire grid
- **2 mobile enemies** with random movement patterns
- **Sparse state space**: 38,416 states (agent position × door position)
- **Distance-based reward shaping** to guide learning
- **PyGame visualization** for real-time rendering
- **TensorBoard integration** for training metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dungeon-crawler-rl-1.0

# Install dependencies
pip install -r requirements.txt
```

### Train an Agent

```bash
# Train Q-Learning agent
python train.py --algorithm qlearning --episodes 5000 --run-name my_qlearning

# Train SARSA agent
python train.py --algorithm sarsa --episodes 5000 --run-name my_sarsa

# Train Expected SARSA agent
python train.py --algorithm expected_sarsa --episodes 5000 --run-name my_expected_sarsa
```

### Visualize Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Open your browser to http://localhost:6006
```

### Evaluate a Trained Agent

```bash
# Run 100 evaluation episodes
python evaluate.py --model models/my_qlearning/final_model.pkl --episodes 100
```

### Watch the Agent Play

```bash
# PyGame visualization with trained agent
python demo_pygame.py --model models/my_qlearning/final_model.pkl --fps 10

# Or play manually
python demo_pygame.py --manual
```

## 🏗️ Environment Details

### State Space

The environment uses a **combined position encoding**:
- **Agent position**: 196 possible positions (14×14 interior grid)
- **Door position**: 196 possible positions (randomized each episode)
- **Total states**: 196 × 196 = **38,416 states**

Enemy positions are **not** included in the state (they move randomly), making the environment partially stochastic.

### Action Space

Four discrete movement actions:
- `0`: Move UP
- `1`: Move DOWN
- `2`: Move LEFT
- `3`: Move RIGHT

### Reward Structure

| Event | Reward | Effect |
|-------|--------|--------|
| **Reach door (WIN)** | **+200.0** | Episode terminates (success) |
| Move closer to door | +1.0 | Positive reinforcement |
| Move away from door | -1.0 | Negative reinforcement |
| **Enemy collision (DEATH)** | **-100.0** | Episode terminates (failure) |
| Wall collision | -1.0 | Small penalty, continues |
| Each step | -0.1 | Efficiency encouragement |

### Episode Termination

Episodes end when:
1. **Agent reaches the door** (success)
2. **Agent collides with an enemy** (failure)
3. **Maximum steps reached** (300 steps, timeout)

## 🧠 Algorithms Implemented

### Q-Learning (Off-Policy)

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                              a'
```

- **Characteristics**: Optimistic, learns optimal policy regardless of behavior policy
- **Best for**: Faster convergence, aggressive exploration

### SARSA (On-Policy)

```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

- **Characteristics**: Conservative, learns policy being followed
- **Best for**: Safer learning in risky environments
- **Special requirement**: Needs next action selected during training

### Expected SARSA (Hybrid)

```
Q(s,a) ← Q(s,a) + α[r + γ Σ π(a'|s')Q(s',a') - Q(s,a)]
                              a'
```

- **Characteristics**: Balances optimism and conservatism
- **Best for**: Stable learning with reduced variance

## 📊 Training Configuration

### Default Hyperparameters

```python
learning_rate (α)      = 0.1
discount_factor (γ)    = 0.95
epsilon (initial)      = 1.0
epsilon_decay          = 0.995
epsilon_min            = 0.01
max_steps_per_episode  = 300
```

### Custom Training

```bash
python train.py --algorithm qlearning \
                --episodes 5000 \
                --alpha 0.2 \
                --gamma 0.99 \
                --epsilon-decay 0.997 \
                --run-name custom_experiment
```

### Curriculum Learning

Continue training from a checkpoint:

```bash
python train.py --algorithm qlearning \
                --episodes 3000 \
                --load-model models/my_qlearning/checkpoint_ep2000.pkl \
                --run-name continued_training
```

## 📈 Monitoring and Evaluation

### TensorBoard Metrics

Training logs include:
- Episode rewards (raw and 100-episode moving average)
- Success rate (percentage of episodes reaching the door)
- Episode length (steps taken)
- Q-table statistics (mean/max/min Q-values, states visited)
- Epsilon decay curve

### Evaluation Metrics

```bash
python evaluate.py --model models/my_qlearning/final_model.pkl \
                   --episodes 100 \
                   --verbose
```

Outputs:
- Success rate (% of episodes reaching door)
- Average reward ± standard deviation
- Average episode length
- Comparison of successful vs. failed episodes

## 🎯 Project Structure

```
dungeon-crawler-rl-1.0/
│
├── environment/              # Gymnasium environment
│   ├── dungeon_env.py       # Main environment implementation
│   └── render_pygame.py     # PyGame visual renderer
│
├── agents/                   # RL algorithms
│   ├── base_agent.py        # Base tabular agent (Q-table, ε-greedy)
│   ├── qlearning.py         # Q-Learning implementation
│   ├── sarsa.py             # SARSA implementation
│   └── expected_sarsa.py    # Expected SARSA implementation
│
├── utils/                    # Utilities
│   └── state_encoder.py     # Observation → state encoding
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── demo_pygame.py            # Interactive demo (manual/AI modes)
│
├── logs/                     # TensorBoard logs (auto-generated)
├── models/                   # Saved Q-tables (auto-generated)
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── CLAUDE.md                 # Developer documentation
```

## 🎮 Interactive Demo Controls

### Manual Mode

- **Arrow Keys**: Move agent (UP/DOWN/LEFT/RIGHT)
- **R**: Reset episode
- **M**: Toggle between Manual/AI mode
- **+/-**: Increase/decrease FPS (AI speed)
- **[/]**: Decrease/increase action cooldown (Manual speed)
- **Q or ESC**: Quit

```bash
# Play manually
python demo_pygame.py --manual

# Watch AI play
python demo_pygame.py --model models/my_qlearning/final_model.pkl
```

## 🔬 Testing

```bash
# Test environment
python -m environment.dungeon_env

# Test state encoder
python -m utils.state_encoder

# Test Q-Learning agent
python -m agents.qlearning

# Test SARSA agent
python -m agents.sarsa
```

## 📝 Implementation Notes

### Critical Design Decisions

1. **Global Vision**: Agent sees entire 16×16 grid (full observability)
2. **Sparse Q-table**: Uses `defaultdict` to only store visited states
3. **Combined State Encoding**: Encodes both agent AND door positions
4. **Random Enemy Movement**: Enemies move independently each step
5. **Border Walls Only**: No interior walls to simplify navigation

### SARSA Training Pattern

SARSA requires special handling during training because it needs the next action:

```python
# For SARSA: select initial action before loop
if algorithm == 'sarsa':
    action = agent.get_action(state, training=True)

while not done:
    if algorithm != 'sarsa':
        action = agent.get_action(state, training=True)

    next_state, reward, done = env.step(action)

    if algorithm == 'sarsa' and not done:
        next_action = agent.get_action(next_state, training=True)
    else:
        next_action = None

    agent.update(state, action, reward, next_state, next_action)

    state = next_state
    if algorithm == 'sarsa':
        action = next_action
```

## 📦 Dependencies

- `gymnasium>=0.29.0` - RL environment framework
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - PyTorch (for potential future extensions)
- `tensorboard>=2.13.0` - Training visualization
- `matplotlib>=3.7.0` - Plotting
- `tqdm>=4.65.0` - Progress bars
- `pygame>=2.5.0` - Visual rendering

## 🎓 Academic Context

This project was developed as an academic assignment for a Reinforcement Learning course:

- **Assignment Type**: Option B - Environment Design
- **Grade Weight**: 20% (16% complexity + 4% documentation)
- **Key Requirements**:
  - Custom Gymnasium environment with tractable state space
  - Multiple tabular RL algorithm implementations
  - Comprehensive documentation and analysis
  - Complexity justification with stochastic elements

## 📊 Expected Performance

Typical learning curves for 5000 training episodes:

- **Q-Learning**: Fast initial learning, potentially higher variance
- **SARSA**: More conservative learning, safer policy
- **Expected SARSA**: Balanced performance between Q-Learning and SARSA

Success rates vary significantly due to random enemy movements. Agents typically learn to navigate toward the door while avoiding enemies, with performance improving substantially after 1000-2000 episodes.

## 🤝 Contributing

This is an academic project. If you find bugs or have suggestions:

1. Check existing issues
2. Create a new issue with detailed description
3. Fork and submit a pull request (for bug fixes)

## 📄 License

This project is developed for academic purposes. See LICENSE file for details.

## 🙏 Acknowledgments

- Built using the [Gymnasium](https://gymnasium.farama.org/) framework
- Visualization powered by [PyGame](https://www.pygame.org/)
- Training monitoring with [TensorBoard](https://www.tensorflow.org/tensorboard)

## 📧 Contact

For questions about this project, please open an issue in the repository.

---

**Made with ❤️ for Reinforcement Learning education**
