# Wheeled Pupper v3 Core – JAX/MJX RL Environment

**The high-performance reinforcement learning "Brain" for the hybrid Wheeled Pupper v3 robot.**

This repository contains the core logic for training the Wheeled Pupper using **MuJoCo MJX** (simulated on GPU) and **JAX**. It features a custom PPO implementation designed to solve the hybrid control problem of combining legged articulation with wheeled locomotion.

## Key Technical Features

### 1. Hybrid Control Architecture (JAX)
Unlike standard quadruped environments, this project implements a **Split Control** scheme. We decoupled the legged joints (position control) from the wheel actuators (velocity control) directly in the physics step logic.

*   **Legs**: High-stiffness PD control for posture maintenance.
*   **Wheels**: Direct velocity command for smooth rolling.

### 2. Custom Reward Engineering
Achieving stable 0.75 m/s velocity tracking required designing a multi-objective reward function that balances tracking performance with energy efficiency.

```python
# pupperv3_mjx/config.py
rewards=config_dict.ConfigDict(
    dict(
        # Primary Objective: Stable Velocity Tracking
        tracking_lin_vel=1.5,
        tracking_ang_vel=0.8,
        
        # Energy Efficiency & Hardware Safety
        torques=-0.0002,           # Minimize motor heat overhead
        mechanical_work=-0.00,     # Minimize Cost of Transport (CoT)
        action_rate=-0.01,         # Prevent high-frequency jitter (smooth control)
        
        # Shaping for Stability
        foot_slip=-0.1,            # Critical for traction loss in wheeled mode
        orientation=-5.0,          # Penalize pitch/roll instability
    )
)
```

### 3. Sim-to-Real: Domain Randomization
To ensure the policy transfers to physical hardware, the environment implements aggressive domain randomization using JAX's `vmap` to simulate thousands of varied physics parameters in parallel.

```python
# pupperv3_mjx/domain_randomization.py
@jax.vmap
def rand(rng):
    # Randomize friction coefficients (0.6 to 1.4) to handle different floor types
    friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
    
    # 50% Randomization of Actuator Gains (Kp/Kd) to model motor wear/variance
    kp = jax.random.uniform(key_kp, (1,), minval=0.75, maxval=1.25) * sys.actuator_gainprm[:, 0]
    
    # Randomize Center of Mass (CoM) to account for battery/electronics placement
    body_com_shift = jax.random.uniform(key_com, (3,), minval=-0.03, maxval=0.03)
    
    return friction, gain, bias, body_com, ...
```

## Architecture Overview

*   **`environment.py`**: The `PupperV3Env` class inherits from `PipelineEnv`. It helps manage the MJX physics state, computes observations, and steps the simulation.
*   **`rewards.py`**: JIT-compiled reward functions calculated on the GPU for maximum throughput.
*   **`domain_randomization.py`**: Handles parameter sampling for robust policy training.

## Installation & Usage

This package is designed to be installed as a dependency for the [Training Colab](https://github.com/TundTT/colab_Wheel_pupperv3).

```bash
# Install with MJX support
pip install -e .
```

To run the environment loop:

```python
import jax
from pupperv3_mjx import environment, config

# Initialize the 8192 parallel environments on GPU
env = environment.PupperV3Env(
    path="path/to/Wheel_pupper.xml",
    reward_config=config.get_config(),
)

# Step function compiles to XLA for microsecond latency
state = env.reset(rng)
next_state = env.step(state, action)
```

## Acknowledgments
*   **Google DeepMind**: For the MuJoCo MJX physics engine.
*   **UW–Madison LeggedAI Lab**: Research support and hardware validation.

---
**Author**: [Tund Theerawit](https://www.linkedin.com/in/tund-theerawit) | [GitHub](https://github.com/TundTT)