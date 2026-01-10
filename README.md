# pupperv3_mjx

A JAX/MuJoCo-based Reinforcement Learning environment for the **Pupper V3 Quadruped Robot (Wheeled Variant)**.

This repository provides a high-performance, GPU-accelerated simulation environment powered by [Brax](https://github.com/google/brax) and [MuJoCo MJX](https://github.com/google-deepmind/mujoco). It is specifically engineered to handle the unique dynamics of a wheeled-legged robot, combining the versatility of legged locomotion with the efficiency of wheeled travel.

## Features

-   **Fully Differentiable**: Built on JAX for massive parallelization and gradient-based optimization.
-   **GPU Augmented**: Capable of simulating thousands of environments per second on a single GPU.
-   **Sim-to-Real Ready**: Includes domain randomization and actuator modeling to bridge the reality gap.
-   **Hybrid Control Scheme**: Custom physics implementation to support both position-controlled joints and velocity-controlled wheels.

## Key Contributions: Legged-to-Wheeled Conversion 🛞

This repository represents a significant evolution from the standard legged Pupper environment. The following core modifications were implemented to support the wheeled architecture:

### 1. Hybrid Control Architecture
Standard legged robots typically use position control (PD) for all joints. For the wheeled variant, we implemented a **Split Control, Split Actuation** logic:
-   **Leg Joints (Hips/Knees)**: Maintain accurate Position Control to hold posture and manage terrain adaptation. High stiffness (`kp=5.0`) and damping (`kd=0.25`) are applied.
-   **Wheel Joints**: Converted to pure Velocity Control. This allows for continuous, unrestricted rolling without fighting against position setpoints.
-   **Implementation**: Custom logic in `PupperV3Env` overrides standard Menagerie parameters, selectively applying gain/bias settings only to specific actuator indices (`leg_actuator_indices`), leaving wheel actuators free-spinning and velocity-driven.

### 2. Continuous Rotation & Infinite Horizon
-   **Zero-Windup Observation**: Standard joint sensors wrap or limit at $\pi$ radians. We modified the observation space to track **Wheel Velocity** instead of **Wheel Position**. This ensures the policy sees a consistent state even after the wheels have rotated millions of degrees.
-   **Limit Removal**: Removed joint limits for wheel actuators in the physics engine to enable true continuous rotation.

### 3. Hybrid Observation Space
The policy input was redesigned to accommodate the dual nature of the robot:
-   **Inputs**: `[IMU Data, Commands, Desired Orientation, Motor State, Last Action]`
-   **Motor State**: A hybrid vector containing:
    -   *Legs*: Joint Angles (Position) relative to default pose.
    -   *Wheels*: Joint Velocities (rad/s).
    
This allows the RL agent to learn stable standing/walking behaviors via the legs while simultaneously learning smooth driving behaviors via the wheels.

## Installation

### Prerequisites
-   Python 3.10+
-   CUDA-enabled GPU (recommended for training)

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/pupperv3_mjx.git
    cd pupperv3_mjx
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct JAX version installed for your CUDA version.*

3.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

### Instantiating the Environment
The environment can be loaded directly using the `PupperV3Env` class.

```python
import jax
from pupperv3_mjx import environment
from pupperv3_mjx import config

# Load default reward configuration
env_config = config.get_config()

# Initialize the environment
env = environment.PupperV3Env(
    path="path/to/your/Wheel_pupper.xml", # Ensure you point to your local MJCF
    reward_config=env_config,
    action_scale=0.5,
    observation_history=15
)

# Reset the environment
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

# Step the environment
action = jax.random.uniform(rng, (12,)) # Random action
next_state = env.step(state, action)
```

## Code Structure

-   `pupperv3_mjx/environment.py`: **Core Logic**. Contains the `PupperV3Env` class, physics steps, reward calculation, and the custom hybrid control implementation.
-   `pupperv3_mjx/rewards.py`: JIT-compiled reward functions for RL training (tracking, stability, energy efficiency).
-   `pupperv3_mjx/config.py`: Configuration dictionaries for rewards and environment parameters.
-   `pupperv3_mjx/domain_randomization.py`: Tools for randomizing mass, friction, and startup capabilities to improve policy robustness.
-   `meshes/`: Contains STL files for the robot visual and collision geometry.