# Autonomous Soccer Robot — Webots TurtleBot Controller

A fully autonomous soccer-playing TurtleBot implemented in [Webots](https://cyberbotics.com/). The robot localizes itself on a 9m x 6m soccer field using an Extended Kalman Filter, tracks a dynamic ball with an exponential smoother, and executes a 5-state behavioral FSM to find the ball, align behind it, and push it into the correct goal — all from noisy, incomplete polar-coordinate sensor readings.

---

## Demo
![Robot scoring a goal](assets/demo.gif)
> Robot starts 1m behind center, facing its target goal. Ball position varies per run.
> The robot must find the ball, position itself behind it, and score.

---
 
## Results
 
The robot successfully completed all three objectives — locating the ball, positioning behind it, and scoring on the correct goal — across every tested ball starting position, consistently finishing within **2 minutes of simulation time**. Localization remained stable throughout each run, with the EKF maintaining reliable pose estimates even during extended dribbling sequences where landmark observations were intermittent. The behavioral FSM handled ball re-acquisition seamlessly in cases where the ball drifted out of the robot's field of view mid-dribble, recovering and completing the goal without manual intervention.
 
---

## Features

- **EKF Localization** — fuses noisy odometry with landmark observations (goals, corners, center circle, penalty crosses) using a full predict-correct cycle with NIS gating to reject outlier measurements
- **Data Association** — resolves unknown landmark correspondences using a scored nearest-neighbor matcher with Euclidean pre-filtering and normalized innovation gating
- **Ball Tracker** — converts per-step polar observations to world-frame estimates using exponential smoothing (configurable alpha), with a confidence counter to handle occlusion
- **5-State Behavioral FSM** — `SEARCH_BALL → MOVE_TO_BALL → ALIGN → DRIBBLE → DONE`, with clean transitions, timeout handling, and re-acquisition logic
- **Proportional Steering Controller** — heading-error-based differential drive with forward speed scaled by alignment angle and proximity braking
- **Monte Carlo Test Suite** — 50-trial EKF validation across 6 scripted scenarios (straight drive, spin-in-place, arc, corner approach, 180° reverse, full field traversal) with RMSE, max error, and 95th-percentile pass/fail thresholds — no Webots required
- **Optional Pygame Visualizer** — real-time overlay of EKF covariance ellipse, ball world estimate, state label, and sensor rays (gracefully disabled if pygame is unavailable)

---

## Repository Structure

```
controllers/
└── robot_one_controller/
    ├── starter_controller.py       # Core implementation (EKF + FSM + ball tracker)
    ├── robot_one_controller.py     # Webots wrapper — sensor noise, motor interface (DO NOT MODIFY)
    ├── soccer_visualizer.py        # Optional Pygame real-time visualizer
    └── test_ekf_localization.py    # Standalone Monte Carlo EKF test suite
worlds/
├── soccer_solo.wbt                 # Single-robot field (main task)
└── soccer_dual.wbt                 # Two-robot field (tournament / extra credit)
```

---

## How It Works

### 1. EKF Localization

The robot maintains a Gaussian belief over its 3-DOF pose `(x, y, θ)`:

**Prediction** — at every timestep, the motion model propagates the pose estimate using noisy odometry `[Δdist, Δθ]`:

```
x'     = x + d·cos(θ)
y'     = y + d·sin(θ)
θ'     = wrap(θ + Δθ)
Σ'     = G·Σ·Gᵀ + V·M·Vᵀ
```

where `G` is the Jacobian of the motion model and `M = diag([0.005, 0.003])` is the odometry noise covariance.

**Correction** — when landmarks are observed (polar coords: distance + bearing), the EKF update runs:

```
innovation = [obs_dist - pred_dist,  wrap(obs_bearing - pred_bearing)]
NIS        = innovationᵀ · S⁻¹ · innovation       (chi² gated at 99th pct)
K          = Σ · Hᵀ · S⁻¹
μ         += K · innovation
Σ          = (I - KH)·Σ·(I - KH)ᵀ + K·Q·Kᵀ       (Joseph form)
```

Outlier observations are rejected via the NIS gate (`χ²(2) = 13.82` at 99th percentile), preventing a single bad landmark from corrupting the pose estimate.

**Data Association** — the robot solves the unknown correspondence problem at every correction step. For each incoming observation, it projects the observation into world space using the current pose estimate, then scores all candidate landmarks of the matching type using a normalized distance + bearing residual. A match is accepted only if the best score falls below a tuned threshold.

### 2. Ball Tracker

Raw polar ball observations are converted to world-frame coordinates using the current EKF pose estimate, then fused with the previous estimate via exponential smoothing:

```
ball_world = α · new_obs + (1 - α) · ball_world      (α = 0.65)
```

A confidence counter increments on each observation and decays when the ball is not seen, enabling the FSM to distinguish "ball not visible right now" from "ball truly lost."

### 3. Behavioral FSM

```
              ball seen
SEARCH_BALL ────────────► MOVE_TO_BALL
     ▲                         │ within 0.4m
     │ ball lost                ▼
     │              ALIGN (position behind ball)
     │                         │ in position
     │                         ▼
     └──────────────── DRIBBLE ──────► DONE (ball > x=4.65)
```

- **SEARCH_BALL** — spins toward last known ball bearing or performs a blind 360° scan if no prior estimate exists
- **MOVE_TO_BALL** — drives to within 0.4m of ball world estimate using proportional steering
- **ALIGN** — computes a "behind-ball" waypoint 0.38m from the ball along the ball-to-goal axis, then drives to it
- **DRIBBLE** — drives toward a point past the goal (x=5.5) at full speed, checking the ball-past-line condition and re-aligning if the robot drifts to the goal-side of the ball

### 4. Proportional Steering Controller

```python
err     = wrap(desired_heading - θ)
forward = speed · max(0, cos(err))        # naturally slows when misaligned
turn    = Kp · err                        # Kp = 5.0
left    = clamp(forward - turn, ±6.25)
right   = clamp(forward + turn, ±6.25)
```

Forward speed scales with `cos(heading_error)` so the robot turns in place when badly misaligned and only drives forward once roughly on course. Proximity braking kicks in within 0.6m of the target.

---

## Sensor Interface

The robot receives a `sensors` dict every simulation timestep (20ms):

| Key | Type | Description |
|-----|------|-------------|
| `odometry` | `[Δdist, Δθ]` | Noisy forward distance and heading change |
| `ball` | `(dist, bearing)` or `None` | Ball in 90° FOV |
| `goal` | list of `(dist, bearing)` | Goal posts in FOV |
| `corners` | list of `(dist, bearing)` | Field corners in FOV |
| `center_circle` | `(dist, bearing)` or `None` | Center circle in FOV |
| `penalty_cross` | list of `(dist, bearing)` | Penalty crosses in FOV |
| `opponent` | `(dist, bearing)` or `None` | Opponent robot in FOV |

All observations are relative polar coordinates. Data association (which goal? which corner?) is not solved by the framework — that is handled inside `starter_controller.py`.

---

## EKF Noise Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `M[0,0]` | 0.005 | Odometry distance std |
| `M[1,1]` | 0.003 | Odometry heading std (rad) |
| `Q[0,0]` | 0.05² | Observation distance variance |
| `Q[1,1]` | (8°)² | Observation bearing variance |
| `NIS_GATE` | 13.82 | χ²(2) 99th percentile gate |

---

## Running the Simulation

### Prerequisites

- [Webots R2023b](https://cyberbotics.com/#download) or later
- Python 3.8+
- `numpy`
- `pygame` (optional, for visualizer)

### Setup

```bash
git clone https://github.com/mjindal8811/Autonomous_soccer_robot.git
cd autonomous-soccer-robot
pip install numpy pygame
```

### Launch in Webots

1. Open Webots
2. `File > Open World > worlds/soccer_solo.wbt`
3. Press Play — the robot will start autonomously

### Run the EKF Test Suite (no Webots needed)

```bash
cd controllers/robot_one_controller
python test_ekf_localization.py
```

Expected output:

```
============================================================
  EKF Localization Exhaustive Test
============================================================

------------------------------------------------------------
  Straight drive (robot=>goal)
  (50 Monte Carlo trials)
------------------------------------------------------------
  pos RMSE  mean=0.0xxx m  max=0.xxxx m  95pct=0.xxxx m
  ...
  Thresholds: pos<0.30m @95pct, angle<15deg @95pct  =>  [PASS]
...
  Overall: ALL PASS — EKF is robust
```

### Tournament Mode (Two Robots)

```bash
# Open worlds/soccer_dual.wbt in Webots
# Your controller is automatically placed in robot_one_controller/ or robot_two_controller/
```

---

## Test Suite Details

`test_ekf_localization.py` validates the EKF across 6 scripted scenarios using 50 Monte Carlo trials each:

| Scenario | Description |
|----------|-------------|
| Straight drive | Robot drives directly toward the goal along x-axis |
| Spin in place | Full 360° rotation to stress heading estimation |
| Arc across field | Curved trajectory mixing translation and rotation |
| Ball at corner | Drive toward (3, 2.5) — corner-heavy landmark set |
| Ball behind robot | Spin 180° then drive opposite direction |
| Full field traversal | Worst-case drift — covers entire 9m x 6m field |

Pass criteria: position RMSE < 0.30m and heading RMSE < 15° at the 95th percentile across all trials.

---

## Skills Demonstrated

- Extended Kalman Filter (EKF) design and tuning for mobile robot localization
- Probabilistic data association with NIS gating
- Behavioral finite state machine architecture for autonomous robot control
- Exponential smoothing for dynamic object tracking under partial observability
- Monte Carlo evaluation methodology for robotics algorithms
- Webots simulation environment and differential-drive robot control
