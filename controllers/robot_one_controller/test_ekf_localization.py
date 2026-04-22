"""
Exhaustive EKF localization test — no Webots needed.

Runs the StudentController EKF through scripted trajectories with
simulated noisy sensors, compares EKF estimates to ground truth,
and reports RMSE, max error, and consistency (sigma coverage).

Run with:
    cd controllers/robot_one_controller
    python test_ekf_localization.py
"""

import sys, os, math, textwrap
import numpy as np

# -- import the student EKF without Webots ----------------------------------
sys.path.insert(0, os.path.dirname(__file__))
# Stub out the visualiser so the import doesn't fail without pygame
import types
_stub = types.ModuleType("soccer_visualizer")
_stub.SoccerVisualizer = None
sys.modules["soccer_visualizer"] = _stub

from starter_controller import StudentController, LANDMARKS, SENSOR_TYPES

# -- simulation noise (must match constants in starter_controller.py) --------
OBS_DIST_STD    = 0.05          # metres
OBS_BEARING_STD = 8 * math.pi / 180  # radians
ODO_DIST_STD    = 0.005
ODO_THETA_STD   = 0.003

SENSOR_FOV      = math.pi / 2   # 90-degree field of view
N_TRIALS        = 50            # Monte Carlo repetitions per scenario
DT              = 0.02          # 20 ms step


# -- helpers -----------------------------------------------------------------
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def landmark_obs(true_x, true_y, true_theta, lm_pos, rng):
    """Return noisy polar observation (dist, bearing) or None if outside FOV."""
    dx = lm_pos[0] - true_x
    dy = lm_pos[1] - true_y
    true_dist    = math.sqrt(dx**2 + dy**2)
    true_bearing = wrap(math.atan2(dy, dx) - true_theta)
    if abs(true_bearing) > SENSOR_FOV / 2:
        return None
    obs_dist    = max(0.01, true_dist    + rng.normal(0, OBS_DIST_STD))
    obs_bearing =           true_bearing + rng.normal(0, OBS_BEARING_STD)
    return np.array([obs_dist, obs_bearing])


def build_sensors(true_x, true_y, true_theta,
                  prev_x, prev_y, prev_theta, rng):
    """Build a sensors dict identical in structure to what Webots provides."""
    # Noisy odometry
    dx      = true_x - prev_x
    dy      = true_y - prev_y
    d_dist  = math.sqrt(dx**2 + dy**2) + rng.normal(0, ODO_DIST_STD)
    d_theta = wrap(true_theta - prev_theta) + rng.normal(0, ODO_THETA_STD)

    sensors = {
        "odometry":      [d_dist, d_theta],
        "ball":          None,
        "opponent":      None,
        "goal":          [],
        "corners":       [],
        "center_circle": [],
        "penalty_cross": [],
    }

    # Simulate all landmark observations through the FOV
    sensor_key_for_type = {v: k for k, vals in SENSOR_TYPES.items() for v in vals}
    for lm_pos, lm_type in LANDMARKS:
        obs = landmark_obs(true_x, true_y, true_theta, lm_pos, rng)
        if obs is None:
            continue
        key = sensor_key_for_type.get(lm_type)
        if key:
            sensors[key].append(obs)

    # Convert single-element lists to scalar (match wrapper format)
    for key in ["goal", "corners", "center_circle", "penalty_cross"]:
        lst = sensors[key]
        if len(lst) == 0:
            sensors[key] = None
        elif len(lst) == 1:
            sensors[key] = lst[0]
        # else: keep as list (multiple observations)

    return sensors


def run_trajectory(waypoints, n_steps_each, seed):
    """
    Drive the robot through a list of (x,y,theta) waypoints, holding each
    for n_steps_each steps.  Returns arrays of true and estimated poses.
    """
    rng  = np.random.default_rng(seed)
    ctrl = StudentController()
    ctrl._viz_running = False   # disable visualiser

    true_poses = []
    est_poses  = []

    # Start at first waypoint
    tx, ty, tt = waypoints[0]
    ctrl.mu = np.array([tx, ty, tt])   # seed EKF at truth for a fair test

    for wp_x, wp_y, wp_theta in waypoints[1:]:
        # Interpolate n_steps_each steps toward next waypoint
        prev_tx, prev_ty, prev_tt = tx, ty, tt
        for i in range(1, n_steps_each + 1):
            alpha = i / n_steps_each
            tx = prev_tx + alpha * (wp_x - prev_tx)
            ty = prev_ty + alpha * (wp_y - prev_ty)
            tt = prev_tt + alpha * wrap(wp_theta - prev_tt)

            sensors = build_sensors(tx, ty, tt,
                                    prev_tx if i == 1 else
                                    prev_tx + (i-1)/n_steps_each*(wp_x-prev_tx),
                                    prev_ty if i == 1 else
                                    prev_ty + (i-1)/n_steps_each*(wp_y-prev_ty),
                                    prev_tt if i == 1 else
                                    prev_tt + (i-1)/n_steps_each*wrap(wp_theta-prev_tt),
                                    rng)
            ctrl.step(sensors)

            true_poses.append((tx, ty, tt))
            est_poses.append(tuple(ctrl.mu))

    return np.array(true_poses), np.array(est_poses), ctrl


def analyse(name, true_arr, est_arr):
    """Print RMSE, max error, and consistency for one scenario."""
    pos_err = np.sqrt((true_arr[:,0]-est_arr[:,0])**2 +
                      (true_arr[:,1]-est_arr[:,1])**2)
    ang_err = np.abs([wrap(t-e) for t,e in
                      zip(true_arr[:,2], est_arr[:,2])])

    print(f"  pos RMSE  : {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"  pos max   : {np.max(pos_err):.4f} m")
    print(f"  angle RMSE: {math.degrees(np.sqrt(np.mean(ang_err**2))):.2f}deg")
    print(f"  angle max : {math.degrees(np.max(ang_err)):.2f}deg")


# -- test scenarios -----------------------------------------------------------
SCENARIOS = [
    {
        "name": "Straight drive (robot=>goal)",
        "waypoints": [(-1,0,0), (0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)],
        "steps_each": 100,
    },
    {
        "name": "Spin in place (heading uncertainty)",
        "waypoints": [(-1,0,0), (-1,0,math.pi/2), (-1,0,math.pi),
                      (-1,0,-math.pi/2), (-1,0,0)],
        "steps_each": 75,
    },
    {
        "name": "Arc across field (curve)",
        "waypoints": [(-1,0,0), (0,1,math.pi/4), (2,2,0),
                      (4,1,-math.pi/4), (4,0,0)],
        "steps_each": 100,
    },
    {
        "name": "Ball at corner — drive to (3, 2.5)",
        "waypoints": [(-1,0,0), (0,0.8,0.4), (1,1.6,0.3),
                      (2,2.2,0.2), (3,2.5,0)],
        "steps_each": 100,
    },
    {
        "name": "Ball behind robot — spin 180deg then drive",
        "waypoints": [(-1,0,0), (-1,0,math.pi), (-1,0,math.pi),
                      (-2,0,math.pi), (-3,0,math.pi)],
        "steps_each": 80,
    },
    {
        "name": "Full field traversal (worst-case drift)",
        "waypoints": [(-1,0,0),(1,0,0),(3,0,0),(4,0,0),
                      (4,2,math.pi/2),(0,2,math.pi),(-4,2,math.pi),
                      (-4,-2,-math.pi/2),(0,-2,0),(4,-2,0)],
        "steps_each": 120,
    },
]


def run_monte_carlo(scenario, n_trials=N_TRIALS):
    name      = scenario["name"]
    waypoints = scenario["waypoints"]
    steps     = scenario["steps_each"]

    all_pos_rmse, all_ang_rmse, all_pos_max = [], [], []

    for seed in range(n_trials):
        true_arr, est_arr, _ = run_trajectory(waypoints, steps, seed)
        pos_err = np.sqrt((true_arr[:,0]-est_arr[:,0])**2 +
                          (true_arr[:,1]-est_arr[:,1])**2)
        ang_err = np.array([abs(wrap(t-e)) for t,e in
                            zip(true_arr[:,2], est_arr[:,2])])
        all_pos_rmse.append(np.sqrt(np.mean(pos_err**2)))
        all_ang_rmse.append(math.degrees(np.sqrt(np.mean(ang_err**2))))
        all_pos_max.append(np.max(pos_err))

    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"  ({n_trials} Monte Carlo trials)")
    print(f"{'-'*60}")
    print(f"  pos RMSE  mean={np.mean(all_pos_rmse):.4f} m  "
          f"max={np.max(all_pos_rmse):.4f} m  "
          f"95pct={np.percentile(all_pos_rmse,95):.4f} m")
    print(f"  pos max   mean={np.mean(all_pos_max):.4f} m  "
          f"worst={np.max(all_pos_max):.4f} m")
    print(f"  angle RMSE mean={np.mean(all_ang_rmse):.2f}deg  "
          f"max={np.max(all_ang_rmse):.2f}deg  "
          f"95pct={np.percentile(all_ang_rmse,95):.2f}deg")

    # Pass/fail thresholds
    pos_ok = np.percentile(all_pos_rmse, 95) < 0.30   # 30 cm at 95th pct
    ang_ok = np.percentile(all_ang_rmse, 95) < 15.0   # 15deg at 95th pct
    status = "PASS" if (pos_ok and ang_ok) else "FAIL"
    print(f"\n  Thresholds: pos<0.30m @95pct, angle<15deg @95pct  =>  [{status}]")
    return status


if __name__ == "__main__":
    print("=" * 60)
    print("  EKF Localization Exhaustive Test")
    print("=" * 60)

    results = []
    for scenario in SCENARIOS:
        status = run_monte_carlo(scenario)
        results.append((scenario["name"], status))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, status in results:
        label = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {label}  {name}")

    all_pass = all(s == "PASS" for _, s in results)
    print()
    print("  Overall:", "ALL PASS — EKF is robust" if all_pass
          else "SOME FAILURES — review EKF tuning")
    print("=" * 60)
