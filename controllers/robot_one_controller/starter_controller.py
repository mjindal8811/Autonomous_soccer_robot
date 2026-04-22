"""student_controller controller — Final Project Soccer Bot."""

import math
import numpy as np

try:
    from soccer_visualizer import SoccerVisualizer
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Field geometry (in our egocentric coordinate frame)
#
# Both robots use the SAME frame:
#   - Start pose: (-1, 0) facing +x (θ=0)
#   - Target (own) goal: (4.5, 0)    ← robot was facing this at t=0
#   - Opponent goal:     (-4.5, 0)
#
# The soccer field is symmetric, so both ROBOT_ONE and ROBOT_TWO observe
# identical landmark patterns relative to themselves at startup.
# ---------------------------------------------------------------------------
TARGET_GOAL  = np.array([ 4.5, 0.0])
OPP_GOAL     = np.array([-4.5, 0.0])

# Known field landmarks: (world_x, world_y, type_string)
LANDMARKS = [
    (np.array([ 4.5,  0.0]), "goal"),
    (np.array([-4.5,  0.0]), "goal"),
    (np.array([ 4.5,  3.0]), "corner"),
    (np.array([ 4.5, -3.0]), "corner"),
    (np.array([-4.5,  3.0]), "corner"),
    (np.array([-4.5, -3.0]), "corner"),
    (np.array([ 0.0,  0.0]), "center"),
    (np.array([ 3.25, 0.0]), "cross"),
    (np.array([-3.25, 0.0]), "cross"),
]

# Sensor key → allowed landmark types
SENSOR_TYPES = {
    "goal":          {"goal"},
    "corners":       {"corner"},
    "center_circle": {"center"},
    "penalty_cross": {"cross"},
}

# EKF noise params
M = np.diag([0.005, 0.003])          # odometry noise (dist, dtheta)
Q = np.diag([0.05**2, (8*math.pi/180)**2])   # measurement noise (dist, bearing)
NIS_GATE = 13.82                      # chi²(2) 99th pct — reject outlier updates

# Drive gains
KP_TURN  = 5.0
BASE_SPD = 5.5
MAX_VEL  = 6.25

# Ball tracker
BALL_ALPHA   = 0.65   # smoothing factor (higher = trust new obs more)
BALL_CONF_MAX = 12


# ---------------------------------------------------------------------------
class StudentController:
    # -----------------------------------------------------------------------
    def __init__(self):
        # --- EKF pose state (x, y, theta) ---
        # Robot always starts 1 m behind centre, facing +x.
        self.mu    = np.array([-1.0, 0.0, 0.0])
        self.sigma = np.diag([0.02, 0.02, 0.01])

        # --- Ball estimate ---
        self.ball_world = None   # np.array([bx, by]) in world frame
        self.ball_conf  = 0      # confidence counter

        # --- Soccer state machine ---
        # States: SEARCH_BALL, MOVE_TO_BALL, ALIGN, DRIBBLE, DONE
        self.state = "SEARCH_BALL"

        # Which direction to spin when searching
        self.spin_dir = 1    # +1 = CCW, -1 = CW

        # Step counter (for debugging / timeouts)
        self.step_count = 0

        # Small hysteresis: number of steps we've been in DRIBBLE
        self.dribble_steps = 0

        # Last commanded controls (for diagnostics)
        self._last_ctrl = (0.0, 0.0)

        # --- Visualiser (lazy init on first step) ---
        self._viz = None
        self._viz_running = _VIZ_AVAILABLE

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    @staticmethod
    def _wrap(angle):
        """Wrap angle to [-π, π]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # -----------------------------------------------------------------------
    # EKF: Prediction step
    # -----------------------------------------------------------------------
    def _ekf_predict(self, odom):
        """Update pose estimate from odometry [delta_dist, delta_theta]."""
        d, dtheta = float(odom[0]), float(odom[1])
        x, y, theta = self.mu

        # Motion model: translate in current heading, then rotate
        new_x     = x + d * math.cos(theta)
        new_y     = y + d * math.sin(theta)
        new_theta = self._wrap(theta + dtheta)
        self.mu   = np.array([new_x, new_y, new_theta])

        # Jacobian of motion w.r.t. state
        G = np.eye(3)
        G[0, 2] = -d * math.sin(theta)
        G[1, 2] =  d * math.cos(theta)

        # Jacobian of motion w.r.t. noise
        V = np.array([
            [math.cos(theta), 0.0],
            [math.sin(theta), 0.0],
            [0.0,             1.0],
        ])

        self.sigma = G @ self.sigma @ G.T + V @ M @ V.T
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    # -----------------------------------------------------------------------
    # EKF: Correction step for a single observation of a known landmark
    # -----------------------------------------------------------------------
    def _ekf_correct(self, obs_dist, obs_bearing, lm_pos):
        """Correct EKF state given an observation of landmark at lm_pos."""
        if not (np.isfinite(obs_dist) and obs_dist > 0.0):
            return

        x, y, theta = self.mu
        lx, ly = lm_pos

        dx = lx - x
        dy = ly - y
        q  = dx*dx + dy*dy
        if q < 1e-6:
            return

        pred_dist    = math.sqrt(q)
        pred_bearing = self._wrap(math.atan2(dy, dx) - theta)

        innovation = np.array([
            obs_dist - pred_dist,
            self._wrap(obs_bearing - pred_bearing),
        ])

        # Measurement Jacobian H (2×3)
        H = np.array([
            [-dx/pred_dist,  -dy/pred_dist,  0.0],
            [ dy/q,          -dx/q,         -1.0],
        ])

        S = H @ self.sigma @ H.T + Q
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # NIS gate — reject statistically unlikely updates
        nis = float(innovation.T @ S_inv @ innovation)
        if nis > NIS_GATE:
            return

        K        = self.sigma @ H.T @ S_inv
        delta    = K @ innovation
        self.mu  = self.mu + delta
        self.mu[2] = self._wrap(self.mu[2])

        # Joseph form for numerical stability
        I_KH       = np.eye(3) - K @ H
        self.sigma = I_KH @ self.sigma @ I_KH.T + K @ Q @ K.T
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    # -----------------------------------------------------------------------
    # Data association: find the best matching landmark given current pose
    # -----------------------------------------------------------------------
    def _best_landmark(self, obs_dist, obs_bearing, allowed_types):
        """
        Return the landmark position that best explains (obs_dist, obs_bearing)
        given current pose estimate.  Uses a simple innovation-norm gate.
        Returns None if no match within threshold.
        """
        x, y, theta = self.mu
        best_pos   = None
        best_score = float("inf")

        # Predicted observation world position
        world_angle = theta + obs_bearing
        pred_lx = x + obs_dist * math.cos(world_angle)
        pred_ly = y + obs_dist * math.sin(world_angle)

        for lm_pos, lm_type in LANDMARKS:
            if lm_type not in allowed_types:
                continue

            lx, ly = lm_pos

            # Quick Euclidean pre-filter in world space
            if (lx - pred_lx)**2 + (ly - pred_ly)**2 > 3.0**2:
                continue

            dx = lx - x
            dy = ly - y
            q  = dx*dx + dy*dy
            if q < 1e-6:
                continue

            p_dist    = math.sqrt(q)
            p_bearing = self._wrap(math.atan2(dy, dx) - theta)

            d_dist    = abs(obs_dist - p_dist)
            d_bearing = abs(self._wrap(obs_bearing - p_bearing))

            # Normalised score (weight bearing error more)
            score = d_dist / 0.3 + d_bearing / (20 * math.pi / 180)

            if score < best_score:
                best_score = score
                best_pos   = lm_pos

        # Accept if within reasonable threshold
        if best_score < 3.5:
            return best_pos
        return None

    # -----------------------------------------------------------------------
    # Localisation: update EKF from all visible landmarks
    # -----------------------------------------------------------------------
    def _localize(self, sensors):
        for sensor_key, allowed_types in SENSOR_TYPES.items():
            raw = sensors.get(sensor_key)
            if raw is None:
                continue

            # Normalise to a list of observations
            obs_list = raw if isinstance(raw, list) else [raw]

            for obs in obs_list:
                if obs is None:
                    continue
                obs_dist, obs_bearing = float(obs[0]), float(obs[1])
                if not (np.isfinite(obs_dist) and obs_dist > 0.0):
                    continue

                lm_pos = self._best_landmark(obs_dist, obs_bearing, allowed_types)
                if lm_pos is not None:
                    self._ekf_correct(obs_dist, obs_bearing, lm_pos)

    # -----------------------------------------------------------------------
    # Ball tracker: convert polar obs to world estimate and smooth
    # -----------------------------------------------------------------------
    def _update_ball(self, ball_obs):
        if ball_obs is not None:
            dist, bearing = float(ball_obs[0]), float(ball_obs[1])
            if np.isfinite(dist) and dist > 0.0:
                x, y, theta = self.mu
                world_angle = theta + bearing
                bx = x + dist * math.cos(world_angle)
                by = y + dist * math.sin(world_angle)

                if self.ball_world is None:
                    self.ball_world = np.array([bx, by])
                else:
                    self.ball_world = (
                        BALL_ALPHA * np.array([bx, by])
                        + (1.0 - BALL_ALPHA) * self.ball_world
                    )

                self.ball_conf = min(self.ball_conf + 1, BALL_CONF_MAX)
                return

        # No observation this step
        self.ball_conf = max(0, self.ball_conf - 1)

    # -----------------------------------------------------------------------
    # Drive-to-point controller (from A4 pattern)
    # Returns (left_vel, right_vel, arrived:bool)
    # -----------------------------------------------------------------------
    def _drive_to(self, tx, ty, stop_dist=0.25, speed=None):
        if speed is None:
            speed = BASE_SPD
        x, y, theta = self.mu
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < stop_dist:
            return 0.0, 0.0, True

        desired  = math.atan2(dy, dx)
        err      = self._wrap(desired - theta)

        # Scale forward speed by alignment (stops sideways driving)
        forward  = speed * max(0.0, math.cos(err))
        # Slow down as we approach
        if dist < 0.6:
            forward *= max(0.3, dist / 0.6)

        turn = KP_TURN * err

        left  = max(-MAX_VEL, min(MAX_VEL, forward - turn))
        right = max(-MAX_VEL, min(MAX_VEL, forward + turn))
        return left, right, False

    # -----------------------------------------------------------------------
    # Spin in place (for SEARCH_BALL)
    # -----------------------------------------------------------------------
    def _spin(self):
        spd = 2.5
        return -self.spin_dir * spd, self.spin_dir * spd

    # -----------------------------------------------------------------------
    # State: SEARCH_BALL — rotate toward last known ball position, or full spin
    # -----------------------------------------------------------------------
    def _step_search_ball(self, sensors):
        # Ball re-acquired — head toward it
        if sensors["ball"] is not None and self.ball_conf >= 1:
            self.state = "MOVE_TO_BALL"
            return 0.0, 0.0

        x, y, theta = self.mu

        # If we have a last known position, spin toward it rather than full 360°
        if self.ball_world is not None:
            bx, by = self.ball_world
            bearing = self._wrap(math.atan2(by - y, bx - x) - theta)

            if abs(bearing) < 0.25:
                # Roughly facing last known ball pos — drive toward it slowly
                lv, rv, arrived = self._drive_to(bx, by, stop_dist=0.4)
                if not arrived:
                    return lv, rv
                # Arrived at last known pos but no ball — do full spin from here
            else:
                # Spin toward the last known ball direction
                spd = 2.5
                if bearing > 0:
                    return -spd, spd   # CCW
                else:
                    return spd, -spd   # CW

        # No prior estimate: blind spin
        lv, rv = self._spin()
        return lv, rv

    # -----------------------------------------------------------------------
    # State: MOVE_TO_BALL — drive within ~0.4 m of ball
    # -----------------------------------------------------------------------
    def _step_move_to_ball(self, sensors):
        # No ball estimate at all — full search
        if self.ball_world is None:
            self.state = "SEARCH_BALL"
            return self._spin()

        # Low confidence but still have an estimate — keep driving toward it
        if self.ball_conf == 0:
            # Re-use last known position; if ball reappears, conf will climb
            pass

        bx, by = self.ball_world

        # Re-orient spin direction based on ball side (used if ball is lost)
        x, y, theta = self.mu
        bearing_to_ball = self._wrap(math.atan2(by - y, bx - x) - theta)
        self.spin_dir = 1 if bearing_to_ball >= 0 else -1

        lv, rv, arrived = self._drive_to(bx, by, stop_dist=0.40)

        if arrived:
            self.state = "ALIGN"
            return 0.0, 0.0   # pause one tick; ALIGN takes over next tick

        return lv, rv

    # -----------------------------------------------------------------------
    # State: ALIGN — arc to position directly BEHIND the ball
    #   "Behind" = on the side of ball away from target goal,
    #   so driving forward will push ball toward goal.
    # -----------------------------------------------------------------------
    def _step_align(self, sensors):
        if self.ball_world is None:
            self.state = "SEARCH_BALL"
            return self._spin()

        bx, by = self.ball_world
        gx, gy = TARGET_GOAL

        # Direction from goal to ball (= direction robot should approach from)
        dx = bx - gx
        dy = by - gy
        norm = math.sqrt(dx*dx + dy*dy)
        if norm < 0.01:
            norm = 0.01

        # Behind-ball position: 0.38 m behind ball away from goal
        BEHIND_DIST = 0.38
        target_x = bx + BEHIND_DIST * dx / norm
        target_y = by + BEHIND_DIST * dy / norm

        lv, rv, arrived = self._drive_to(target_x, target_y, stop_dist=0.12)

        if arrived:
            self.state = "DRIBBLE"
            self.dribble_steps = 0
            return 0.0, 0.0   # pause one tick; DRIBBLE takes over next tick

        return lv, rv

    # -----------------------------------------------------------------------
    # State: DRIBBLE — drive toward target goal, pushing ball
    # -----------------------------------------------------------------------
    def _step_dribble(self, sensors):
        self.dribble_steps += 1

        # Check scoring: ball past goal line threshold
        if self.ball_world is not None:
            bx, by = self.ball_world
            if bx > 4.65:
                self.state = "DONE"
                return 0.0, 0.0

        # Ball lost: trust momentum for a few steps, then go re-acquire it
        if self.ball_conf == 0:
            if self.dribble_steps < 40:
                pass   # keep driving toward goal
            else:
                # We have ball_world estimate — go back and pick it up
                self.state = "MOVE_TO_BALL"
                return 0.0, 0.0

        # Check alignment: are we still behind ball relative to goal?
        # dot(ball→goal, ball→robot) < 0  means they point in opposite directions
        # = robot is on the FAR side of ball from goal = CORRECT position.
        # Realign only when dot > 0 (robot crept onto the goal-side of the ball).
        # Skip dot-check when ball is already close to goal — geometry is too
        # sensitive there and re-aligning just wastes time.
        if self.ball_world is not None:
            bx, by = self.ball_world
            x, y, theta = self.mu
            tbgx = TARGET_GOAL[0] - bx
            tbgy = TARGET_GOAL[1] - by
            tbrx = x - bx
            tbry = y - by
            dot = tbgx * tbrx + tbgy * tbry
            dist_robot_ball = math.sqrt(tbrx**2 + tbry**2)

            near_goal = bx > 4.2  # close enough — skip all alignment checks
            if not near_goal and (dot > 0 or dist_robot_ball > 1.2):
                self.state = "ALIGN"
                return 0.0, 0.0   # pause one tick; ALIGN takes over                                                                                                                                                                                              ..........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................  next tick

        # Drive toward a point PAST the goal so the robot never brakes early.
        # We exit via the ball-past-line check above, not by "arriving".
        lv, rv, _ = self._drive_to(5.5, 0.0, stop_dist=0.1, speed=6.25)
        return lv, rv

    # -----------------------------------------------------------------------
    # Main step — called every simulation tick
    # -----------------------------------------------------------------------
    def step(self, sensors):
        """
        Input:  sensors dict (ball, goal, center_circle, penalty_cross,
                              corners, opponent, odometry)
        Output: {"left_motor": float, "right_motor": float}
        """
        self.step_count += 1

        # Goal scored — shut everything down immediately
        if self.state == "DONE":
            return {"left_motor": 0.0, "right_motor": 0.0}

        # 1. EKF predict
        self._ekf_predict(sensors["odometry"])

        # 2. EKF correct with visible landmarks
        self._localize(sensors)

        # 3. Update ball estimate
        self._update_ball(sensors["ball"])

        # 4. State machine
        if self.state == "SEARCH_BALL":
            lv, rv = self._step_search_ball(sensors)
        elif self.state == "MOVE_TO_BALL":
            lv, rv = self._step_move_to_ball(sensors)
        elif self.state == "ALIGN":
            lv, rv = self._step_align(sensors)
        elif self.state == "DRIBBLE":
            lv, rv = self._step_dribble(sensors)
        else:  # DONE
            lv, rv = 0.0, 0.0

        self._last_ctrl = (lv, rv)

        # 5. Visualiser update (lazy init + running guard)
        if self._viz_running:
            if self._viz is None:
                self._viz = SoccerVisualizer()
            still_open = self._viz.update(
                state=self.state,
                mu=self.mu,
                sigma=self.sigma,
                ball_world=self.ball_world,
                ball_conf=self.ball_conf,
                sensors=sensors,
                step=self.step_count,
            )
            if not still_open:
                self._viz.close()
                self._viz_running = False

        return {"left_motor": lv, "right_motor": rv}
