"""
soccer_visualizer.py — Real-time Pygame visualiser for the Final Project soccer bot.

Pattern mirrors ekf_visualizer.py (A3) and maze_visualizer.py (A4):
  - Instantiated lazily in StudentController
  - update() called at the end of every step()
  - Returns False when user closes the window
"""

import math
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Field geometry (egocentric frame — same as starter_controller.py)
#   x: [-4.5, 4.5]  (target goal at +4.5, opponent goal at -4.5)
#   y: [-3.0, 3.0]
# ---------------------------------------------------------------------------
FIELD_X_MIN, FIELD_X_MAX = -4.5, 4.5
FIELD_Y_MIN, FIELD_Y_MAX = -3.0, 3.0

# Penalty cross positions and center
CROSSES = [(3.25, 0.0), (-3.25, 0.0)]
CENTER  = (0.0, 0.0)

# Screen layout
MARGIN   = 40          # px around the field
HUD_H    = 64          # px status bar height
FIELD_W_M = FIELD_X_MAX - FIELD_X_MIN   # 9 m
FIELD_H_M = FIELD_Y_MAX - FIELD_Y_MIN   # 6 m


class SoccerVisualizer:
    # ── Colours ──────────────────────────────────────────────────────────────
    BG            = ( 15,  15,  25)
    FIELD_GREEN   = ( 34,  85,  34)
    FIELD_DARK    = ( 28,  70,  28)   # alternating stripes (decorative)
    LINE_WHITE    = (240, 240, 240)
    GOAL_TARGET   = ( 60, 200, 255)   # our goal (the one we score in)
    GOAL_OPP      = (255,  80,  80)   # opponent goal
    CORNER_COL    = (220, 180,  50)
    CROSS_COL     = (220, 180,  50)
    CENTER_COL    = (200, 200, 200)
    ROBOT_COL     = ( 80, 200,  80)
    ROBOT_HDG     = (255, 255, 255)
    ELLIPSE_ROB   = (255, 220,  60)
    BALL_COL      = (255, 140,   0)
    BALL_LOW_CONF = (160,  90,  20)
    OBS_BALL      = (255, 200,  80)
    OBS_LM        = (160, 160,  80)
    TRAJ_COL      = ( 50, 160, 255)
    TEXT          = (210, 210, 210)
    TEXT_BRIGHT   = (255, 255, 180)
    HUD_BG        = ( 12,  12,  20)
    STATE_COLORS  = {
        "SEARCH_BALL": (200, 100,  50),
        "MOVE_TO_BALL": (100, 200, 100),
        "ALIGN":        (100, 150, 255),
        "DRIBBLE":      ( 50, 220, 220),
        "DONE":         (200, 200,  50),
    }
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, width=None):
        pygame.init()

        # Compute window size preserving field aspect ratio
        if width is None:
            width = 900
        self.PX_PER_M = (width - 2 * MARGIN) / FIELD_W_M
        self.FIELD_PX_W = int(FIELD_W_M * self.PX_PER_M)
        self.FIELD_PX_H = int(FIELD_H_M * self.PX_PER_M)
        self.W = self.FIELD_PX_W + 2 * MARGIN
        self.H = self.FIELD_PX_H + 2 * MARGIN

        self.screen = pygame.display.set_mode((self.W, self.H + HUD_H))
        pygame.display.set_caption("Soccer Bot – live visualiser")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("monospace", 14)
        self.sfont = pygame.font.SysFont("monospace", 11)

        self._traj: list = []
        self._max_traj = 5000

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _w2s(self, wx, wy):
        """World metres → screen pixel (y-axis flipped)."""
        sx = MARGIN + (wx - FIELD_X_MIN) * self.PX_PER_M
        sy = MARGIN + (FIELD_Y_MAX - wy) * self.PX_PER_M
        return int(sx), int(sy)

    def _scale(self, d):
        """World distance in metres → pixels."""
        return max(1, int(d * self.PX_PER_M))

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_field(self):
        """Draw the soccer field: green background, white lines, goals."""
        # Background
        field_rect = pygame.Rect(MARGIN, MARGIN, self.FIELD_PX_W, self.FIELD_PX_H)
        pygame.draw.rect(self.screen, self.FIELD_GREEN, field_rect)

        # Alternating vertical stripes (decorative, like a real pitch)
        stripe_w = self._scale(1.0)
        for i in range(int(FIELD_W_M)):
            if i % 2 == 1:
                sx = MARGIN + i * stripe_w
                stripe = pygame.Rect(sx, MARGIN, stripe_w, self.FIELD_PX_H)
                pygame.draw.rect(self.screen, self.FIELD_DARK, stripe)

        # Outer boundary
        pygame.draw.rect(self.screen, self.LINE_WHITE, field_rect, 2)

        # Centre line (x = 0)
        cx, cy_top = self._w2s(0, FIELD_Y_MAX)
        _, cy_bot  = self._w2s(0, FIELD_Y_MIN)
        pygame.draw.line(self.screen, self.LINE_WHITE, (cx, cy_top), (cx, cy_bot), 1)

        # Centre circle (r = 0.75 m)
        ccx, ccy = self._w2s(0, 0)
        pygame.draw.circle(self.screen, self.LINE_WHITE,
                           (ccx, ccy), self._scale(0.75), 1)
        pygame.draw.circle(self.screen, self.LINE_WHITE, (ccx, ccy), 3)

        # Goals: 2.6 m wide, 0.5 m deep
        GOAL_HALF_W = 1.3   # half-width in y
        GOAL_DEPTH  = 0.5

        # Target goal (+x side, cyan)
        gx1, gy1 = self._w2s(FIELD_X_MAX, GOAL_HALF_W)
        gx2, gy2 = self._w2s(FIELD_X_MAX + GOAL_DEPTH, -GOAL_HALF_W)
        goal_rect = pygame.Rect(min(gx1,gx2), min(gy1,gy2),
                                abs(gx2-gx1), abs(gy2-gy1))
        pygame.draw.rect(self.screen, self.GOAL_TARGET, goal_rect, 2)
        lbl = self.sfont.render("OUR GOAL", True, self.GOAL_TARGET)
        self.screen.blit(lbl, (gx1 + 2, min(gy1, gy2) - 14))

        # Opponent goal (−x side, red)
        gx1, gy1 = self._w2s(FIELD_X_MIN, GOAL_HALF_W)
        gx2, gy2 = self._w2s(FIELD_X_MIN - GOAL_DEPTH, -GOAL_HALF_W)
        goal_rect = pygame.Rect(min(gx1,gx2), min(gy1,gy2),
                                abs(gx2-gx1), abs(gy2-gy1))
        pygame.draw.rect(self.screen, self.GOAL_OPP, goal_rect, 2)
        lbl = self.sfont.render("OPP GOAL", True, self.GOAL_OPP)
        self.screen.blit(lbl, (min(gx1,gx2) - 62, min(gy1,gy2) - 14))

        # Penalty areas: 2 m wide (y), 1 m deep (x)
        for sign, base_x in [(1, FIELD_X_MAX), (-1, FIELD_X_MIN)]:
            px1, py1 = self._w2s(base_x, 1.0)
            px2, py2 = self._w2s(base_x - sign * 1.0, -1.0)
            prect = pygame.Rect(min(px1,px2), min(py1,py2),
                                abs(px2-px1), abs(py2-py1))
            pygame.draw.rect(self.screen, self.LINE_WHITE, prect, 1)

    def _draw_landmarks(self):
        """Draw known field landmarks (corners, crosses, center)."""
        # Corners
        corners = [
            (FIELD_X_MAX, FIELD_Y_MAX),
            (FIELD_X_MAX, FIELD_Y_MIN),
            (FIELD_X_MIN, FIELD_Y_MAX),
            (FIELD_X_MIN, FIELD_Y_MIN),
        ]
        for cx, cy in corners:
            sx, sy = self._w2s(cx, cy)
            size = 6
            pygame.draw.line(self.screen, self.CORNER_COL,
                             (sx-size, sy-size), (sx+size, sy+size), 2)
            pygame.draw.line(self.screen, self.CORNER_COL,
                             (sx-size, sy+size), (sx+size, sy-size), 2)

        # Penalty crosses
        for cx, cy in CROSSES:
            sx, sy = self._w2s(cx, cy)
            size = 5
            pygame.draw.line(self.screen, self.CROSS_COL,
                             (sx-size, sy), (sx+size, sy), 2)
            pygame.draw.line(self.screen, self.CROSS_COL,
                             (sx, sy-size), (sx, sy+size), 2)

    def _draw_cov_ellipse(self, overlay, cov2x2, cx, cy, color, n_sigma=2):
        """Draw filled 2σ confidence ellipse on the overlay surface."""
        try:
            evals, evecs = np.linalg.eigh(cov2x2)
            evals = np.maximum(evals, 1e-9)
            angle = math.atan2(evecs[1, 1], evecs[0, 1])
            a_px = max(2, min(self._scale(n_sigma * math.sqrt(evals[1])), 300))
            b_px = max(2, min(self._scale(n_sigma * math.sqrt(evals[0])), 300))

            pts = []
            for t in np.linspace(0, 2*math.pi, 48, endpoint=False):
                ex = a_px * math.cos(t)
                ey = b_px * math.sin(t)
                rx =  ex * math.cos(angle) - ey * math.sin(angle)
                ry =  ex * math.sin(angle) + ey * math.cos(angle)
                pts.append((cx + int(rx), cy - int(ry)))
            if len(pts) > 2:
                pygame.draw.polygon(overlay, (*color, 40), pts)
                pygame.draw.polygon(overlay, (*color, 180), pts, 1)
        except Exception:
            pass

    def _draw_robot(self, rx, ry, theta):
        """Draw robot circle + heading arrow."""
        sx, sy = self._w2s(rx, ry)
        r_px   = max(5, self._scale(0.105))   # TurtleBot3 radius ≈ 0.105 m
        pygame.draw.circle(self.screen, self.ROBOT_COL, (sx, sy), r_px, 2)
        hdx = int(r_px * 1.8 *  math.cos(theta))
        hdy = int(r_px * 1.8 * -math.sin(theta))   # screen y flipped
        pygame.draw.line(self.screen, self.ROBOT_HDG,
                         (sx, sy), (sx + hdx, sy + hdy), 2)

    def _draw_ball(self, ball_world, ball_conf):
        """Draw ball estimate, colour-coded by confidence."""
        if ball_world is None:
            return
        bx, by = ball_world
        sx, sy = self._w2s(bx, by)
        b_px   = max(4, self._scale(0.11))   # RoboCup soccer ball ≈ 0.11 m radius

        alpha = min(1.0, ball_conf / 8)
        color = tuple(int(a*alpha + b*(1-alpha))
                      for a, b in zip(self.BALL_COL, self.BALL_LOW_CONF))
        pygame.draw.circle(self.screen, color, (sx, sy), b_px)
        pygame.draw.circle(self.screen, self.LINE_WHITE, (sx, sy), b_px, 1)

    def _draw_hud(self, state, mu, ball_world, ball_conf, step):
        """Status bar below the field."""
        y0 = self.H
        pygame.draw.rect(self.screen, self.HUD_BG, (0, y0, self.W, HUD_H))

        state_color = self.STATE_COLORS.get(state, self.TEXT)
        x, y, theta = mu

        # State indicator
        state_lbl = self.font.render(f"State: {state}", True, state_color)
        self.screen.blit(state_lbl, (8, y0 + 4))

        # Pose
        pose_lbl = self.font.render(
            f"EKF: x={x:+.2f}  y={y:+.2f}  θ={math.degrees(theta):+.1f}°",
            True, self.TEXT)
        self.screen.blit(pose_lbl, (8, y0 + 22))

        # Ball info
        if ball_world is not None:
            bx, by = ball_world
            ball_lbl = self.font.render(
                f"Ball: x={bx:+.2f}  y={by:+.2f}  conf={ball_conf}",
                True, self.BALL_COL)
        else:
            ball_lbl = self.font.render("Ball: not estimated", True, self.BALL_LOW_CONF)
        self.screen.blit(ball_lbl, (8, y0 + 40))

        # Step counter (right side)
        step_lbl = self.font.render(f"step {step:5d}", True, self.TEXT)
        self.screen.blit(step_lbl, (self.W - 100, y0 + 4))

        # ESC hint
        hint = self.sfont.render("ESC / close = stop visualiser", True, (100, 100, 100))
        self.screen.blit(hint, (self.W - 210, y0 + 44))

    def _draw_legend(self):
        items = [
            (self.ROBOT_COL,   "Robot (EKF pose)"),
            (self.ELLIPSE_ROB, "2σ uncertainty"),
            (self.BALL_COL,    "Ball estimate"),
            (self.OBS_BALL,    "Ball observation"),
            (self.OBS_LM,      "Landmark observation"),
            (self.TRAJ_COL,    "Robot trajectory"),
            (self.GOAL_TARGET, "Our goal  (+x)"),
            (self.GOAL_OPP,    "Opponent goal (−x)"),
        ]
        x0 = self.W - 190
        y0 = MARGIN + 6
        bg = pygame.Surface((185, len(items)*15 + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        self.screen.blit(bg, (x0 - 4, y0 - 2))
        for color, label in items:
            pygame.draw.rect(self.screen, color, (x0, y0+2, 10, 10))
            lbl = self.sfont.render(label, True, self.TEXT)
            self.screen.blit(lbl, (x0 + 14, y0))
            y0 += 15

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, state, mu, sigma, ball_world, ball_conf,
               sensors, step):
        """
        Render one frame.

        Parameters
        ----------
        state      : str            current FSM state name
        mu         : np.ndarray     EKF pose [x, y, theta]
        sigma      : np.ndarray     3×3 EKF covariance
        ball_world : np.ndarray|None  ball position estimate in world frame
        ball_conf  : int            ball confidence counter
        sensors    : dict           raw sensor dict from the controller wrapper
        step       : int            simulation step counter

        Returns True to keep running, False if user closed the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # Accumulate trajectory
        self._traj.append((mu[0], mu[1]))
        if len(self._traj) > self._max_traj:
            self._traj = self._traj[-self._max_traj:]

        # ── Background ─────────────────────────────────────────────────────
        self.screen.fill(self.BG)
        self._draw_field()
        self._draw_landmarks()

        # ── Transparency overlay for ellipses and observation lines ────────
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        # Robot covariance ellipse
        rx, ry = self._w2s(mu[0], mu[1])
        self._draw_cov_ellipse(overlay, sigma[0:2, 0:2], rx, ry,
                               self.ELLIPSE_ROB, n_sigma=2)

        # Observation lines to visible landmarks
        for sensor_key, types in [("goal", "goal"), ("corners", "corner"),
                                   ("center_circle", "center"),
                                   ("penalty_cross", "cross")]:
            raw = sensors.get(sensor_key)
            if raw is None:
                continue
            obs_list = raw if isinstance(raw, list) else [raw]
            for obs in obs_list:
                if obs is None:
                    continue
                dist, bearing = float(obs[0]), float(obs[1])
                angle = mu[2] + bearing
                ox = mu[0] + dist * math.cos(angle)
                oy = mu[1] + dist * math.sin(angle)
                osx, osy = self._w2s(ox, oy)
                pygame.draw.line(overlay, (*self.OBS_LM, 120),
                                 (rx, ry), (osx, osy), 1)
                pygame.draw.circle(overlay, (*self.OBS_LM, 180),
                                   (osx, osy), 4)

        # Observation line to ball (if visible)
        ball_obs = sensors.get("ball")
        if ball_obs is not None:
            dist, bearing = float(ball_obs[0]), float(ball_obs[1])
            angle = mu[2] + bearing
            bx_obs = mu[0] + dist * math.cos(angle)
            by_obs = mu[1] + dist * math.sin(angle)
            osx, osy = self._w2s(bx_obs, by_obs)
            pygame.draw.line(overlay, (*self.OBS_BALL, 200),
                             (rx, ry), (osx, osy), 2)

        # Opponent observation line
        opp_obs = sensors.get("opponent")
        if opp_obs is not None:
            dist, bearing = float(opp_obs[0]), float(opp_obs[1])
            angle = mu[2] + bearing
            ox = mu[0] + dist * math.cos(angle)
            oy = mu[1] + dist * math.sin(angle)
            osx, osy = self._w2s(ox, oy)
            pygame.draw.line(overlay, (200, 80, 200, 160),
                             (rx, ry), (osx, osy), 2)
            pygame.draw.circle(overlay, (200, 80, 200, 200), (osx, osy), 7)

        self.screen.blit(overlay, (0, 0))

        # ── Trajectory ─────────────────────────────────────────────────────
        if len(self._traj) > 1:
            pts = [self._w2s(p[0], p[1]) for p in self._traj]
            pygame.draw.lines(self.screen, self.TRAJ_COL, False, pts, 1)

        # ── Ball + robot (drawn last so they're on top) ─────────────────
        self._draw_ball(ball_world, ball_conf)
        self._draw_robot(mu[0], mu[1], mu[2])

        # ── HUD + legend ───────────────────────────────────────────────────
        self._draw_legend()
        self._draw_hud(state, mu, ball_world, ball_conf, step)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        pygame.quit()
