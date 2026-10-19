"""
vpso_core.py
============
Environments and spline path model.

  ENVIRONMENTS     the five conventional environments O1-O5 (100 x 100
                   workspace, robot radius 1.5)
  FastEnvironment  geometry of one environment: clamped cubic spline through
                   start, n control points and goal, sampled at N_s points.
                   The clamped spline on a fixed knot vector is linear in the
                   interpolated points, so its basis matrix B (Eq. 4) is
                   precomputed once and a path is a single matrix product.
  make_cost_function_v2, VPSO
                   per-particle cost and optimizer of the earlier formulation
                   (two-phase inertia weight, transfer exponent 1/1.75,
                   population 300, clamp factor 0.05), used only for the
                   first-stage design-sensitivity sweeps of Section 3.7.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Conventional environments O1-O5
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    '1': {
        'name': 'Three vertical obstacles',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [12, 60], 'goal': [92, 60],
        'obstacles': [
            {'center': [20, 73], 'width': 10, 'height': 54},
            {'center': [50, 27], 'width': 10, 'height': 54},
            {'center': [80, 73], 'width': 10, 'height': 54},
        ],
    },
    '2': {
        'name': 'Z-shaped corridor',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [20, 65], 'goal': [80, 10],
        'obstacles': [
            {'center': [20, 80], 'width': 40, 'height': 4},
            {'center': [60, 65], 'width': 4,  'height': 30},
            {'center': [30, 50], 'width': 60, 'height': 4},
            {'center': [80, 30], 'width': 30, 'height': 4},
        ],
    },
    '3': {
        'name': 'Dense mixed obstacles',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [5, 5], 'goal': [95, 95],
        'obstacles': [
            {'center': [50, 65], 'radius': 9},
            {'center': [80, 20], 'radius': 9},
            {'center': [70, 75], 'width': 12, 'height': 60},
            {'center': [30, 20], 'width': 12, 'height': 60},
            {'center': [80, 80], 'width': 10, 'height': 10},
        ],
    },
    '4': {
        'name': 'Two tall rectangular blockers',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [18, 70], 'goal': [78, 50],
        'obstacles': [
            {'center': [33, 70], 'width': 10, 'height': 60},
            {'center': [66, 30], 'width': 10, 'height': 60},
        ],
    },
    '5': {
        'name': 'Forest (many circles)',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [5, 5], 'goal': [95, 95],
        'obstacles': [
            {'center': [20, 20], 'radius': 6},
            {'center': [35, 50], 'radius': 6},
            {'center': [20, 75], 'radius': 6},
            {'center': [50, 30], 'radius': 6},
            {'center': [50, 70], 'radius': 6},
            {'center': [65, 15], 'radius': 6},
            {'center': [75, 55], 'radius': 6},
            {'center': [80, 85], 'radius': 6},
            {'center': [40, 90], 'radius': 6},
        ],
    },
}


class FastEnvironment:
    """Vectorized environment: evaluates a whole path (N,2) at once."""

    def __init__(self, cfg, num_control_points=8, resolution=100):
        self.width = cfg['width']
        self.height = cfg['height']
        self.rr = cfg['robot_radius']
        self.start = np.array(cfg['start'], float)
        self.goal = np.array(cfg['goal'], float)
        self.num_cp = num_control_points
        self.resolution = resolution

        self.circles = []   # (center(2,), radius)
        self.rects = []     # (center(2,), half_w, half_h)
        for o in cfg['obstacles']:
            if 'radius' in o:
                self.circles.append((np.array(o['center'], float), o['radius']))
            else:
                self.rects.append((np.array(o['center'], float),
                                   o['width'] / 2.0, o['height'] / 2.0))

        if self.circles:
            self.c_centers = np.array([c for c, _ in self.circles])          # (K,2)
            self.c_radii = np.array([r for _, r in self.circles]) + self.rr  # (K,)
        if self.rects:
            self.r_centers = np.array([c for c, _, _ in self.rects])         # (M,2)
            self.r_half = np.array([[hw, hh] for _, hw, hh in self.rects])   # (M,2)

        # Precompute the clamped cubic-spline basis matrix M (resolution x n_pts):
        # spline(tt) = M @ points, because the clamped spline is linear in data.
        n_pts = num_control_points + 2
        t = np.linspace(0, 1, n_pts)
        tt = np.linspace(0, 1, resolution)
        B = np.zeros((resolution, n_pts))
        for j in range(n_pts):
            e = np.zeros(n_pts)
            e[j] = 1.0
            B[:, j] = CubicSpline(t, e, bc_type='clamped')(tt)
        self.basis = B

    # -- geometry ----------------------------------------------------------
    def path_from_cp(self, xy_norm):
        """xy_norm: flat array (2*num_cp,) of normalized control points."""
        cp = xy_norm.reshape(-1, 2) * np.array([self.width, self.height])
        pts = np.vstack((self.start, cp, self.goal))          # (n_pts, 2)
        path = self.basis @ pts                                # (res, 2)
        # clip to the workspace
        path[:, 0] = np.clip(path[:, 0], self.rr, self.width - self.rr)
        path[:, 1] = np.clip(path[:, 1], self.rr, self.height - self.rr)
        return path

    def collision_count(self, path):
        hit = np.zeros(len(path), dtype=bool)
        if self.circles:
            d = np.linalg.norm(path[:, None, :] - self.c_centers[None], axis=2)
            hit |= (d <= self.c_radii[None]).any(axis=1)
        if self.rects:
            diff = np.abs(path[:, None, :] - self.r_centers[None]) - self.r_half[None]
            diff = np.maximum(diff, 0.0)
            hit |= ((diff ** 2).sum(axis=2) <= self.rr ** 2).any(axis=1)
        return int(hit.sum())

    def env_violation_count(self, path):
        out = ((path[:, 0] < self.rr) | (path[:, 0] > self.width - self.rr) |
               (path[:, 1] < self.rr) | (path[:, 1] > self.height - self.rr))
        return int(out.sum())

    @staticmethod
    def path_length(path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def smoothness(self, path):
        """Mean absolute turning angle (rad) between consecutive segments.
        Lower = smoother. Reported as a metric, not part of the cost."""
        v = np.diff(path, axis=0)
        n = np.linalg.norm(v, axis=1)
        n = np.where(n < 1e-12, 1e-12, n)
        cosang = np.clip((v[:-1] * v[1:]).sum(axis=1) / (n[:-1] * n[1:]), -1, 1)
        return float(np.abs(np.arccos(cosang)).mean())


def make_cost_function(env,
                       path_length_weight=1.0,
                       start_violation_pen=500.0,
                       goal_violation_pen=500.0,
                       env_violation_pen=50.0,
                       collision_pen=500.0):
    """Per-particle cost: length + penalties (binary per-sample collisions)."""
    def cost(xy_norm):
        path = env.path_from_cp(xy_norm)
        length = env.path_length(path)
        col = env.collision_count(path)
        envv = env.env_violation_count(path)
        start_v = np.linalg.norm(path[0] - env.start) > env.rr
        goal_v = np.linalg.norm(path[-1] - env.goal) > env.rr

        c = path_length_weight * length
        if start_v:
            c += start_violation_pen
        if goal_v:
            c += goal_violation_pen
        c += env_violation_pen * envv
        c += collision_pen * col
        details = {'length': length, 'collisions': col, 'env_violations': envv,
                   'feasible': (col == 0 and envv == 0
                                and not start_v and not goal_v)}
        return c, details
    return cost


# ---------------------------------------------------------------------------
# VPSO of the earlier formulation (Section 3.7, first-stage sweeps)
# ---------------------------------------------------------------------------
def VPSO(problem, seed,
         max_iter=100, pop_size=300,
         adaptive_w=True, adaptive_c=True, resetting=25,
         fixed_w=0.7298, fixed_c1=1.49618, fixed_c2=1.49618,
         C_max=2.0, C_min=0.5, W_max=0.9, W_min=0.4, T_y_frac=0.6,
         reset_frac=0.20, verbose=False):

    rng = np.random.default_rng(seed)
    T_y = int(T_y_frac * max_iter)

    var_min, var_max = problem['var_min'], problem['var_max']
    num_var = problem['num_var']
    cost_fn = problem['cost_function']
    v_max = 0.05 * (var_max - var_min)
    v_min = -v_max

    def compute_W(t):
        if t <= T_y:
            return W_max - (W_max - W_min) * (t / T_y) ** 2
        return W_min + (W_max - W_min) * ((t - T_y) / (max_iter - T_y)) ** 2

    def compute_C1C2(t):
        ratio = np.clip(t / max_iter, 0.0, 1.0)
        p = ratio ** (1 / 1.75)
        return (float(C_max - p * (C_max - C_min)),
                float(C_min + p * (C_max - C_min)))

    # init
    X = rng.uniform(var_min, var_max, (pop_size, num_var))
    V = np.zeros((pop_size, num_var))
    costs = np.empty(pop_size)
    details = [None] * pop_size
    for i in range(pop_size):
        costs[i], details[i] = cost_fn(X[i])
    pbest_X = X.copy()
    pbest_c = costs.copy()
    g = int(np.argmin(pbest_c))
    gbest_x, gbest_c, gbest_d = pbest_X[g].copy(), pbest_c[g], details[g]

    history = np.empty(max_iter)

    for it in range(1, max_iter + 1):
        w = compute_W(it) if adaptive_w else fixed_w
        if adaptive_c:
            c1, c2 = compute_C1C2(it)
        else:
            c1, c2 = fixed_c1, fixed_c2

        # worst-20% restart
        reset_idx = np.array([], dtype=int)
        if resetting and (it % resetting == 0):
            k = max(1, int(reset_frac * pop_size))
            reset_idx = np.argpartition(costs, -k)[-k:]

        r1 = rng.random((pop_size, num_var))
        r2 = rng.random((pop_size, num_var))
        V = (w * V
             + c1 * r1 * (pbest_X - X)
             + c2 * r2 * (gbest_x[None] - X))
        V = np.clip(V, v_min, v_max)
        X = np.clip(X + V, var_min, var_max)

        if len(reset_idx):
            X[reset_idx] = rng.uniform(var_min, var_max,
                                       (len(reset_idx), num_var))
            V[reset_idx] = 0.0

        for i in range(pop_size):
            costs[i], details[i] = cost_fn(X[i])
            if costs[i] < pbest_c[i]:
                pbest_c[i] = costs[i]
                pbest_X[i] = X[i].copy()
                if costs[i] < gbest_c:
                    gbest_c = costs[i]
                    gbest_x = X[i].copy()
                    gbest_d = details[i]

        history[it - 1] = gbest_c
        if verbose and it % 20 == 0:
            print(f'  it {it:3d} cost {gbest_c:9.3f}')

    return {'best_cost': float(gbest_c), 'best_x': gbest_x,
            'details': gbest_d, 'history': history}


def make_cost_function_v2(env,
                          path_length_weight=1.0,
                          start_violation_pen=500.0,
                          goal_violation_pen=500.0,
                          env_violation_pen=50.0,
                          collision_pen=500.0,
                          collision_mode='count'):
    """collision_mode:
       'count' — pen * number of colliding path samples (binary per-sample penalty)
       'depth' — pen * total penetration depth (graded penalty)
    """
    def penetration_depth(path):
        depth = np.zeros(len(path))
        if env.circles:
            d = np.linalg.norm(path[:, None, :] - env.c_centers[None], axis=2)
            depth += np.maximum(env.c_radii[None] - d, 0.0).sum(axis=1)
        if env.rects:
            q = np.abs(path[:, None, :] - env.r_centers[None]) - env.r_half[None]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
            inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
            sd_box = outside + inside            # signed distance to core rect
            depth += np.maximum(env.rr - sd_box, 0.0).sum(axis=1)
        return float(depth.sum())

    def cost(xy_norm):
        path = env.path_from_cp(xy_norm)
        length = env.path_length(path)
        col = env.collision_count(path)
        envv = env.env_violation_count(path)
        start_v = np.linalg.norm(path[0] - env.start) > env.rr
        goal_v = np.linalg.norm(path[-1] - env.goal) > env.rr

        c = path_length_weight * length
        if start_v:
            c += start_violation_pen
        if goal_v:
            c += goal_violation_pen
        c += env_violation_pen * envv
        if collision_mode == 'depth':
            c += collision_pen * penetration_depth(path)
        else:
            c += collision_pen * col
        details = {'length': length, 'collisions': col, 'env_violations': envv,
                   'feasible': (col == 0 and envv == 0
                                and not start_v and not goal_v)}
        return c, details
    return cost
