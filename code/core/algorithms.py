"""
algorithms.py
=============
Particle swarm parameter schedules, the batched single-objective path cost,
the benchmark functions and a population-batched single-objective PSO runner.

Schedules (Table 1 of the manuscript; t = iteration, T = maximum iterations)

  PSO       w = 0.7298,                     c1 = c2 = 1.49618        (Clerc & Kennedy)
  IPSO      w = 0.9 -> 0.4 linear,          c1 = c2 = 2.0            (Shi & Eberhart)
  APSO      w = 0.9 -> 0.4 linear,          c1 2.5 -> 0.5, c2 0.5 -> 2.5 (Ratnaweera et al.)
  RandWPSO  w = 0.5 + U(0,1)/2,             c1 = c2 = 1.49618        (Eberhart & Shi)
  VPSO      w = 0.9 -> 0.4 linear (Eq. 9),  C1, C2 from Eq. (10) with C_max = 2.0,
            C_min = 0.5, transfer exponent gamma = 1.75; worst 20 % of the swarm
            re-initialized every 25 iterations (Eq. 13)
  VPSO-noRestart   VPSO without the periodic restart (ablation, Section 3.5)

All schedules share initialization, velocity clamping v_max = k (x_max - x_min),
position clipping, evaluation budget and random-number seeding, so that runs
with the same seed start from the same swarm.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Batched path-planning cost (single-objective weighted-sum formulation)
# ---------------------------------------------------------------------------
def make_batched_path_cost(env, collision_pen=500.0, env_pen=50.0,
                           collision_mode='depth'):
    """Return cost(X) for a whole population X (P, D) of normalized control
    points. cost = path length + env_pen * (#samples outside the workspace)
    + collision_pen * (penetration depth  [collision_mode='depth']
                       or #colliding samples [collision_mode='count'])."""
    W, H, rr = env.width, env.height, env.rr
    scale = np.array([W, H])
    B = env.basis                                    # (R, n_pts)

    def cost(X):
        P = X.shape[0]
        cp = X.reshape(P, -1, 2) * scale
        pts = np.concatenate(
            [np.broadcast_to(env.start, (P, 1, 2)), cp,
             np.broadcast_to(env.goal, (P, 1, 2))], axis=1)   # (P, n, 2)
        paths = np.einsum('rn,pnd->prd', B, pts)              # (P, R, 2)
        outx = (paths[..., 0] < rr) | (paths[..., 0] > W - rr)
        outy = (paths[..., 1] < rr) | (paths[..., 1] > H - rr)
        envv = (outx | outy).sum(axis=1)
        paths[..., 0] = np.clip(paths[..., 0], rr, W - rr)
        paths[..., 1] = np.clip(paths[..., 1], rr, H - rr)

        seg = np.diff(paths, axis=1)
        lengths = np.linalg.norm(seg, axis=2).sum(axis=1)

        hit = np.zeros(paths.shape[:2], dtype=bool)
        depth = np.zeros(paths.shape[:2])
        if env.circles:
            d = np.linalg.norm(paths[:, :, None, :] - env.c_centers[None, None],
                               axis=3)
            hit |= (d <= env.c_radii[None, None]).any(axis=2)
            depth += np.maximum(env.c_radii[None, None] - d, 0).sum(axis=2)
        if env.rects:
            q = (np.abs(paths[:, :, None, :] - env.r_centers[None, None])
                 - env.r_half[None, None])
            outside = np.linalg.norm(np.maximum(q, 0), axis=3)
            inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0)
            sd = outside + inside
            hit |= (sd <= rr).any(axis=2)
            depth += np.maximum(rr - sd, 0).sum(axis=2)
        col_count = hit.sum(axis=1)

        c = lengths + env_pen * envv
        if collision_mode == 'depth':
            c = c + collision_pen * depth.sum(axis=1)
        else:
            c = c + collision_pen * col_count
        aux = {'length': lengths, 'collisions': col_count,
               'feasible': (col_count == 0) & (envv == 0)}
        return c, aux
    return cost


# ---------------------------------------------------------------------------
# Benchmark functions (batched; X: (P, D)); global minimum 0 for all
# ---------------------------------------------------------------------------
def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1 + (X ** 2).sum(1) / 4000 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


def bohachevsky(X):
    x, y = X[:, 0], X[:, 1]
    return (x**2 + 2*y**2 - 0.3*np.cos(3*np.pi*x)
            - 0.4*np.cos(4*np.pi*y) + 0.7)


def bukin6(X):
    x, y = X[:, 0], X[:, 1]
    return 100*np.sqrt(np.abs(y - 0.01*x**2)) + 0.01*np.abs(x + 10)


def ackley(X):
    d = X.shape[1]
    return (-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/d))
            - np.exp(np.cos(2*np.pi*X).sum(1)/d) + 20 + np.e)


def rastrigin(X):
    return 10*X.shape[1] + (X**2 - 10*np.cos(2*np.pi*X)).sum(1)


def levy(X):
    w = 1 + (X - 1)/4
    t1 = np.sin(np.pi*w[:, 0])**2
    t3 = (w[:, -1]-1)**2 * (1+np.sin(2*np.pi*w[:, -1])**2)
    mid = ((w[:, :-1]-1)**2 * (1+10*np.sin(np.pi*w[:, :-1]+1)**2)).sum(1)
    return t1 + mid + t3


BENCHMARKS = {
    'Griewank':    dict(f=griewank,    dim=30, lo=-600.0,   hi=600.0),
    'Bohachevsky': dict(f=bohachevsky, dim=2,  lo=-100.0,   hi=100.0),
    'Bukin':       dict(f=bukin6,      dim=2,
                        lo=np.array([-15.0, -3.0]), hi=np.array([-5.0, 3.0])),
    'Ackley':      dict(f=ackley,      dim=30, lo=-32.768,  hi=32.768),
    'Rastrigin':   dict(f=rastrigin,   dim=30, lo=-5.12,    hi=5.12),
    'Levy':        dict(f=levy,        dim=30, lo=-10.0,    hi=10.0),
}


# ---------------------------------------------------------------------------
# Parameter schedules
# ---------------------------------------------------------------------------
def transfer_schedule(t, T, gamma=1.75, C_max=2.0, C_min=0.5):
    """Learning factors of Eq. (10): C1 + C2 = C_max + C_min for every t."""
    p = np.clip(t / T, 0, 1) ** gamma
    return C_max - p*(C_max - C_min), C_min + p*(C_max - C_min)


def linear_w(t, T, rng=None, w_max=0.9, w_min=0.4):
    """Linearly decreasing inertia weight of Eq. (9)."""
    return w_max - (w_max - w_min) * t / T


ALGOS = {
    'PSO': dict(
        w=lambda t, T, rng: 0.7298,
        c=lambda t, T: (1.49618, 1.49618), reset=None),
    'IPSO': dict(
        w=lambda t, T, rng: 0.9 - 0.5 * t / T,
        c=lambda t, T: (2.0, 2.0), reset=None),
    'APSO': dict(
        w=lambda t, T, rng: 0.9 - 0.5 * t / T,
        c=lambda t, T: (2.5 - 2.0*t/T, 0.5 + 2.0*t/T), reset=None),
    'RandWPSO': dict(
        w=lambda t, T, rng: 0.5 + rng.random() / 2.0,
        c=lambda t, T: (1.49618, 1.49618), reset=None),
    'VPSO': dict(
        w=lambda t, T, rng: 0.9 - 0.5 * t / T,
        c=lambda t, T: transfer_schedule(t, T, 1.75), reset=25),
}

#: The five schedules compared in the manuscript, in table order.
SCHEDULES = ['PSO', 'IPSO', 'APSO', 'RandWPSO', 'VPSO']

#: Ablation variant: the proposed schedule without the periodic restart.
VARIANTS = {
    'VPSO-noRestart': dict(ALGOS['VPSO'], reset=None),
}


def get_schedule(algo):
    """Accept a schedule name (ALGOS or VARIANTS) or a schedule dict."""
    if isinstance(algo, dict):
        return algo
    return ALGOS[algo] if algo in ALGOS else VARIANTS[algo]


# ---------------------------------------------------------------------------
# Population-batched single-objective PSO (shared by every schedule)
# ---------------------------------------------------------------------------
def run_pso(cost, num_var, lo, hi, algo, seed, pop_size=300, max_iter=100,
            k=0.05, reset_frac=0.20, record_saturation=False):
    """Minimize cost(X) -> (costs, aux) with the schedule `algo`.

    lo, hi: scalars or (D,) arrays. k: velocity clamp factor (Eq. 2).
    Returns dict(best_cost, best_x, aux, history[, saturation]) where
    `saturation` is the fraction of clipped velocity components per iteration.
    """
    rng = np.random.default_rng(seed)
    cfg = get_schedule(algo)
    lo = np.broadcast_to(np.asarray(lo, float), (num_var,)).copy()
    hi = np.broadcast_to(np.asarray(hi, float), (num_var,)).copy()
    v_max = k * (hi - lo)

    X = rng.uniform(lo, hi, (pop_size, num_var))
    V = np.zeros((pop_size, num_var))
    costs, _ = cost(X)
    pbX, pbC = X.copy(), costs.copy()
    g = int(np.argmin(pbC))
    gbX, gbC = pbX[g].copy(), float(pbC[g])
    history = np.empty(max_iter)
    saturation = np.empty(max_iter)

    for t in range(1, max_iter + 1):
        w = cfg['w'](t, max_iter, rng)
        c1, c2 = cfg['c'](t, max_iter)
        r1 = rng.random((pop_size, num_var))
        r2 = rng.random((pop_size, num_var))
        V = w*V + c1*r1*(pbX - X) + c2*r2*(gbX[None] - X)
        saturation[t - 1] = float((np.abs(V) > v_max[None]).mean())
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lo, hi)

        if cfg['reset'] and t % cfg['reset'] == 0:
            n_reset = max(1, int(reset_frac * pop_size))
            worst = np.argpartition(costs, -n_reset)[-n_reset:]
            X[worst] = rng.uniform(lo, hi, (n_reset, num_var))
            V[worst] = 0.0

        costs, _ = cost(X)
        imp = costs < pbC
        pbC[imp] = costs[imp]
        pbX[imp] = X[imp]
        g = int(np.argmin(pbC))
        if pbC[g] < gbC:
            gbC = float(pbC[g])
            gbX = pbX[g].copy()
        history[t - 1] = gbC

    _, fa = cost(gbX[None])
    aux = {key: v[0] for key, v in fa.items()} if fa else {}
    out = {'best_cost': gbC, 'best_x': gbX, 'aux': aux, 'history': history}
    if record_saturation:
        out['saturation'] = saturation
    return out
