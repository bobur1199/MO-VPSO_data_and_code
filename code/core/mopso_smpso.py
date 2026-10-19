"""
mopso_smpso.py
==============
Reference implementations of two published multi-objective PSO algorithms,
used as external baselines.

MOPSO — Coello Coello, C.A.; Pulido, G.T.; Lechuga, M.S. "Handling multiple
objectives with particle swarm optimization." IEEE Trans. Evol. Comput.
2004, 8(3), 256-279.
  * external repository bounded by an ADAPTIVE GRID over objective space
  * leader chosen by roulette wheel over hypercubes: a hypercube containing
    fewer non-dominated solutions gets proportionally higher selection
    probability (fitness 10/count), then a member is drawn uniformly from it
  * repository pruned by deleting members of the most crowded hypercubes
  * w = 0.4, c1 = c2 = 1.0 as specified in the paper
  * uniform mutation applied to a fraction of the swarm, the fraction
    decaying with iteration (paper Section IV)

SMPSO — Nebro, A.J.; Durillo, J.J.; Garcia-Nieto, J.; Coello Coello, C.A.;
Luna, F.; Alba, E. "SMPSO: A new PSO-based metaheuristic for multi-objective
optimization." IEEE MCDM 2009, pp. 66-73.
  * constriction coefficient chi computed from phi = c1 + c2
  * VELOCITY CONSTRICTION: per-variable delta_j = (upper_j - lower_j)/2 with
    velocities limited to [-delta_j, delta_j] (this is the mechanism SMPSO
    introduced to prevent the velocity explosion of MOPSO)
  * c1, c2 ~ U(1.5, 2.5); w ~ U(0.1, 0.5)
  * polynomial mutation applied to 15% of particles
  * leader archive bounded by crowding distance, leaders chosen by binary
    tournament on crowding distance

Both use Deb's constrained-domination principle for constraint handling, so
they are directly comparable with MO-VPSO and NSGA-II.
"""
import numpy as np
from nsga2 import fast_non_dominated_sort, crowding_distance, poly_mutation


def _nondominated(F, V):
    """Indices of the constrained-non-dominated set."""
    fronts = fast_non_dominated_sort(F, V)
    return fronts[0] if len(fronts) else np.array([], dtype=int)


# ---------------------------------------------------------------- MOPSO ----
class AdaptiveGrid:
    """Adaptive grid archive of Coello Coello et al. (2004)."""

    def __init__(self, divisions=30, cap=100):
        self.div = divisions
        self.cap = cap
        self.X = np.empty((0, 0))
        self.F = np.empty((0, 2))
        self.V = np.empty((0,))

    def _hypercubes(self):
        if len(self.F) == 0:
            return np.zeros(0, dtype=int)
        lo, hi = self.F.min(0), self.F.max(0)
        span = np.where(hi - lo <= 0, 1.0, hi - lo)
        idx = np.floor((self.F - lo) / span * (self.div - 1e-9)).astype(int)
        return idx[:, 0] * self.div + idx[:, 1]

    def add(self, X, F, V):
        if self.X.shape[1] == 0:
            self.X = np.empty((0, X.shape[1]))
        allX = np.vstack([self.X, X])
        allF = np.vstack([self.F, F])
        allV = np.concatenate([self.V, V])
        keep = _nondominated(allF, allV)
        self.X, self.F, self.V = allX[keep], allF[keep], allV[keep]
        # deduplicate
        _, u = np.unique(np.round(self.F, 10), axis=0, return_index=True)
        self.X, self.F, self.V = self.X[u], self.F[u], self.V[u]
        if len(self.X) > self.cap:
            # delete from the most crowded hypercubes first
            hc = self._hypercubes()
            counts = {h: (hc == h).sum() for h in set(hc.tolist())}
            weight = np.array([counts[h] for h in hc], float)
            order = np.argsort(-weight)          # crowded first
            drop = order[:len(self.X) - self.cap]
            mask = np.ones(len(self.X), bool); mask[drop] = False
            self.X, self.F, self.V = self.X[mask], self.F[mask], self.V[mask]

    def leaders(self, n, rng):
        """Roulette over hypercubes: fitness 10/count favours sparse cells."""
        if len(self.X) == 0:
            return None
        hc = self._hypercubes()
        cells = np.array(sorted(set(hc.tolist())))
        counts = np.array([(hc == c).sum() for c in cells], float)
        fit = 10.0 / counts
        prob = fit / fit.sum()
        chosen = rng.choice(len(cells), size=n, p=prob)
        out = np.empty((n, self.X.shape[1]))
        for i, ci in enumerate(chosen):
            members = np.where(hc == cells[ci])[0]
            out[i] = self.X[rng.choice(members)]
        return out


def run_mopso(eval_fn, D, lo, hi, seed, pop_size=50, max_iter=100,
              archive_cap=100, divisions=30):
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    vmax = (hi - lo)                       # paper does not clamp tightly
    X = rng.uniform(lo, hi, (pop_size, D))
    V = np.zeros((pop_size, D))
    F, Vc = eval_fn(X)
    pbX, pbF, pbV = X.copy(), F.copy(), Vc.copy()
    arch = AdaptiveGrid(divisions, archive_cap)
    arch.add(X, F, Vc)

    for t in range(1, max_iter + 1):
        L = arch.leaders(pop_size, rng)
        if L is None:
            L = pbX.copy()
        w, c1, c2 = 0.4, 1.0, 1.0
        V = (w * V + c1 * rng.random((pop_size, D)) * (pbX - X)
             + c2 * rng.random((pop_size, D)) * (L - X))
        V = np.clip(V, -vmax, vmax)
        X = X + V
        # reflect at bounds, as in the paper
        below = X < lo; above = X > hi
        X = np.where(below, lo, np.where(above, hi, X))
        V = np.where(below | above, -V, V)

        # uniform mutation on a decaying fraction of the swarm
        frac = (1.0 - t / max_iter) ** (5.0 / 1.0)
        nmut = int(pop_size * frac)
        if nmut > 0:
            idx = rng.choice(pop_size, nmut, replace=False)
            dims = rng.integers(0, D, nmut)
            rangev = (hi - lo) * frac
            for i, dm in zip(idx, dims):
                X[i, dm] = np.clip(
                    X[i, dm] + rng.uniform(-rangev[dm]/2, rangev[dm]/2),
                    lo[dm], hi[dm])

        F, Vc = eval_fn(X)
        for i in range(pop_size):
            if _dom_single(F[i], Vc[i], pbF[i], pbV[i]) or (
                    rng.random() < 0.5 and not _dom_single(pbF[i], pbV[i], F[i], Vc[i])):
                pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), Vc[i]
        arch.add(X, F, Vc)

    feas = arch.V <= 1e-9
    return {'front_F': arch.F[feas], 'front_X': arch.X[feas],
            'n_feasible': int(feas.sum())}


def _dom_single(fa, va, fb, vb):
    if va <= 0 and vb > 0: return True
    if va > 0 and vb <= 0: return False
    if va > 0 and vb > 0: return va < vb
    return np.all(fa <= fb) and np.any(fa < fb)


# ---------------------------------------------------------------- SMPSO ----
def run_smpso(eval_fn, D, lo, hi, seed, pop_size=50, max_iter=100,
              archive_cap=100):
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    delta = (hi - lo) / 2.0                      # velocity constriction

    X = rng.uniform(lo, hi, (pop_size, D))
    V = np.zeros((pop_size, D))
    F, Vc = eval_fn(X)
    pbX, pbF, pbV = X.copy(), F.copy(), Vc.copy()

    aX, aF, aV = X.copy(), F.copy(), Vc.copy()
    keep = _nondominated(aF, aV)
    aX, aF, aV = aX[keep], aF[keep], aV[keep]

    for t in range(max_iter):
        # leader selection: binary tournament on crowding distance
        if len(aX):
            cd = crowding_distance(aF)
            i1 = rng.integers(0, len(aX), pop_size)
            i2 = rng.integers(0, len(aX), pop_size)
            pick = np.where(cd[i1] >= cd[i2], i1, i2)
            L = aX[pick]
        else:
            L = pbX.copy()

        c1 = rng.uniform(1.5, 2.5, (pop_size, 1))
        c2 = rng.uniform(1.5, 2.5, (pop_size, 1))
        w = rng.uniform(0.1, 0.5, (pop_size, 1))
        rho = c1 + c2
        # constriction coefficient (jMetal SMPSO): chi = 1 when rho <= 4
        with np.errstate(invalid='ignore'):
            chi_big = 2.0 / np.abs(2.0 - rho - np.sqrt(np.maximum(rho**2 - 4.0*rho, 0.0)))
        chi = np.where(rho > 4.0, chi_big, 1.0)

        V = chi * (w * V
                   + c1 * rng.random((pop_size, D)) * (pbX - X)
                   + c2 * rng.random((pop_size, D)) * (L - X))
        V = np.clip(V, -delta, delta)
        X = X + V
        below = X < lo; above = X > hi
        X = np.where(below, lo, np.where(above, hi, X))
        V = np.where(below | above, -0.001 * V, V)

        # polynomial mutation on 15% of particles
        nm = max(1, int(0.15 * pop_size))
        for i in rng.choice(pop_size, nm, replace=False):
            X[i] = poly_mutation(X[i], lo, hi, rng, eta=20.0, pm=1.0/D)

        F, Vc = eval_fn(X)
        for i in range(pop_size):
            if _dom_single(F[i], Vc[i], pbF[i], pbV[i]):
                pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), Vc[i]
            elif not _dom_single(pbF[i], pbV[i], F[i], Vc[i]) and rng.random() < 0.5:
                pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), Vc[i]

        aX = np.vstack([aX, X]); aF = np.vstack([aF, F]); aV = np.concatenate([aV, Vc])
        keep = _nondominated(aF, aV)
        aX, aF, aV = aX[keep], aF[keep], aV[keep]
        _, u = np.unique(np.round(aF, 10), axis=0, return_index=True)
        aX, aF, aV = aX[u], aF[u], aV[u]
        if len(aX) > archive_cap:
            cd = crowding_distance(aF)
            order = np.argsort(-cd)[:archive_cap]
            aX, aF, aV = aX[order], aF[order], aV[order]

    feas = aV <= 1e-9
    return {'front_F': aF[feas], 'front_X': aX[feas],
            'n_feasible': int(feas.sum())}
