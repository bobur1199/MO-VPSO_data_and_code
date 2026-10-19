"""
metrics.py
==========
Performance indicators for two-objective minimization problems.

  hypervolume_2d   exact 2-D hypervolume with respect to a reference point
  reference_point  1.1 x the per-objective maximum over a set of fronts
  nondominated     indices of the Pareto non-dominated rows of a matrix
  igd              inverted generational distance (used for ZDT verification)
  igd_plus         IGD+ (Ishibuchi et al., 2015), Pareto-compliant variant of IGD
"""
import numpy as np


def hypervolume_2d(F, ref):
    """Exact 2-D hypervolume (both objectives minimized) of the point set F
    with respect to the reference point ref. Points that do not dominate ref
    contribute nothing."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    pts = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return max(hv, 0.0)


def reference_point(fronts, margin=1.1):
    """Reference point = margin x the per-objective maximum over all points of
    all non-empty fronts (list of (n_i, 2) arrays)."""
    pts = [np.asarray(f, float) for f in fronts if len(f)]
    if not pts:
        return np.array([1000.0, 1.0])
    allF = np.vstack(pts)
    return allF.max(axis=0) * margin


def nondominated(F):
    """Indices of the non-dominated rows of F (minimization)."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return np.array([], dtype=int)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dominated = (le & lt).any(axis=0)
    return np.where(~dominated)[0]


def igd(front, true_front):
    """Inverted generational distance: mean Euclidean distance from each point
    of the true front to its nearest point of the approximation."""
    front = np.asarray(front, float)
    if len(front) == 0:
        return np.inf
    d = np.linalg.norm(true_front[:, None, :] - front[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def igd_plus(front, ref_front, ideal, nadir):
    """IGD+ (Ishibuchi, Masuda, Tanigaki and Nojima, EMO 2015).

    Objectives are normalized to [0, 1] between `ideal` and `nadir`. For each
    reference point z, d+(z, a) = || max(a - z, 0) ||, and IGD+ is the mean
    over z of the minimum d+ over the approximation set."""
    front = np.asarray(front, float)
    if len(front) == 0:
        return np.nan
    span = np.where(nadir - ideal <= 0, 1.0, nadir - ideal)
    A = (front - ideal) / span
    Z = (np.asarray(ref_front, float) - ideal) / span
    diff = np.maximum(A[None, :, :] - Z[:, None, :], 0.0)
    d = np.sqrt((diff ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())
