import numpy as np

# Computes a normalized Critical Lower Bound (CLB) for each node.
# CLB is the earliest possible completion time given precedence constraints in A,
# assuming operations start as early as possible and only job-order dependencies.
def clb(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    dur = X[:, -1].astype(np.float32)          # (T,)
    preds = (A > 0)                             # edge mask
    indeg = preds.sum(axis=0).astype(int)       # (T,)

    q = list(np.where(indeg == 0)[0])
    clb = dur.copy()

    for u in q:                                 # q grows while iterating
        for v in np.where(preds[u])[0]:         # u -> v
            clb[v] = max(clb[v], clb[u] + dur[v])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(q) != A.shape[0]:
        raise ValueError("A has a cycle (CLB needs a DAG).")

    clb = clb / clb.max()
    clb = clb.reshape(-1, 1)
    return clb