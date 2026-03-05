import numpy as np

def rho_powerexp(d_km: np.ndarray, a_km: float, b: float) -> np.ndarray:
    return np.exp(- (d_km / a_km) ** b)

def pairwise_dist_km(xy_m: np.ndarray) -> np.ndarray:
    dx = xy_m[:, 0:1] - xy_m[None, :, 0]
    dy = xy_m[:, 1:2] - xy_m[None, :, 1]
    return np.sqrt(dx * dx + dy * dy) / 1000.0

def ok_weights_pinv(
    target_xy_m: np.ndarray,
    gauge_xy_m: np.ndarray,
    a_km: float,
    b: float,
    nugget: float = 0.0,
    rcond: float | None = 1e-12,
):
    """
    Ordinary Kriging weights using pseudo-inverse (SVD).
    Solves: A @ [w, lambda] = rhs  via pinv(A) @ rhs

    target_xy_m: (2,) [E,N] meters
    gauge_xy_m : (n,2) [E,N] meters
    a_km, b    : correlation parameters
    nugget     : optional diagonal stabilization added to C
    rcond      : cutoff for small singular values in pinv (None uses numpy default)
    """
    n = gauge_xy_m.shape[0]

    # Gauge-gauge correlation
    Dg = pairwise_dist_km(gauge_xy_m)
    C = rho_powerexp(Dg, a_km, b)
    if nugget > 0.0:
        C = C + np.eye(n) * nugget

    # Target-gauge correlation vector
    d0 = np.sqrt(((gauge_xy_m - target_xy_m) ** 2).sum(axis=1)) / 1000.0
    c0 = rho_powerexp(d0, a_km, b)

    # OK system matrix
    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = C
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0

    rhs = np.zeros(n + 1, dtype=float)
    rhs[:n] = c0
    rhs[n] = 1.0

    # Pseudo-inverse solve
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    sol = A_pinv @ rhs

    w = sol[:n]
    lam = sol[n]

    return {
        "A": A,
        "A_pinv": A_pinv,
        "weights": w,
        "lambda": lam,
        "sum_weights": float(w.sum()),
        "cond_A": float(np.linalg.cond(A)),
    }

if __name__ == "__main__":
    # Example with your numbers (UTM 15N meters): [Easting, Northing]
    a_km = 35.263931092804
    b = 2.0502286979262

    target_xy = np.array([344311.0562,4305546.271], dtype=float)

    gauge_ids = ["16007", "16067", "16025"]
    gauge_xy = np.array([
        [345047.3408, 4303755.455],
        [342501.5953, 4303616.941],
        [348893.5564, 4305275.372],
    ], dtype=float)

    out = ok_weights_pinv(target_xy, gauge_xy, a_km=a_km, b=b, nugget=0.01, rcond=1e-12)

    np.set_printoptions(precision=10, suppress=True)
    print("cond(A):", out["cond_A"])
    print("weights:", out["weights"])
    print("lambda :", out["lambda"])
    print("sum(w) :", out["sum_weights"])