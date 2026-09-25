"""Dependency-light statistics shared by the metric-relationships layer.

TextGrader keeps highly correlated metrics on purpose (see the project
philosophy: a metric is a sensor, not an opinion) because *where* two normally
related measurements diverge is itself a signal.  This module holds the small
amount of real statistics needed to say that precisely: correlation measures
that do not assume linearity, a closed-form linear/multivariate regression fit
so a "residual" has a defined meaning, an order-independent (orthogonal)
residual for when neither metric is naturally the "cause", and a couple of
corpus-level diagnostics (mutual information, PCA).

Every function here is pure Python plus the standard library.  This is a
deliberate choice, not an oversight: at the reference corpus's scale (tens of
books, at most a few hundred candidate metric columns) the closed-form/O(n^2)
implementations below cost microseconds to low milliseconds, so pulling in
``dcor``/``pingouin``/``hyppo`` for distance correlation, partial correlation
and independence testing would add three new dependencies (each with its own
install/version risk) for numbers this module already computes exactly, the
same way :mod:`textgrader.stats` hand-implements entropy, bimodality and
double-MAD rather than reaching for a package.  ``numpy``/``scipy``/``sklearn``
remain available project-wide and are not needed here.

Nothing in this module knows about ``DocumentAnalysis``, corpus profiles or
:mod:`textgrader.metrics.common`'s ``finding`` shape -- it is pure numeric
plumbing, reused by :mod:`textgrader.metrics.metric_relationships` and safe to
unit test on its own.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence

#: 1 / 0.6745, the same constant :mod:`textgrader.stats` uses to scale a
#: median absolute deviation to a normal-consistent sigma.
MAD_TO_SIGMA = 1.4826


# --------------------------------------------------------------- basic pairing

def paired(xs: Sequence[Any], ys: Sequence[Any]) -> list[tuple[float, float]]:
    """Zip two same-length sequences, keeping only rows where both are finite numbers."""

    out: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if x is None or y is None or isinstance(x, bool) or isinstance(y, bool):
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            out.append((xf, yf))
    return out


def robust_center_scale(values: Sequence[float]) -> tuple[float, float]:
    """Median and a MAD-derived sigma, floored so a constant sample never divides by zero."""

    numbers = [float(v) for v in values if v is not None and not isinstance(v, bool)
              and math.isfinite(float(v))]
    if not numbers:
        return 0.0, 1.0
    median = statistics.median(numbers)
    mad = statistics.median([abs(v - median) for v in numbers])
    scale = mad * MAD_TO_SIGMA
    if scale <= 0:
        spread = statistics.pstdev(numbers) if len(numbers) > 1 else 0.0
        scale = spread if spread > 0 else 1.0
    return median, scale


def robust_z(value: float | None, reference: Sequence[float]) -> float | None:
    """This value's position in ``reference``, in robust standard-deviation units."""

    if value is None or isinstance(value, bool):
        return None
    numbers = [float(v) for v in reference if v is not None and not isinstance(v, bool)
              and math.isfinite(float(v))]
    if not numbers:
        return None
    median, scale = robust_center_scale(numbers)
    return (float(value) - median) / scale


# ------------------------------------------------------------- correlation

def pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    n = len(pairs)
    if n < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    return sxy / math.sqrt(sx * sy)


def rank(values: Sequence[float]) -> list[float]:
    """Average ranks, 1-based, with ties sharing the mean of their positions."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    rx = rank([p[0] for p in pairs])
    ry = rank([p[1] for p in pairs])
    return pearson(list(zip(rx, ry)))


def kendall_tau(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Kendall's tau-b, which corrects for ties -- common among small-corpus rates."""

    n = len(pairs)
    if n < 2:
        return None
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        xi, yi = pairs[i]
        for j in range(i + 1, n):
            dx = xi - pairs[j][0]
            dy = yi - pairs[j][1]
            if dx == 0 and dy == 0:
                tie_x += 1
                tie_y += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt(max(n0 - tie_x, 0) * max(n0 - tie_y, 0))
    if denom <= 0:
        return None
    return (concordant - discordant) / denom


def distance_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Szekely/Rizzo distance correlation: zero iff the variables are independent.

    O(n^2); fine at reference-corpus scale (tens of books).  Implemented
    directly rather than via the ``dcor`` package -- see the module docstring.
    """

    n = len(pairs)
    if n < 4:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    a = [[abs(xs[i] - xs[j]) for j in range(n)] for i in range(n)]
    b = [[abs(ys[i] - ys[j]) for j in range(n)] for i in range(n)]

    def double_center(m: list[list[float]]) -> list[list[float]]:
        row_mean = [sum(row) / n for row in m]
        col_mean = [sum(m[i][j] for i in range(n)) / n for j in range(n)]
        grand = sum(row_mean) / n
        return [[m[i][j] - row_mean[i] - col_mean[j] + grand for j in range(n)] for i in range(n)]

    big_a, big_b = double_center(a), double_center(b)
    dcov2 = sum(big_a[i][j] * big_b[i][j] for i in range(n) for j in range(n)) / (n * n)
    dvar_x2 = sum(v * v for row in big_a for v in row) / (n * n)
    dvar_y2 = sum(v * v for row in big_b for v in row) / (n * n)
    denom = math.sqrt(dvar_x2 * dvar_y2)
    if denom <= 0:
        return None
    return math.sqrt(max(dcov2, 0.0) / denom)


def mutual_information(pairs: Sequence[tuple[float, float]], bins: int = 5) -> float | None:
    """Equal-frequency-binned mutual information, in bits.

    A histogram estimator is biased upward for small samples -- reported
    as an approximation, not a p-value -- but needs no additional package and
    matches :mod:`textgrader.stats`'s own binned entropy estimator.
    """

    n = len(pairs)
    if n < 8:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    k = max(2, min(bins, int(math.sqrt(n))))

    def bin_index(values: Sequence[float]) -> list[int]:
        ordered = sorted(values)
        edges = [_quantile(ordered, i / k) for i in range(1, k)]
        out = []
        for v in values:
            b = 0
            for edge in edges:
                if v > edge:
                    b += 1
                else:
                    break
            out.append(min(b, k - 1))
        return out

    bx, by = bin_index(xs), bin_index(ys)
    joint: dict[tuple[int, int], int] = {}
    cx: dict[int, int] = {}
    cy: dict[int, int] = {}
    for i, j in zip(bx, by):
        joint[(i, j)] = joint.get((i, j), 0) + 1
        cx[i] = cx.get(i, 0) + 1
        cy[j] = cy.get(j, 0) + 1
    mi = 0.0
    for (i, j), count in joint.items():
        pij = count / n
        pi = cx[i] / n
        pj = cy[j] / n
        if pij > 0 and pi > 0 and pj > 0:
            mi += pij * math.log2(pij / (pi * pj))
    return max(mi, 0.0)


def quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted sequence."""

    return _quantile(ordered, q)


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# --------------------------------------------------------------- linear models

def solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float] | None:
    """Solve ``matrix @ x = vector`` by Gauss-Jordan elimination with partial pivoting.

    ``None`` on a singular system.  Small (p <= ~8), so no numpy is needed.
    """

    p = len(vector)
    if p == 0 or len(matrix) != p:
        return None
    aug = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(p):
        pivot_row = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-10:
            return None
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(p):
            if r != col:
                factor = aug[r][col]
                if factor:
                    aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(p + 1)]
    return [aug[i][p] for i in range(p)]


def ols_fit(rows: Sequence[Sequence[float]], y: Sequence[float]) -> list[float] | None:
    """Ordinary least squares via the normal equations. ``rows`` includes the constant column."""

    n = len(rows)
    p = len(rows[0]) if rows else 0
    if n < p or p == 0:
        return None
    gram = [[sum(rows[k][i] * rows[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    # A tiny ridge term keeps near-collinear predictors (words/sentence and
    # sentences/paragraph both predicting words/paragraph, say) from producing
    # a numerically singular system, without perceptibly changing a
    # well-conditioned fit.
    trace = sum(gram[i][i] for i in range(p)) or 1.0
    ridge = 1e-9 * trace / p
    for i in range(p):
        gram[i][i] += ridge
    target = [sum(rows[k][i] * y[k] for k in range(n)) for i in range(p)]
    return solve_linear(gram, target)


def fit_residual_model(rows: Sequence[Mapping[str, Any]], target_key: str,
                       predictor_keys: Sequence[str], min_n: int = 6) -> dict[str, Any] | None:
    """Fit ``target_key ~ predictor_keys`` over corpus rows and score each row's residual.

    Returns the coefficients (intercept first), the residual distribution
    (median/robust scale) needed to standardize a new document's residual, and
    the in-sample residuals themselves (used as the reference distribution for
    percentile/threshold findings) -- or ``None`` when there is not enough
    jointly-observed data to fit a stable model.
    """

    data: list[tuple[float, list[float]]] = []
    for row in rows:
        target = row.get(target_key)
        preds = [row.get(key) for key in predictor_keys]
        if target is None or any(p is None for p in preds):
            continue
        try:
            target_f = float(target)
            preds_f = [float(p) for p in preds]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(target_f) or not all(math.isfinite(p) for p in preds_f):
            continue
        data.append((target_f, preds_f))
    n = len(data)
    if n < max(min_n, len(predictor_keys) + 2):
        return None
    design = [[1.0, *preds] for _, preds in data]
    y = [target for target, _ in data]
    coefficients = ols_fit(design, y)
    if coefficients is None:
        return None
    residuals = [y[i] - sum(c * x for c, x in zip(coefficients, design[i])) for i in range(n)]
    median, scale = robust_center_scale(residuals)
    return {
        "coefficients": coefficients, "predictor_keys": list(predictor_keys),
        "target_key": target_key, "n": n,
        "residual_median": median, "residual_scale": scale,
        "residuals": residuals,
    }


def predict(model: Mapping[str, Any], predictor_values: Sequence[float]) -> float:
    coefficients = model["coefficients"]
    return coefficients[0] + sum(c * v for c, v in zip(coefficients[1:], predictor_values))


def standardized_residual(model: Mapping[str, Any], target_value: float,
                          predictor_values: Sequence[float]) -> float:
    raw = target_value - predict(model, predictor_values)
    return (raw - model["residual_median"]) / model["residual_scale"]


def residual_reference_percentile(model: Mapping[str, Any], percentile: float) -> float:
    """The ``percentile``-th value of the fit's in-sample |standardized residual|."""

    scale = model["residual_scale"]
    magnitudes = sorted(abs((r - model["residual_median"]) / scale) for r in model["residuals"])
    return _quantile(magnitudes, percentile / 100.0)


# ------------------------------------------------------------ orthogonal (TLS)

def fit_orthogonal_model(pairs: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
    """Total-least-squares (orthogonal) line through standardized ``pairs``.

    Neither variable is treated as the cause: the fitted line minimizes
    perpendicular distance, so the resulting residual is symmetric in A and B
    (swapping the two columns and refitting gives the same distances, up to
    floating-point noise) -- the "symmetric residual" the task spec asks for,
    as distinct from the two directional "A given B"/"B given A" fits.
    """

    n = len(pairs)
    if n < 6:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs) / n
    syy = sum((y - my) ** 2 for y in ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs) / n
    # Largest eigenvector of the 2x2 covariance matrix, closed form.
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(trace * trace / 4 - det, 0.0))
    lam = trace / 2 + disc
    if abs(sxy) < 1e-12 and abs(sxx - lam) < 1e-12:
        direction = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
    else:
        vx, vy = sxy, lam - sxx
        norm = math.hypot(vx, vy)
        direction = (vx / norm, vy / norm) if norm > 0 else (1.0, 0.0)
    # Perpendicular offset of each point from the line through (mx, my).
    normal = (-direction[1], direction[0])
    offsets = [((x - mx) * normal[0] + (y - my) * normal[1]) for x, y in pairs]
    median, scale = robust_center_scale(offsets)
    return {"mean": (mx, my), "direction": direction, "normal": normal,
           "offset_median": median, "offset_scale": scale, "offsets": offsets, "n": n}


def orthogonal_residual(model: Mapping[str, Any], point: tuple[float, float]) -> float:
    mx, my = model["mean"]
    nx, ny = model["normal"]
    offset = (point[0] - mx) * nx + (point[1] - my) * ny
    return (offset - model["offset_median"]) / model["offset_scale"]


# ----------------------------------------------------------------- diagnostics

def bootstrap_correlation_stability(pairs: Sequence[tuple[float, float]], samples: int,
                                    seed: int, method=spearman) -> float | None:
    """Std dev of ``method`` across seeded bootstrap resamples: lower = a more stable estimate."""

    n = len(pairs)
    if n < 6 or samples < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        value = method(sample)
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def leave_one_out_stability(pairs: Sequence[tuple[float, float]], method=spearman) -> float | None:
    """Std dev of ``method`` recomputed with each observation held out in turn."""

    n = len(pairs)
    if n < 6:
        return None
    values = []
    for i in range(n):
        held_out = pairs[:i] + pairs[i + 1:]
        value = method(held_out)
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def partial_correlation(a_values: Sequence[float], b_values: Sequence[float],
                        controls: Sequence[Sequence[float]]) -> float | None:
    """Correlation of A and B once each has been linearly regressed on ``controls``."""

    n = len(a_values)
    if n != len(b_values) or any(len(c) != n for c in controls):
        return None
    if n < len(controls) + 3:
        return None
    design = [[1.0, *(control[i] for control in controls)] for i in range(n)]
    coeff_a = ols_fit(design, list(a_values))
    coeff_b = ols_fit(design, list(b_values))
    if coeff_a is None or coeff_b is None:
        return None
    resid_a = [a_values[i] - sum(c * x for c, x in zip(coeff_a, design[i])) for i in range(n)]
    resid_b = [b_values[i] - sum(c * x for c, x in zip(coeff_b, design[i])) for i in range(n)]
    return pearson(list(zip(resid_a, resid_b)))


def principal_components(rows: Sequence[Sequence[float]], max_components: int = 3,
                         iterations: int = 100) -> dict[str, Any] | None:
    """Explained-variance shares of a standardized matrix's leading components.

    Power iteration with deflation -- diagnostic only (see the task spec: PCA
    here is never used to drop or replace an original metric), so an exact
    eigendecomposition is not needed, only the top few components' shares.
    """

    n = len(rows)
    if n < 4 or not rows or not rows[0]:
        return None
    p = len(rows[0])
    means = [statistics.fmean(row[j] for row in rows) for j in range(p)]
    stds = []
    for j in range(p):
        spread = statistics.pstdev([row[j] for row in rows])
        stds.append(spread if spread > 0 else 1.0)
    standardized = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in rows]
    # Covariance matrix of the standardized columns.
    cov = [[sum(standardized[k][i] * standardized[k][j] for k in range(n)) / n
           for j in range(p)] for i in range(p)]
    total_variance = sum(cov[i][i] for i in range(p))
    if total_variance <= 0:
        return None
    shares = []
    working = [row[:] for row in cov]
    for _ in range(min(max_components, p)):
        vector = [1.0] * p
        eigenvalue = 0.0
        for _ in range(iterations):
            new_vector = [sum(working[i][j] * vector[j] for j in range(p)) for i in range(p)]
            norm = math.sqrt(sum(v * v for v in new_vector))
            if norm < 1e-12:
                break
            vector = [v / norm for v in new_vector]
            eigenvalue = norm
        eigenvalue = max(eigenvalue, 0.0)
        shares.append(eigenvalue / total_variance)
        # Deflate: remove this component's contribution before the next pass.
        for i in range(p):
            for j in range(p):
                working[i][j] -= eigenvalue * vector[i] * vector[j]
    return {"explained_variance_share": shares, "n_features": p, "n_rows": n,
           "cumulative_share": sum(shares)}


__all__ = [
    "MAD_TO_SIGMA", "paired", "robust_center_scale", "robust_z",
    "pearson", "rank", "spearman", "kendall_tau", "distance_correlation", "quantile",
    "mutual_information", "solve_linear", "ols_fit", "fit_residual_model", "predict",
    "standardized_residual", "residual_reference_percentile",
    "fit_orthogonal_model", "orthogonal_residual",
    "bootstrap_correlation_stability", "leave_one_out_stability",
    "partial_correlation", "principal_components",
]
