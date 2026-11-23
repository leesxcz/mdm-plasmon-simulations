"""Dispersion, propagation and vertical profiles of the MDM gap modes."""

import math
import numpy as np

from . import materials, storage

SINHC_SWITCH = 1e-2  # sinhc series for |v| < 1e-2
H_SERIES_RADIUS = 0.5
H_SERIES_COEF = tuple(2.0 * n / math.factorial(2 * n + 1) for n in range(1, 9))
CALLS = {"newton": 0}  # scalar root-finder calls of this module
GAP_MODES = ("central gap", "plateau gap")
NEWTON = {"newton_relative_step": 1e-14, "newton_max_iterations": 50}
A_GAP_UM = 0.003  # a = 3 nm
STEP_NM = 1.0  # largest thickness change of any layer between continuation points
EPS_KEYS = {
    materials.WATER: "water",
    materials.AG: "silver",
    materials.GAP: "gap",
    materials.GLASS: "glass",
    materials.CR: "chromium",
}


def cj(z):
    z = complex(z)
    return {"re": z.real, "im": z.imag}


def cval(d):
    return complex(d["re"], d["im"])


# ---------------- layer algebra (k0 explicit, 1/um) ----------------


def kappa(beta, eps, k0):
    """kappa = sqrt(beta^2 - eps k0^2), Re kappa >= 0."""
    beta = np.asarray(beta, dtype=np.complex128)
    k = np.sqrt(beta * beta - eps * k0 * k0)
    return np.where(k.real >= 0, k, -k)


def sinhc(v):
    """sinh(v)/v; series 1 + v^2/6 + v^4/120 + v^6/5040 for |v| < 1e-2, sinhc(0) = 1."""
    v = np.asarray(v, dtype=np.complex128)
    w = v * v
    series = 1.0 + w / 6.0 + w * w / 120.0 + w * w * w / 5040.0
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.sinh(v) / v
    return np.where(np.abs(v) < SINHC_SWITCH, series, direct)


def hfun(v):
    """h(v) = (v cosh v - sinh v)/v^3; series sum_{n=1..8} 2n/(2n+1)! v^(2n-2) for |v| < 0.5."""
    v = np.asarray(v, dtype=np.complex128)
    w = v * v
    series = np.zeros_like(v)
    for coef in reversed(H_SERIES_COEF):
        series = series * w + coef
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        direct = (v * np.cosh(v) - np.sinh(v)) / (v * v * v)
    return np.where(np.abs(v) < H_SERIES_RADIUS, series, direct)


def layer(beta, eps, length, k0):
    """c = cosh(kappa l), s = l sinhc(kappa l), c' = beta l s, s' = beta l^3 h(kappa l), zeta, kappa."""
    beta = np.asarray(beta, dtype=np.complex128)
    zeta = beta * beta - eps * k0 * k0
    k = kappa(beta, eps, k0)
    v = k * length
    c = np.cosh(v)
    s = length * sinhc(v)
    cp = beta * length * s
    sp = beta * length**3 * hfun(v)
    return c, s, cp, sp, zeta, k


def snc(x):
    """sin(x)/x for real x, 1 at x = 0."""
    return 1.0 if x == 0.0 else math.sin(x) / x


def em1(x):
    """(exp(x) - 1)/x for real x, 1 at x = 0."""
    return 1.0 if x == 0.0 else math.expm1(x) / x


def em1n(x):
    """(1 - exp(-x))/x for real x, 1 at x = 0."""
    return 1.0 if x == 0.0 else -math.expm1(-x) / x


# ---------------- stacks and the mode condition ----------------


class ModeStack:
    """Substrate | layers (name, eps, t_um) bottom -> top | superstrate; leaky=True puts the substrate on the
    growing (outgoing) branch."""

    def __init__(self, name, eps_sub, layers, eps_sup, leaky=False):
        self.name = name
        self.eps_sub = complex(eps_sub)
        self.layers = [(str(n), complex(e), float(tt)) for n, e, tt in layers]
        self.eps_sup = complex(eps_sup)
        self.leaky = bool(leaky)
        self.bottoms = np.concatenate([[0.0], np.cumsum([tt for _, _, tt in self.layers])])
        self.total = float(self.bottoms[-1])

    def record(self):
        return {
            "name": self.name,
            "eps_sub": cj(self.eps_sub),
            "eps_sup": cj(self.eps_sup),
            "leaky": self.leaky,
            "layers": [{"name": n, "eps": cj(e), "t_um": tt} for n, e, tt in self.layers],
        }

    @classmethod
    def from_record(cls, r):
        return cls(
            r["name"],
            cval(r["eps_sub"]),
            [(lay["name"], cval(lay["eps"]), lay["t_um"]) for lay in r["layers"]],
            cval(r["eps_sup"]),
            r["leaky"],
        )


def _layer(beta, eps, length, k0):
    c, s, cp, sp, zeta, k = layer(beta, eps, length, k0)
    return complex(c), complex(s), complex(cp), complex(sp), complex(zeta), complex(k)


def propagate(beta, st, k0):
    """Regular states u = (h, h'/eps) at the bottom of every layer and at the top, and du/dbeta."""
    beta = complex(beta)
    ks = complex(kappa(beta, st.eps_sub, k0))
    sg = -1.0 if st.leaky else 1.0
    u = np.array([1.0 + 0.0j, sg * ks / st.eps_sub], dtype=np.complex128)
    du = np.array([0.0j, sg * beta / (st.eps_sub * ks)], dtype=np.complex128)
    states, dstates = [u.copy()], [du.copy()]
    for _, eps, tt in st.layers:
        c, s, cp, sp, zeta, _ = _layer(beta, eps, tt, k0)
        M = np.array([[c, eps * s], [zeta * s / eps, c]], dtype=np.complex128)
        dM = np.array([[cp, eps * sp], [(2.0 * beta * s + zeta * sp) / eps, cp]], dtype=np.complex128)
        du = dM @ u + M @ du
        u = M @ u
        states.append(u.copy())
        dstates.append(du.copy())
    return states, dstates, ks, sg


def dispersion(beta, st, k0):
    """F = q_top + P_sup h_top and dF/dbeta (1/um), with the propagated interface states."""
    states, dstates, ks, sg = propagate(beta, st, k0)
    u, du = states[-1], dstates[-1]
    kw = complex(kappa(beta, st.eps_sup, k0))
    pw = kw / st.eps_sup
    dpw = complex(beta) / (st.eps_sup * kw)
    F = u[1] + pw * u[0]
    dF = du[1] + dpw * u[0] + pw * du[0]
    with np.errstate(all="ignore"):
        G = 0.5 * (u[0] + u[1] / pw)
    return (
        complex(F),
        complex(dF),
        {
            "states": states,
            "kappa_sub": ks,
            "branch_sign": sg,
            "kappa_sup": kw,
            "P_sup": pw,
            "h_top": complex(u[0]),
            "q_top": complex(u[1]),
            "G": complex(G),
        },
    )


def _finite(z):
    z = complex(z)
    return math.isfinite(z.real) and math.isfinite(z.imag)


def newton(beta0, st, settings, k0):
    """A-1 complex Newton on F(beta); counts one root-finder call (the points of continue_mode)."""
    CALLS["newton"] += 1
    rtol, maxit = float(settings["newton_relative_step"]), int(settings["newton_max_iterations"])
    beta = complex(beta0)
    rec = {"start_per_um": cj(beta0), "stack": st.name, "converged": False}
    rel = None
    for it in range(1, maxit + 1):
        with np.errstate(all="ignore"):
            F, dF, _ = dispersion(beta, st, k0)
        if not (_finite(F) and _finite(dF)) or dF == 0:
            rec.update(iterations=it - 1, termination="nonfinite F or zero derivative", root_per_um=None)
            return rec, None
        step = F / dF
        new = beta - step
        if not _finite(new) or new == 0:
            rec.update(iterations=it, termination="nonfinite iterate", root_per_um=None)
            return rec, None
        rel = abs(step) / abs(new)
        beta = new
        if rel <= rtol:
            F, _, _ = dispersion(beta, st, k0)
            rec.update(
                converged=True,
                iterations=it,
                termination="A-1 criterion met",
                root_per_um=cj(beta),
                final_relative_step=rel,
                abs_F_at_root_per_um=abs(F),
            )
            return rec, beta
    rec.update(
        iterations=maxit,
        termination="iteration cap without the A-1 criterion",
        root_per_um=cj(beta),
        final_relative_step=rel,
    )
    return rec, None


# ---------------- profiles and normalization ----------------


def profile(beta, st, s, k0):
    """f and df/ds (1/um) of the propagated profile at the heights s (um from the lower stack interface)."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    _, _, info = dispersion(beta, st, k0)
    states, ks, sg, kw = info["states"], info["kappa_sub"], info["branch_sign"], info["kappa_sup"]
    f = np.zeros(s.shape, dtype=np.complex128)
    fp = np.zeros(s.shape, dtype=np.complex128)
    m = s < 0.0
    e = np.exp(sg * ks * s[m])
    f[m], fp[m] = e, sg * ks * e
    for j, (_, eps, tt) in enumerate(st.layers):
        lo, hi = st.bottoms[j], st.bottoms[j + 1]
        m = (s >= lo) & ((s < hi) if j < len(st.layers) - 1 else (s <= hi))
        if np.any(m):
            c, sv, _, _, zeta, _ = layer(complex(beta), eps, s[m] - lo, k0)
            u = states[j]
            f[m] = c * u[0] + eps * sv * u[1]
            fp[m] = eps * (zeta * sv / eps * u[0] + c * u[1])
    m = s > st.total
    ut = states[-1]
    e = ut[0] * np.exp(-kw * (s[m] - st.total))
    f[m], fp[m] = e, -kw * e
    return f, fp


def eps_at(st, s):
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.full(s.shape, st.eps_sub, dtype=np.complex128)
    for j, (_, eps, _) in enumerate(st.layers):
        lo, hi = st.bottoms[j], st.bottoms[j + 1]
        out[(s >= lo) & ((s < hi) if j < len(st.layers) - 1 else (s <= hi))] = eps
    out[s > st.total] = st.eps_sup
    return out


class Mode:
    """A gap mode: stack, beta (1/um), k0 (1/um) and the profile API at the configured modal origin."""

    def __init__(self, label, st, beta, kind, k0, record=None):
        self.label, self.stack, self.beta, self.kind, self.k0 = label, st, complex(beta), kind, float(k0)
        self.record = record

    def profile_s(self, s_um):
        return profile(self.beta, self.stack, s_um, self.k0)

    def profile_z(self, z_nm, z_origin_nm):
        """f and df/dz (1/um) at C1-frame heights z (nm), s_mode = (z - z_glass_Ag)/1000."""
        return profile(
            self.beta, self.stack, (np.asarray(z_nm, dtype=np.float64) - z_origin_nm) / 1000.0, self.k0
        )


def configured_stack(cfg, label):
    """The configured stack of a gap mode as a ModeStack: glass | bottom-up layers (thickness um to 12 decimals) |
    water."""
    eps, name = cfg["eps"], {"central gap": "central", "plateau gap": "plateau"}[label]
    layers = []
    for j, (m, lo, hi) in enumerate(cfg["stacks"][name]["layers"]):
        lay = {materials.GAP: "gap", materials.CR: "Cr"}.get(m, "lower Ag" if j == 0 else "upper Ag")
        layers.append((lay, complex(eps[m]), round((hi - lo) / 1000.0, 12)))
    return ModeStack(name, complex(eps[materials.GLASS]), layers, complex(eps[materials.WATER]))


def thickness_path(seed, target):
    """The continuation stacks from a seed stack to the target when they differ in layer thicknesses only (same
    substrate and superstrate, the same bottom-up layer permittivities; the layers beyond the shorter list count as zero
    thickness on that side, so a removed Cr layer approaches zero): n = ceil(largest change / STEP_NM) points with every
    thickness linear in the point number, a layer of zero thickness omitted, the last point the target itself. None
    when the stacks differ otherwise."""
    if seed.leaky or target.leaky or seed.eps_sub != target.eps_sub or seed.eps_sup != target.eps_sup:
        return None
    a, b = seed.layers, target.layers
    m = min(len(a), len(b))
    if [e for _, e, _ in a[:m]] != [e for _, e, _ in b[:m]]:
        return None
    longer = a if len(a) > len(b) else b
    ta = [tt for _, _, tt in a] + [0.0] * (len(longer) - len(a))
    tb = [tt for _, _, tt in b] + [0.0] * (len(longer) - len(b))
    n = max(1, math.ceil(round(max(abs(y - x) for x, y in zip(ta, tb)) * 1000.0 / STEP_NM, 9)))
    pts = []
    for i in range(1, n):
        t = [x + (y - x) * i / n for x, y in zip(ta, tb)]
        lay = [(nm, e, tt) for (nm, e, _), tt in zip(longer, t) if tt > 0.0]
        pts.append(ModeStack(target.name, target.eps_sub, lay, target.eps_sup))
    return pts + [target]


def origin_vector(mode, A0, z_nm, z_origin_nm, omega_eps0, eps):
    """(E_x, E_z) in V/m of the forward mode H = A0 f exp(i beta s) at height z in a medium of permittivity eps:
    E_x = -i A0 f'(z)/(omega eps0 eps), E_z = -beta A0 f(z)/(omega eps0 eps), f' and beta converted to SI."""
    f, fp = mode.profile_z(np.array([float(z_nm)]), z_origin_nm)
    return complex(A0) * np.array([-1j * fp[0], -mode.beta * f[0]]) * 1e6 / (omega_eps0 * eps)


def device_modes(cfg):
    """Start from the packaged bound roots and continue in layer thickness when needed."""
    target_k = cfg["frequency"]["k0_per_um"]
    records = {}
    seed_files = sorted((storage.DATA / "modes").glob("*.json"))
    for label in GAP_MODES:
        target = configured_stack(cfg, label)
        candidates = []
        for file in seed_files:
            doc = storage.read_json(file)
            for item in doc["modes"]:
                if item["label"] != label:
                    continue
                previous = ModeStack.from_record(item["stack"])
                path = thickness_path(previous, target)
                if path is not None:
                    distance = sum(abs(a[2] - b[2]) for a, b in zip(previous.layers, target.layers))
                    candidates.append((abs(doc["applies_to"]["k0_per_um"] - target_k), distance, item, path))
        dk, distance, item, path = min(candidates, key=lambda candidate: candidate[:2])
        if dk == 0.0 and distance == 0.0 and ModeStack.from_record(item["stack"]).record() == target.record():
            path = []
        beta = cval(item["beta_per_um"])
        for stack in path:
            rec, beta = newton(beta, stack, NEWTON, target_k)
            if beta is None:
                raise ArithmeticError("Gap-mode Newton iteration did not converge for " + label)
        records[label] = Mode(
            label,
            target,
            beta,
            item["kind"],
            target_k,
            {"label": label, "kind": item["kind"], "stack": target.record(), "beta_per_um": cj(beta)},
        )
    return records
