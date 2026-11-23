"""Thickness-dependent coefficients of the local XY equation."""

import numpy as np
from scipy.interpolate import CubicSpline
from sim2d.materials import CR, WATER
from . import c1_xy_common as cc, profile

T_MIN_NM, T_MAX_NM = cc.TM_NM, cc.TM_NM + cc.PLATEAU_NM
DT_NM = 0.05
N_GL_Z, N_GL_T = 64, 5

SPLINE_KEYS = ("beta_per_um", "A0_nm", "b0_0_per_nm", "F0", "G0_per_nm", "Gc_per_nm", "n")


def tau_um(t_nm):
    """The ModeFamily thickness key (um) of t (nm): one conversion everywhere, so cached roots are found again."""
    return float(t_nm) * 1e-3


def moments_nm(fam, t_nm):
    """Vertical moments in nm units."""
    m = profile.moments(fam, tau_um(t_nm), N_GL_Z)
    return {
        "beta_per_um": complex(m["beta"]),
        "A0_nm": complex(m["A"]) * 1e3,
        "b0_0_per_nm": complex(m["I"]) * 1e-3,
        "F0": complex(m["B"]),
        "G0_per_nm": complex(m["G"]) * 1e-3,
    }


def _stored(fam, t_nm, store):
    t = float(t_nm)
    m = store.get(t)
    if m is None:
        m = moments_nm(fam, t)
        store[t] = m
    return m


def log_n_intervals(fam, edges_nm, store):
    """int F0/A0 dtau (dimensionless; F0/A0 in 1/nm, dtau in nm) over every interval of the edges (nm): N_GL_T-point
    Gauss-Legendre per interval with the integrand from the direct moments at the nodes (store: t_nm -> moments).
    """
    x, w = np.polynomial.legendre.leggauss(N_GL_T)
    edges = np.asarray(edges_nm, dtype=np.float64)
    out = np.empty(edges.size - 1, dtype=np.complex128)
    for k in range(edges.size - 1):
        lo, hi = float(edges[k]), float(edges[k + 1])
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        acc = 0.0 + 0.0j
        for xq, wq in zip(x, w):
            m = _stored(fam, mid + half * xq, store)
            acc += wq * m["F0"] / m["A0_nm"]
        out[k] = half * acc
    return out


def build_tables(fam, dt_nm=DT_NM):
    """Integrate the uncapped profiles and apply the Cr material correction on the plateau."""
    n_int = int(round((T_MAX_NM - T_MIN_NM) / dt_nm))
    t = np.round(T_MIN_NM + dt_nm * np.arange(n_int + 1), 12)
    t[0], t[-1] = T_MIN_NM, T_MAX_NM
    store = {}
    rows = [_stored(fam, tk, store) for tk in t]  # ascending: the continuation walks up the thickness once
    tables = {"t_nm": t}
    for k in ("beta_per_um", "A0_nm", "b0_0_per_nm", "F0", "G0_per_nm"):
        tables[k] = np.array([r[k] for r in rows], dtype=np.complex128)
    tables["Gc_per_nm"] = tables["G0_per_nm"] - tables["F0"] ** 2 / tables["A0_nm"]
    seg = log_n_intervals(fam, t, store)
    tables["n"] = np.exp(np.concatenate([[0.0 + 0.0j], np.cumsum(seg)]))
    # the Cr plateau: t = 40 nm, Cr over 20 < z < 30 nm (d = z - 20 nm above the uncapped top), water-tail profile
    eps, prov = cc.permittivities()
    e40 = fam.root(tau_um(T_MAX_NM))
    kw = complex(e40["info"]["kappa_sup"]) * 1e-3  # 1/nm
    h = complex(e40["info"]["h_top"])
    eta_cr, eta_w = 1.0 / complex(eps[CR]), 1.0 / complex(eps[WATER])
    J = (1.0 - np.exp(-2.0 * kw * cc.CR_NM)) / (2.0 * kw)  # nm
    A_cr = tables["A0_nm"][-1] + (eta_cr - eta_w) * h * h * J
    b0_cr = tables["b0_0_per_nm"][-1] + (eta_cr - eta_w) * h * h * kw * kw * J
    beta_red = complex(np.sqrt(-b0_cr / A_cr))
    if beta_red.real < 0.0:
        beta_red = -beta_red
    tables.update(
        {"A_cr_nm": complex(A_cr), "b0_cr_per_nm": complex(b0_cr), "beta_red_plateau_per_nm": beta_red}
    )
    return tables


def _splines(tables):
    spl = tables.get("_spl")
    if spl is None:
        t = np.asarray(tables["t_nm"], dtype=np.float64)
        spl = {}
        for k in SPLINE_KEYS:
            v = np.asarray(tables[k], dtype=np.complex128)
            spl[k] = (CubicSpline(t, v.real), CubicSpline(t, v.imag))
        tables["_spl"] = spl
    return spl


def interp(tables, key, t_nm):
    """The table entry key at the thicknesses t (nm, within [20, 40]) by the not-a-knot cubic spline (complex)."""
    t = np.asarray(t_nm, dtype=np.float64)
    re, im = _splines(tables)[key]
    return re(t) + 1j * im(t)


def n_of_t(t_nm, tables):
    """n(t) (complex, dimensionless) at t (nm): a complex scalar for a scalar t, else an array of t's shape."""
    v = interp(tables, "n", t_nm)
    return complex(v) if np.ndim(v) == 0 else v


def coefficients_xy(x_nm, y_nm, tables):
    """Cx (nm), Cy (nm), M (1/nm) of equation (2) at the lateral points (complex128, the shape of x): (t, tx, ty, cr)
    from c1_xy_common.geometry_xy, the Cr-actual A and b0 where cr (t = 40 nm, tx = ty = 0 there), the interpolated
    uncapped tables elsewhere."""
    t, tx, ty, cr = cc.geometry_xy(x_nm, y_nm)
    n = interp(tables, "n", t)
    A = np.where(cr, tables["A_cr_nm"], interp(tables, "A0_nm", t))
    b0 = np.where(cr, tables["b0_cr_per_nm"], interp(tables, "b0_0_per_nm", t))
    Gc = interp(tables, "Gc_per_nm", t)
    n2 = n * n
    return n2 / (b0 + Gc * tx * tx), n2 / (b0 + Gc * ty * ty), n2 / A


def pml_coefficients(x_nm, y_nm, tables):
    """The PML-transformed coefficients of equation (9): CxP = sy/sx Cx (nm), CyP = sx/sy Cy (nm), MP = sx sy M (1/nm)
    with s_x, s_y = c1_xy_common.pml_stretch (1 inside the real domain)."""
    Cx, Cy, M = coefficients_xy(x_nm, y_nm, tables)
    sx, sy = cc.pml_stretch(x_nm, y_nm)
    return sy / sx * Cx, sx / sy * Cy, sx * sy * M


def profile_normalized(fam, t_nm, z_nm, tables):
    """fbar = f/n, fbar_z = f_z/n (1/nm), fbar_t = (f_t - (F0/A0) f)/n (1/nm) of the uncapped profile of thickness t
    (nm) at the absolute heights z (nm; modal coordinate s_um = (z - ZG_NM)/1000): one scalar t with an array z (the
    shape of z), or arrays t and z of equal shape (elementwise, grouped by distinct t). The profile continues through
    the Cr interval and the metals as the uncapped trial profile; material masking belongs to the caller."""
    t, z = np.broadcast_arrays(np.asarray(t_nm, dtype=np.float64), np.asarray(z_nm, dtype=np.float64))
    shape = z.shape
    t, z = t.ravel(), z.ravel()
    fb = np.empty(z.shape, dtype=np.complex128)
    fbz = np.empty_like(fb)
    fbt = np.empty_like(fb)
    uniq, inv = np.unique(t, return_inverse=True)
    for i, tu in enumerate(uniq):
        sel = inv == i
        f, fs, ft = fam.profile_tangent(tau_um(tu), (z[sel] - cc.ZG_NM) * 1e-3)
        n = complex(interp(tables, "n", tu))
        r = complex(interp(tables, "F0", tu)) / complex(interp(tables, "A0_nm", tu))  # F0/A0, 1/nm
        fb[sel] = f / n
        fbz[sel] = fs * 1e-3 / n
        fbt[sel] = (ft * 1e-3 - r * f) / n
    return fb.reshape(shape), fbz.reshape(shape), fbt.reshape(shape)
