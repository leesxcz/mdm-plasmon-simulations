"""One-dimensional physical calibration of the equivalent X feed."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from . import c1_xy_common as cc, c1_xy_coeff as cf, profile

H_NM, N_GL_X = 5.0, 12
Y_CONST_NM = 2000.0
X_CUTS_NM = (1700.0, 2200.0)
A_BAR_P_UNIT = 1.0 + 0j


def piecewise_boundaries(breaks_nm, h_nm):
    """Element boundaries (nm): every interval between consecutive breaks divided into round(length/h) (at least 1)
    equal elements, so that every break is a boundary and every element is within a fraction of h."""
    b = np.array(sorted(set(float(v) for v in breaks_nm)), dtype=np.float64)
    pts = [b[0]]
    for lo, hi in zip(b[:-1], b[1:]):
        n = max(1, int(round((hi - lo) / float(h_nm))))
        pts.extend(np.linspace(lo, hi, n + 1)[1:].tolist())
    out = np.array(pts, dtype=np.float64)
    out[0], out[-1] = b[0], b[-1]
    return out


def assemble_1d(xb_nm, C_q, M_q, n_gl_x):
    """K_ij = int C N_i' N_j' + M N_i N_j dx on the P4 elements xb (nm) from the coefficients at the n_gl_x
    Gauss-Legendre points of every element (C_q, M_q of shape (n_el, nq)); complex128 CSR (the element matrices
    scattered by their DOF indices and summed), dimensionless."""
    xb = np.asarray(xb_nm, dtype=np.float64)
    n_el = xb.size - 1
    h = np.diff(xb)
    t, w = np.polynomial.legendre.leggauss(int(n_gl_x))
    N, dN = profile.lagrange_p4(t)  # (nq, 5)
    J = 0.5 * h  # dx/dxi (nm)
    wq = w[None, :] * J[:, None]  # (n_el, nq) nm
    dNx = dN[None, :, :] / J[:, None, None]  # (n_el, nq, 5) per nm
    Nn = np.broadcast_to(N[None, :, :], dNx.shape)
    Ke = np.einsum("eq,eqi,eqj->eij", wq * C_q, dNx, dNx) + np.einsum("eq,eqi,eqj->eij", wq * M_q, Nn, Nn)
    n = 4 * n_el + 1
    dofs = 4 * np.arange(n_el)[:, None] + np.arange(5)[None, :]
    rows = np.broadcast_to(dofs[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], Ke.shape).ravel()
    return sparse.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n), dtype=np.complex128).tocsr()


def quadrature_points(xb_nm, n_gl_x):
    xb = np.asarray(xb_nm, dtype=np.float64)
    t, _ = np.polynomial.legendre.leggauss(int(n_gl_x))
    return xb[:-1, None] + 0.5 * np.diff(xb)[:, None] * (t[None, :] + 1.0)


def central_constants(tables):
    """beta_c (per nm), A_c (nm), b0_c (per nm), C_c = 1/b0_c (nm), M_c = 1/A_c (per nm) of the central film t = 20 nm
    (the first table point, n = 1)."""
    beta_c = complex(tables["beta_per_um"][0]) * 1e-3
    A_c, b0_c = complex(tables["A0_nm"][0]), complex(tables["b0_0_per_nm"][0])
    return {
        "beta_c_per_nm": beta_c,
        "A_c_nm": A_c,
        "b0_c_per_nm": b0_c,
        "C_c_nm": 1.0 / b0_c,
        "M_c_per_nm": 1.0 / A_c,
    }


def c1_line_coefficients(x_nm, tables, y_const=Y_CONST_NM):
    """The X-taper line with Cr: C, M of c1_xy_coeff.coefficients_xy(x, y_const) (t = 20 + s(x), t_x = -0.04 on
    a < x < b, the Cr-actual A and b0 for x < a) and the 1D PML pullback C^P = C/s_x, M^P = s_x M with s_x from
    c1_xy_common.pml_stretch(x, 0) (equation (9) with s_y = 1)."""
    x = np.asarray(x_nm, dtype=np.float64)
    Cx, _, M = cf.coefficients_xy(x, y_const, tables)
    sx = cc.pml_stretch(x, 0.0)[0]
    t, tx, _, cr = cc.geometry_xy(x, y_const)
    return {
        "C_nm": Cx / sx,
        "M_per_nm": sx * M,
        "C_real_nm": Cx,
        "M_real_per_nm": M,
        "t_nm": t,
        "tx": tx,
        "cr": cr,
        "sx": sx,
    }


def amplitudes(sol, C_fn, x_nm, beta_per_nm, A_bar_nm):
    """a_+- = (1/2)[hbar_y +- Psi/(i beta A_bar)] at x (nm) with hbar_y = C Psi' (the projector of equation (7) in
    the local gauge; central film: beta_c, A_c; plateau: beta_red,p, A_bar_p)."""
    x = np.atleast_1d(np.asarray(x_nm, dtype=np.float64))
    psi, dpsi = sol(x)
    hy = np.asarray(C_fn(x)["C_nm"], dtype=np.complex128) * dpsi
    q = psi / (1j * complex(beta_per_nm) * complex(A_bar_nm))
    return 0.5 * (hy + q), 0.5 * (hy - q), psi, hy


def feed_geometry(tables, coefficients):
    return {
        "x_L_nm": cc.OUTER_X[0],
        "x_R_nm": cc.OUTER_X[1],
        "breaks_nm": (cc.REAL_X[0], cc.XS_NM, cc.A_NM, cc.B_NM, cc.REAL_X[1]),
        "coefficients": coefficients,
        "ports": {"kind": "dirichlet_feed", "x_s_nm": cc.XS_NM, "a_bar_p": A_BAR_P_UNIT},
    }


def calibrate(tables, launch, directory):
    """Solve the unit line feed once and scale its mean forward amplitude to the 2D A0."""
    coefficients = lambda x: c1_line_coefficients(x, tables)
    geometry = feed_geometry(tables, coefficients)
    h_nm, n_gl_x = H_NM, N_GL_X
    xb = piecewise_boundaries([geometry["x_L_nm"]] + list(geometry["breaks_nm"]) + [geometry["x_R_nm"]], h_nm)
    xq = quadrature_points(xb, n_gl_x)
    co = geometry["coefficients"](xq.ravel())
    C_q = np.asarray(co["C_nm"], dtype=np.complex128).reshape(xq.shape)
    M_q = np.asarray(co["M_per_nm"], dtype=np.complex128).reshape(xq.shape)
    K = assemble_1d(xb, C_q, M_q, n_gl_x)
    n = K.shape[0]
    b = np.zeros(n, dtype=np.complex128)
    ports = geometry["ports"]
    nodes = profile.Solution(xb, np.zeros(n)).nodes_um()  # positions in nm here (the class is unit-agnostic)
    k = int(np.argmin(np.abs(nodes - float(ports["x_s_nm"]))))
    b[k] = -2.0 * complex(ports["a_bar_p"])
    inner = np.arange(1, n - 1)
    KI = K[inner, :][:, inner].tocsc()
    U = np.zeros(n, dtype=np.complex128)
    U[inner] = np.asarray(spsolve(KI, b[inner]), dtype=np.complex128)
    sol = profile.Solution(xb, U)
    central = central_constants(tables)
    beta, area = central["beta_c_per_nm"], central["A_c_nm"]
    cuts = np.asarray(X_CUTS_NM)
    forward, _, _, _ = amplitudes(sol, coefficients, cuts, beta, area)
    a_unit = forward * np.exp(-1j * beta * (cuts - cc.B_NM))
    gamma = launch["A0_A_per_m"] / np.mean(a_unit)
    nodes = sol.nodes_um()
    psi, derivative = sol(nodes)
    co = coefficients(nodes)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "calib_1d.npz",
        x=nodes,
        Psi=psi,
        hbar_y=co["C_nm"] * derivative,
        xb_nm=xb,
        x_cuts_nm=cuts,
        a_unit_plus=a_unit,
        **co
    )
    return {
        "a_2D": launch["A0_A_per_m"],
        "a_unit_plus": a_unit,
        "a_unit_plus_mean": np.mean(a_unit),
        "gamma": gamma,
        "a_bar_p": gamma * A_BAR_P_UNIT,
    }
