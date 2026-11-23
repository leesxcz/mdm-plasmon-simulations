"""Focused glass launch and full-band oblique reference spectrum."""

import math
import numpy as np
from .materials import GLASS, WATER


def launch_design(eps, x_f, D_nm, H0, fwhm_nm, dx, N, freq):
    """Discrete design of one glass sheet: table x (nm), sigma (A/m^2), spectrum and ideal power over the full
    propagating band |k_m| < k_j."""
    k0 = freq["k0_per_m"]
    kj = k0 * math.sqrt(eps.real)
    w0 = float(fwhm_nm) / math.sqrt(2.0 * math.log(2.0))
    i = np.arange(N, dtype=np.float64)
    x = x_f + (i - 0.5 * (N - 1)) * dx
    Hf = float(H0) * np.exp(-(((x - x_f) / w0) ** 2))
    m = np.fft.fftfreq(N) * N
    km = 2.0 * math.pi * m / (N * dx * 1e-9)  # 1/m
    A = np.fft.fft(Hf) * np.exp(1j * math.pi * m * (N - 1) / N)  # sum_i H exp(-i k_m (x_i - x_f))
    band = np.abs(km) < kj
    kz = np.where(band, np.sqrt(np.abs(kj * kj - km * km)), 1.0)
    D = D_nm * 1e-9
    sh = np.where(band, A * (2.0 * kz / (1j * eps)) * np.exp(-1j * kz * D), 0.0)
    sigma = np.fft.ifft(sh * np.exp(-1j * math.pi * m * (N - 1) / N))
    P = (
        0.5
        * (dx * 1e-9 / N)
        * float(np.sum(np.abs(A[band]) ** 2 * kz[band]))
        / (freq["omega_eps0_S_per_m"] * eps.real)
    )
    return {
        "x_nm": x,
        "sigma": sigma,
        "A": A,
        "k_m": km,
        "band": band,
        "kz": kz,
        "P_ideal_W_per_m": P,
        "w0_nm": w0,
        "x_f": x_f,
        "dx_nm": dx,
        "N": N,
        "eps": eps,
        "k_j": kj,
        "D_nm": D_nm,
    }


def reflect(d, x_m):
    """Exact reflection of a table about x_m: x_R,i = 2 x_m - x_L,N-1-i, sigma_R,i = sigma_L,N-1-i."""
    r = dict(d)
    r["x_nm"] = 2.0 * x_m - d["x_nm"][::-1]
    r["sigma"] = d["sigma"][::-1].copy()
    r["x_f"] = 2.0 * x_m - d["x_f"]
    r["reflected_from_left"] = True
    return r


def smoothstep(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t <= 0.0, 0.0, np.where(t >= 1.0, 1.0, 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5))


def window(src, s):
    sa, sb, L = float(src["s_a_um"]), float(src["s_b_um"]), float(src["L_um"])
    return smoothstep((s - sa) / (-sa)) * smoothstep((sb - s) / (sb - L))


def g_in(src, s):
    """Declared real envelope at the source plane (unfiltered)."""
    return window(src, s) * np.exp(-float(src["gamma_per_um"]) * np.asarray(s, dtype=np.float64))


def gl(n):
    """Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(int(n))


def declared_spectrum(src, k_per_m, n_gl=200):
    """h_hat(k) = int_aperture h_in(x') exp(-i k (x' - x_origin)) dx' (A) of the declared input, n_gl-point
    Gauss-Legendre on each smooth piece [s_a, 0], [0, L], [L, s_b] (W has third-derivative jumps only at piece ends).
    """
    ka, qa = float(src["k_a_per_um"]), float(src["q_a_per_um"])
    c0 = float(src["H_unit_A_per_m"]) * np.exp(-1j * qa * (float(src["z_T_um"]) - float(src["z_p_um"])))
    sa, sb, L = float(src["s_a_um"]), float(src["s_b_um"]), float(src["L_um"])
    k_um = np.asarray(k_per_m, dtype=np.float64).ravel() * 1e-6
    t, w = gl(n_gl)
    out = np.zeros(k_um.size, dtype=np.complex128)
    for lo, hi in ((sa, 0.0), (0.0, L), (L, sb)):
        s = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
        f = c0 * g_in(src, s) * np.exp(1j * ka * s) * (0.5 * (hi - lo) * w)
        for s0 in range(0, k_um.size, 4096):
            out[s0 : s0 + 4096] += np.exp(-1j * k_um[s0 : s0 + 4096, None] * s[None, :]) @ f
    return out * 1e-6


BAND = "full propagating band -k_w < k_x < k_w (theta parametrization, open Gauss-Legendre panels)"


def theta_nodes(src, eps_w, k_w_m, n_panels=512, p=20, n_gl=200):
    """Composite p-point Gauss-Legendre nodes on n_panels equal panels of theta in (-pi/2, pi/2): k = k_w sin(theta),
    q = k_w cos(theta) > 0 at every node, weights w_k = k_w cos(theta) w_theta (the Jacobian dk = k_w cos(theta)
    dtheta), the declared spectrum h_hat(k) (A) of the input at the nodes."""
    ew = float(np.real(eps_w))
    kw = float(k_w_m)
    t, w = gl(p)
    e = np.linspace(-0.5 * math.pi, 0.5 * math.pi, int(n_panels) + 1)
    mid, half = 0.5 * (e[:-1] + e[1:]), 0.5 * (e[1:] - e[:-1])
    th = (mid[:, None] + half[:, None] * t[None, :]).ravel()
    wth = (half[:, None] * w[None, :]).ravel()
    cos_th = np.cos(th)
    k = kw * np.sin(th)
    q = kw * cos_th
    wk = wth * kw * cos_th
    hh = declared_spectrum(src, k, int(n_gl))
    return {
        "k": k,
        "w": wk,
        "h_hat": hh,
        "q": q,
        "theta": th,
        "w_theta": wth,
        "k_w": kw,
        "k_band": kw,
        "eps_w": ew,
        "n_panels": int(n_panels),
        "p": int(p),
        "n_gl": int(n_gl),
        "nodes": int(k.size),
        "parametrization": "theta",
        "band": BAND,
    }


class FullBandBeam:
    """Beam B over the full propagating band: H_b(x, z) = sum_m c_m exp(i k_m (x - x_0) + i q_m (z_T - z)) in
    homogeneous water, c_m = w_m h_hat_m/(2 pi), with the theta nodes above."""

    def __init__(self, src, eps_w, k_w_m, omega_eps0, n_panels=512, p=20, n_gl=200):
        self.src = src
        self.nodes = theta_nodes(src, eps_w, k_w_m, n_panels, p, n_gl)
        n = self.nodes
        self.ew = n["eps_w"]
        self.k = n["k"]
        self.c = n["w"] * n["h_hat"] / (2.0 * math.pi)
        self.q = n["q"]
        self.kw = n["k_w"]
        self.x0 = float(src["x_origin_nm"]) * 1e-9
        self.zT = float(src["z_T_um"]) * 1e-6
        self.om = omega_eps0 * self.ew

    def band_power(self):
        """(1/2 pi) int_{-k_w}^{k_w} q/(2 omega eps0 eps_w) |h_hat|^2 dk (W/m): the downward power of the beam."""
        return float(
            np.sum(self.nodes["w"] * self.q / (2.0 * self.om) * np.abs(self.nodes["h_hat"]) ** 2)
            / (2.0 * math.pi)
        )


def reference_src(cfg, H=None):
    """The reference input in the accepted source-dictionary form from the resolved configuration (carrier k_a, q_a
    derived together from energy, water and angle; window continuation from the physical window)."""
    r, w, c = cfg["reference"], cfg["window"], cfg["carrier"]
    return {
        "x_origin_nm": r["x_origin_nm"],
        "z_T_um": r["z_T_nm"] / 1000.0,
        "z_p_um": r["z_p_nm"] / 1000.0,
        "s_a_um": w["s_a_um"],
        "s_b_um": w["s_b_um"],
        "L_um": w["L_um"],
        "gamma_per_um": r["gamma_in_per_um"],
        "k_a_per_um": c["k_a_per_um"],
        "q_a_per_um": c["q_a_per_um"],
        "alpha_deg": r["alpha_deg"],
        "H_unit_A_per_m": r["H_in_A_per_m"] if H is None else float(H),
    }


def oblique_beam(cfg, quadrature="primary", H=None):
    """The configured full-band reference beam (physical amplitude H_in unless H is given) on the named quadrature."""
    freq, ew = cfg["frequency"], float(cfg["eps"][WATER].real)
    n_panels, p, n_gl = cfg["reference"]["quadrature"][quadrature]
    return FullBandBeam(
        reference_src(cfg, H),
        ew,
        freq["k0_per_m"] * math.sqrt(ew),
        freq["omega_eps0_S_per_m"],
        n_panels,
        p,
        n_gl,
    )


def launch_tables(cfg):
    launch = cfg["launch"]
    left = launch_design(
        cfg["eps"][GLASS],
        launch["focus_left_nm"][0],
        launch["D_nm"],
        launch["H0_A_per_m"],
        launch["intensity_fwhm_nm"],
        launch["dx_nm"],
        launch["N"],
        cfg["frequency"],
    )
    return {"left": left, "right": reflect(left, cfg["layout"]["xm"])}
