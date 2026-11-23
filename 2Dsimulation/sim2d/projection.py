"""Reciprocal launch projection and matched-amplitude normalization."""

import numpy as np
from . import fem_p4, layered, mesh, modes, sources
from .materials import WATER

cj, cv = modes.cj, modes.cval


def z_stretch(z_um, lo_um, hi_um, strength, t_um):
    """Complex z of the stretched z absorbers and dZ/dz: Z = z -+ i a d^3/(3 t^2), J = 1 + i a (d/t)^2."""
    z = np.asarray(z_um, dtype=np.float64)
    Z, J = z.astype(np.complex128), np.ones(z.shape, dtype=np.complex128)
    a_, b_ = z < lo_um, z > hi_um
    dl, du = lo_um - z[a_], z[b_] - hi_um
    Z[a_] -= 1j * strength * dl**3 / (3.0 * t_um**2)
    J[a_] += 1j * strength * (dl / t_um) ** 2
    Z[b_] += 1j * strength * du**3 / (3.0 * t_um**2)
    J[b_] += 1j * strength * (du / t_um) ** 2
    return Z, J


class ModeAnalytic:
    """Independent profile of a central-stack mode: f from beta and the layer list by the continuous state (H, q =
    H_z/eps), gauge f = 1 at the stack bottom, glass branch exp(gamma (Z - z_b)), water f_t exp(-kappa_w (Z - z_t)),
    analytic reciprocal normalization N = int f^2/eps dZ including the exterior tails."""

    def __init__(self, rec, k0, zb_um):
        self.beta = cv(rec["beta_per_um"])
        self.eg, self.ew = cv(rec["stack"]["eps_sub"]), cv(rec["stack"]["eps_sup"])
        self.gamma = np.sqrt(self.beta**2 - self.eg * k0**2)
        if rec["kind"] == "leaky":
            self.gamma = -self.gamma
        self.kw = np.sqrt(self.beta**2 - self.ew * k0**2)
        self.zb, self.layers = zb_um, []
        h, q, z = 1.0 + 0.0j, self.gamma / self.eg, zb_um
        N = 1.0 / (2.0 * self.gamma * self.eg)
        for layer in rec["stack"]["layers"]:
            e, d = cv(layer["eps"]), float(layer["t_um"])
            kap = np.sqrt(self.beta**2 - e * k0**2)
            A_, B_ = (h + e * q / kap) / 2.0, (h - e * q / kap) / 2.0
            self.layers.append((z, z + d, e, kap, A_, B_))
            N += (
                A_ * A_ * np.expm1(2 * kap * d) / (2 * kap)
                + 2 * A_ * B_ * d
                - B_ * B_ * np.expm1(-2 * kap * d) / (2 * kap)
            ) / e
            h, q = (
                h * np.cosh(kap * d) + e * q * np.sinh(kap * d) / kap,
                kap * h * np.sinh(kap * d) / e + q * np.cosh(kap * d),
            )
            z += d
        self.zt, self.ht, self.qt = z, h, q
        self.N = N + h * h / (2.0 * self.kw * self.ew)

    def eval(self, z_um, Z):
        z = np.asarray(z_um, dtype=np.float64)
        f, fp, eps = (np.zeros(z.shape, dtype=np.complex128) for _ in range(3))
        a_, b_ = z < self.zb, z >= self.zt
        f[a_], eps[a_] = np.exp(self.gamma * (Z[a_] - self.zb)), self.eg
        fp[a_] = self.gamma * f[a_]
        f[b_], eps[b_] = self.ht * np.exp(-self.kw * (Z[b_] - self.zt)), self.ew
        fp[b_] = -self.kw * f[b_]
        for lo, hi, e, k, A_, B_ in self.layers:
            use = (z >= lo) & (z < hi)
            dz = Z[use] - lo
            f[use] = A_ * np.exp(k * dz) + B_ * np.exp(-k * dz)
            fp[use] = k * (A_ * np.exp(k * dz) - B_ * np.exp(-k * dz))
            eps[use] = e
        return f, fp, eps

    def vector(self, z_um, om):
        """(E_x, E_z) of the forward mode with unit coefficient at the height z (um), SI (V/m per A/m)."""
        z = np.array([float(z_um)])
        f, fp, e = self.eval(z, z.astype(np.complex128))
        return np.array([-1j * fp[0] * 1e6 / (om * e[0]), -self.beta * f[0] * 1e6 / (om * e[0])])


def central_transfer(cfg):
    """C (Ohm): (E_x, E_z) at the observation origin of the central stack's incident-plus-reflected response to the
    configured reference with a unit incident magnetic amplitude (no FEM)."""
    L, freq = cfg["layout"], cfg["frequency"]
    beam = sources.oblique_beam(cfg, H=1.0)
    stack = layered.Stack(
        beam,
        cfg["eps"],
        cfg["stacks"]["central"],
        freq["k0_per_m"],
        freq["omega_eps0_S_per_m"],
        backend_name="numpy",
    )
    F = stack.fields(np.array([L["xte_l"]]), np.array([L["z_em"]]), stack.n_regions - 1)
    return np.array([F["Ex"][0], F["Ez"][0]])


def from_launch(cfg, tab, solution, gap):
    """Project the left launch at the first forward cut and set the matched top-reference amplitude."""
    layout, freq = cfg["layout"], cfg["frequency"]
    om = freq["omega_eps0_S_per_m"]
    cut = layout["cuts_f_l"][0]
    rule = mesh.line_rule_p4(tab.P, tab.T, 0, cut, layout["Z0"], layout["Z1"], +1, 12)
    gpu = fem_p4.GPUTri4(tab, cfg["eps"][tab.material], "launch projection", assembly=False)
    try:
        gpu.set_solution(solution)
        samples = gpu.points(rule["el"], rule["l0"], rule["l1"], keys=("H", "Hx"))
        values = fem_p4.p4_vals(samples)
    finally:
        gpu.close()
    z = rule["z"] * 1e-3
    pml = cfg["pml"]
    Z, J = z_stretch(
        z, layout["zmin"] * 1e-3, layout["zmax"] * 1e-3, pml["strength"], pml["thickness_nm"] * 1e-3
    )
    profile = ModeAnalytic(gap.record, freq["k0_per_um"], layout["zg"] * 1e-3)
    f, _, eps = profile.eval(z, Z)
    E_back = J * gap.beta * f * 1e6 / (om * eps)
    E_field = J * 1j * values["Hx"] / (om * eps)
    a_plus = (
        om
        * np.sum(rule["w"] * 1e-3 * (E_back * values["H"] - E_field * f))
        / (2 * gap.beta * 1e6 * profile.N)
    )
    A0 = a_plus * np.exp(-1j * gap.beta * (cut - layout["xte_l"]) * 1e-3)
    u = modes.origin_vector(gap, A0, layout["z_em"], layout["zg"], om, cfg["eps"][WATER])
    P = float(np.vdot(u, u).real)
    C = central_transfer(cfg)
    H_in = float(np.sqrt(P) / np.linalg.norm(C))
    return {
        "A0_A_per_m": cj(A0),
        "u_SPP_origin_V_per_m": {"x": cj(u[0]), "z": cj(u[1])},
        "P_V2_per_m2": P,
        "H_in_A_per_m": H_in,
        "C_Ohm": {"x": cj(C[0]), "z": cj(C[1])},
        "chi_rad": float(np.angle(np.vdot(H_in * C, u))),
    }
