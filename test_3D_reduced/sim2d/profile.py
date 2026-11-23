"""Vertical gap mode, thickness tangent and quadrature."""

import bisect
import math
import numpy as np
from sim2d import modes
from sim2d.materials import AG, GAP, GLASS, WATER

XI_NODES = np.linspace(-1.0, 1.0, 5)


class ModeFamily:
    """Bound vertical profile as a function of the local upper silver thickness (micrometres)."""

    def __init__(self, eps, tm_um, td_um, k0_per_um, seed_tau_um, seed_beta, max_step_um, q_q):
        self.eps_sub, self.eps_ag = complex(eps[GLASS]), complex(eps[AG])
        self.eps_gap, self.eps_sup = complex(eps[GAP]), complex(eps[WATER])
        self.tm, self.td, self.k0 = float(tm_um), float(td_um), float(k0_per_um)
        self.max_step, self.q_q = float(max_step_um), float(q_q)
        self.cache, self.order, self.newton_calls = {}, [], 0
        self.seed = {"tau_um": float(seed_tau_um), "beta_per_um": complex(seed_beta)}
        self._newton(float(seed_tau_um), complex(seed_beta))

    def stack(self, tau):
        return modes.ModeStack(
            "roof",
            self.eps_sub,
            [
                ("lower Ag", self.eps_ag, self.tm),
                ("gap", self.eps_gap, self.td),
                ("upper Ag", self.eps_ag, float(tau)),
            ],
            self.eps_sup,
        )

    def _newton(self, tau, beta0):
        st = self.stack(tau)
        rec, root = modes.newton(beta0, st, modes.NEWTON, self.k0)
        self.newton_calls += 1
        if root is None:
            raise ValueError(
                "gap-mode continuation failed at upper-Ag thickness %g nm: %s" % (1e3 * tau, rec)
            )
        _, _, info = modes.dispersion(root, st, self.k0)
        entry = {
            "tau": tau,
            "beta": complex(root),
            "stack": st,
            "info": info,
            "iterations": rec["iterations"],
            "abs_F": rec["abs_F_at_root_per_um"],
        }
        self.cache[tau] = entry
        bisect.insort(self.order, tau)
        return entry

    def root(self, tau):
        """The record of the thickness tau (um): cached, else continued from the nearest cached thickness."""
        tau = float(tau)
        hit = self.cache.get(tau)
        if hit is not None:
            return hit
        i = bisect.bisect_left(self.order, tau)
        near = min(
            (self.order[j] for j in (i - 1, i) if 0 <= j < len(self.order)), key=lambda t: abs(t - tau)
        )
        n = max(1, int(math.ceil(abs(tau - near) / self.max_step - 1e-12)))
        beta = self.cache[near]["beta"]
        for k in range(1, n + 1):
            tk = tau if k == n else near + (tau - near) * k / n
            entry = self.cache.get(tk) or self._newton(tk, beta)
            beta = entry["beta"]
        return entry

    def profile(self, tau, s_um):
        """f and df/ds (1/um) of the mode of thickness tau at the modal heights s (um)."""
        e = self.root(tau)
        return modes.profile(e["beta"], e["stack"], s_um, self.k0)

    def tangent(self, tau):
        """d/dtau at the root of thickness tau: beta_tau = -F_tau/F_beta with F_tau = (P_sup, 1) . M_tau s_b,
        M_tau = [[kappa_a^2 s, eps_a c], [kappa_a^2 c/eps_a, kappa_a^2 s]] (the upper-Ag transfer matrix at fixed
        beta), the top state s_top,tau = M_tau s_b + s_top,beta beta_tau, h_top,tau, kappa_w,tau and kappa_g,tau,
        with the states and their beta-derivatives (modes.propagate) kept for the profile derivative."""
        e = self.root(tau)
        if "tangent" in e:
            return e["tangent"]
        st, beta, k0 = e["stack"], e["beta"], self.k0
        states, dstates, ks, sg = modes.propagate(beta, st, k0)
        F, dF, info = modes.dispersion(beta, st, k0)
        eps_a = self.eps_ag
        kap = complex(modes.kappa(beta, eps_a, k0))
        c, s, _, _, _, _ = modes._layer(beta, eps_a, float(tau), k0)
        s_b = states[2]  # the state at the bottom of the upper Ag (depends on beta only)
        M_tau = np.array(
            [[kap * kap * s, eps_a * c], [kap * kap * c / eps_a, kap * kap * s]], dtype=np.complex128
        )
        Mts = M_tau @ s_b
        F_tau = info["P_sup"] * Mts[0] + Mts[1]
        beta_tau = -F_tau / dF
        s_top_tau = Mts + dstates[3] * beta_tau
        tg = {
            "beta_tau": complex(beta_tau),
            "F_tau": complex(F_tau),
            "F_beta": complex(dF),
            "abs_F": abs(F),
            "h_top_tau": complex(s_top_tau[0]),
            "s_top_tau": s_top_tau,
            "kappa_w_tau": complex(beta * beta_tau / info["kappa_sup"]),
            "kappa_g_tau": complex(beta * beta_tau / info["kappa_sub"]),
            "f_z_top_minus": complex(Mts[0]),  # (M_tau s_b)[0] = f_z at the top from below (step 1, item 5)
            "states": states,
            "dstates": dstates,
        }
        e["tangent"] = tg
        return tg

    def profile_tangent(self, tau, s_um):
        """f, df/ds and the analytic f_tau at fixed physical height (1/um per um): glass s kappa_g,tau f; every layer
        through beta only, [cp h_j + c h_j,beta + eps_j (sp q_j + s q_j,beta)] beta_tau; water
        exp(-kappa_w d) [h_top,tau - h_top kappa_w,tau d + h_top kappa_w]."""
        e = self.root(tau)
        tg = self.tangent(tau)
        st, beta, k0, info = e["stack"], e["beta"], self.k0, e["info"]
        s = np.atleast_1d(np.asarray(s_um, dtype=np.float64))
        f, fs = modes.profile(beta, st, s, k0)
        ft = np.zeros(s.shape, dtype=np.complex128)
        kg, kw, h = info["kappa_sub"], info["kappa_sup"], info["h_top"]
        m = s < 0.0
        ft[m] = s[m] * tg["kappa_g_tau"] * np.exp(kg * s[m])
        for j, (_, eps, _) in enumerate(st.layers):
            lo, hi = st.bottoms[j], st.bottoms[j + 1]
            m = (s >= lo) & ((s < hi) if j < len(st.layers) - 1 else (s <= hi))
            if np.any(m):
                c, sv, cp, sp, _, _ = modes.layer(beta, eps, s[m] - lo, k0)
                u, du = tg["states"][j], tg["dstates"][j]
                ft[m] = (cp * u[0] + c * du[0] + eps * (sp * u[1] + sv * du[1])) * tg["beta_tau"]
        m = s > st.total
        d = s[m] - st.total
        ft[m] = np.exp(-kw * d) * (tg["h_top_tau"] - h * tg["kappa_w_tau"] * d + h * kw)
        return f, fs, ft


def eta_of(fam):
    return {
        "glass": 1.0 / fam.eps_sub,
        "silver": 1.0 / fam.eps_ag,
        "gap": 1.0 / fam.eps_gap,
        "water": 1.0 / fam.eps_sup,
    }


def moments(fam, tau, n_gl):
    """Integrate bilinear profile and thickness-tangent moments, including the analytical exterior tails."""
    e0 = fam.root(tau)
    st, beta, k0 = e0["stack"], e0["beta"], fam.k0
    eta = eta_of(fam)
    kg, kw, h = e0["info"]["kappa_sub"], e0["info"]["kappa_sup"], e0["info"]["h_top"]
    top = st.total
    segs = [(0.0, fam.tm, eta["silver"]), (fam.tm, fam.tm + fam.td, eta["gap"])]
    segs.append((fam.tm + fam.td, top, eta["silver"]))
    x, w = np.polynomial.legendre.leggauss(int(n_gl))
    A = Iz = 0.0 + 0.0j
    B = G = 0.0 + 0.0j
    for lo, hi, et in segs:
        s = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        ws = 0.5 * (hi - lo) * w
        f, fs, ft = fam.profile_tangent(tau, s)
        B += et * np.sum(ws * f * ft)
        G += et * np.sum(ws * ft * ft)
        A += et * np.sum(ws * f * f)
        Iz += np.sum(ws * (et * fs * fs - k0 * k0 * f * f))
    A += eta["glass"] / (2.0 * kg)
    Iz += (eta["glass"] * kg * kg - k0 * k0) / (2.0 * kg)
    d0 = 0.0

    def tail(kappa):  # int_{d0}^inf exp(-kappa d) dd
        return np.exp(-kappa * d0) / kappa

    A += eta["water"] * h * h * tail(2.0 * kw)
    Iz += (eta["water"] * kw * kw - k0 * k0) * h * h * tail(2.0 * kw)
    out = {"A": complex(A), "I": complex(Iz), "beta": beta, "tau": tau}
    tg = fam.tangent(tau)
    kgt, kwt, ht = tg["kappa_g_tau"], tg["kappa_w_tau"], tg["h_top_tau"]
    B += eta["glass"] * kgt * (-1.0 / (2.0 * kg) ** 2)
    G += eta["glass"] * kgt * kgt * 2.0 / (2.0 * kg) ** 3
    a0, a1 = ht + h * kw, -h * kwt
    B += eta["water"] * h * (a0 / (2.0 * kw) + a1 / (2.0 * kw) ** 2)
    G += eta["water"] * (
        a0 * a0 / (2.0 * kw) + 2.0 * a0 * a1 / (2.0 * kw) ** 2 + 2.0 * a1 * a1 / (2.0 * kw) ** 3
    )
    out.update({"B": complex(B), "G": complex(G)})
    return out


def lagrange_p4(xi):
    """The five P4 Lagrange functions and their derivatives (d/dxi) at xi on [-1, 1] (equispaced nodes)."""
    xi = np.asarray(xi, dtype=np.float64)
    N = np.empty(xi.shape + (5,))
    dN = np.empty(xi.shape + (5,))
    for k in range(5):
        others = [m for m in range(5) if m != k]
        den = float(np.prod([XI_NODES[k] - XI_NODES[m] for m in others]))
        terms = [xi - XI_NODES[m] for m in others]
        N[..., k] = np.prod(terms, axis=0) / den
        d = np.zeros_like(xi)
        for j in range(4):
            d = d + np.prod([terms[i] for i in range(4) if i != j], axis=0)
        dN[..., k] = d / den
    return N, dN


class Solution:
    """The 1D P4 solution: element boundaries xb (um), nodal values U (complex), basis evaluation."""

    def __init__(self, xb_um, U):
        self.xb = np.asarray(xb_um, dtype=np.float64)
        self.h = np.diff(self.xb)
        self.U = np.asarray(U, dtype=np.complex128)
        self.n_el = self.xb.size - 1
        self.dofs = 4 * np.arange(self.n_el)[:, None] + np.arange(5)[None, :]

    def nodes_um(self):
        inner = [(self.xb[e] + 0.5 * self.h[e] * (XI_NODES[:-1] + 1.0)) for e in range(self.n_el)]
        return np.concatenate(inner + [self.xb[-1:]])

    def __call__(self, x_um):
        """u and du/dx (per um) at x (um)."""
        x = np.asarray(x_um, dtype=np.float64)
        e = np.clip(np.searchsorted(self.xb, x, side="right") - 1, 0, self.n_el - 1)
        xi = 2.0 * (x - self.xb[e]) / self.h[e] - 1.0
        N, dN = lagrange_p4(xi)
        Ue = self.U[self.dofs[e]]
        return np.sum(N * Ue, axis=-1), np.sum(dN * Ue, axis=-1) * 2.0 / self.h[e]
