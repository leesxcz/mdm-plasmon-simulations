"""Incident and reflected full-band fields of stratified media."""

import math

import numpy as np

from .materials import GLASS, WATER

CHUNK_GPU = 2048
CHUNK_CPU = 256
BASE_KEYS = ("H", "Hx", "Hz", "Hxx", "Hxz", "Hzz")  # H_b and its derivatives (spectral sums)
DERIVED_NEEDS = {
    "Ex": ("Hz",),
    "Ez": ("Hx",),
    "dxEx": ("Hxz",),
    "dxEz": ("Hxx",),
    "q": ("Hz",),
    "F": ("Hx", "Hz"),
}


def backend(name):
    """(xp, label): 'cupy' for the production spectral sums (requires a CUDA device), or 'numpy' when explicitly
    requested."""
    if name == "numpy":
        return np, "numpy %s (complex128)" % np.__version__
    import cupy as cp

    return cp, "cupy %s (complex128)" % cp.__version__


class Stack:
    """The background field H_b of one planar stack for the beam's spectral components: incident from the water (the
    beam), reflected, the layer fields and the downward transmitted wave in glass. spec: config.stacks entry (layers
    (material, bottom, top) in C1 nm bottom-up, z_top_nm, interfaces_nm, region_mat)."""

    def __init__(self, beam, eps_codes, spec, k0_m, omega_eps0, backend_name="cupy", chunk=None):
        self.beam = beam
        self.eps = np.asarray(eps_codes, dtype=np.complex128)
        self.k0 = float(k0_m)
        self.omega_eps0 = float(omega_eps0)
        k = beam.k
        self.k = k
        self.ew, self.eg = self.eps[WATER], self.eps[GLASS]
        self.qw = beam.q
        qg = np.sqrt(self.eg * self.k0**2 - k * k)
        self.qg = qg.real + 0.0j
        layers = tuple(tuple(lay) for lay in spec["layers"])
        self.spec = spec
        self.layer_spec = layers
        self.zb = float(layers[0][1]) * 1e-9
        self.zt = float(spec["z_top_nm"]) * 1e-9
        self.zT = beam.zT
        self.x0 = beam.x0
        self.region_mat = (GLASS,) + tuple(int(m) for m, _, _ in layers) + (WATER,)
        self.interfaces_nm = tuple(float(lo) for _, lo, _ in layers) + (float(spec["z_top_nm"]),)
        self.n_regions = len(self.region_mat)
        y = np.stack([np.ones_like(k, dtype=np.complex128), -1j * self.qg / self.eg])
        self.layers, self.states = [], [y]
        for mat, lo, hi in layers:
            e = self.eps[mat]
            q = np.sqrt(e * self.k0**2 - k * k)
            d = (hi - lo) * 1e-9
            c, s = np.cos(q * d), np.sin(q * d)
            y0, y1 = y
            y = np.stack([c * y0 + e * s / q * y1, -q * s / e * y0 + c * y1])
            self.layers.append({"mat": int(mat), "eps": e, "q": q, "lo_m": lo * 1e-9, "hi_m": hi * 1e-9})
            self.states.append(y)
        m0, m1 = self.states[-1]
        self.t = (2j * self.qw / self.ew) / (1j * self.qw * m0 / self.ew - m1)
        self.r = self.t * m0 - 1.0
        self.u = beam.c * np.exp(1j * self.qw * (self.zT - self.zt))
        self.ut, self.ur = self.u * self.t, self.u * self.r
        for j, lay in enumerate(self.layers, start=1):
            y0, y1 = self.states[j - 1]
            e, q = lay["eps"], lay["q"]
            lay["A"] = 0.5 * (
                y0 - 1j * e * y1 / q
            )  # cos(q d) y0 + (e/q) sin(q d) y1 = A e^{iqd} + B e^{-iqd}
            lay["B"] = 0.5 * (y0 + 1j * e * y1 / q)
            lay["Cq"] = (
                0.5j * q * y0 / e + 0.5 * y1
            )  # -(q/e) sin(q d) y0 + cos(q d) y1 = Cq e^{iqd} + Dq e^{-iqd}
            lay["Dq"] = -0.5j * q * y0 / e + 0.5 * y1
        self.xp, self.backend = backend(backend_name)
        self.chunk = int(chunk) if chunk else (CHUNK_GPU if self.xp is not np else CHUNK_CPU)
        xp = self.xp
        self.dev = {
            "k": xp.asarray(k),
            "ik": xp.asarray(1j * k),
            "mk2": xp.asarray(-k * k),
            "qw": xp.asarray(self.qw),
            "qg": xp.asarray(self.qg),
            "c": xp.asarray(beam.c),
            "ur": xp.asarray(self.ur),
            "ut": xp.asarray(self.ut),
        }
        for j, lay in enumerate(self.layers, start=1):
            for key in ("q", "A", "B", "Cq", "Dq"):
                self.dev["%s%d" % (key, j)] = xp.asarray(lay[key])

    # ---- regions of this stack ----

    def region_of_z(self, z_nm):
        """Background region of points strictly inside a region; an interface point goes to the region above it
        (callers decide sides explicitly where it matters)."""
        return np.searchsorted(
            np.asarray(self.interfaces_nm), np.asarray(z_nm, dtype=np.float64), side="right"
        ).astype(np.int64)

    def region_bounds(self, j, z_lo=-math.inf, z_hi=math.inf):
        edges = (z_lo,) + self.interfaces_nm + (z_hi,)
        return edges[j], edges[j + 1]

    def fields(self, x_nm, z_nm, region, keys=BASE_KEYS, device=False):
        """H_b and the requested derivatives (SI) at the points, each evaluated with the formulas of its declared
        background region (0 glass, 1..n layers, n + 1 water), with the electric quantities the computed ones give:
        E_x (from H_z), E_z (H_x), d_x E_x (H_xz), d_x E_z (H_xx), F (E_x and E_z), q (H_z); also the background eps
        and the region. A requested electric quantity brings the derivatives it needs; keys default to all six.
        device=True keeps spectral sums and results on the GPU."""
        xp = self.xp
        am = xp if device else np  # array module of the inputs, the region bookkeeping and the outputs

        def total(values):
            return values.sum(axis=1)

        need = set(keys)
        for key in keys:
            need.update(DERIVED_NEEDS.get(key, ()))
        base = [key for key in BASE_KEYS if key in need]
        need_h, need_q = any(k in base for k in ("H", "Hx", "Hxx")), any(k in base for k in ("Hz", "Hxz"))
        x = am.asarray(x_nm, dtype=am.float64).ravel() * 1e-9 - self.x0
        z = am.asarray(z_nm, dtype=am.float64).ravel() * 1e-9
        reg = am.broadcast_to(am.asarray(region, dtype=am.int64), x.shape).ravel()
        n = x.size
        out = {key: am.empty(n, dtype=am.complex128) for key in base}
        D = self.dev
        for rj in am.unique(reg).tolist():
            idx = am.flatnonzero(reg == rj)
            er = self.eps[self.region_mat[rj]]
            for s0 in range(0, idx.size, self.chunk):
                ii = idx[s0 : s0 + self.chunk]
                xs = xp.asarray(x[ii])[:, None]
                zs = xp.asarray(z[ii])[:, None]
                EX = xp.exp(1j * xs * D["k"][None, :])
                qc = q2 = None
                if rj == 0:
                    hc = D["ut"][None, :] * xp.exp(-1j * D["qg"][None, :] * (zs - self.zb))
                    if need_q:
                        qc = (-1j * D["qg"] / self.eg)[None, :] * hc
                    if "Hzz" in base:
                        q2 = (-D["qg"] ** 2)[None, :] * hc
                elif rj == len(self.layers) + 1:
                    dn = xp.exp(1j * D["qw"][None, :] * (self.zT - zs))
                    up = xp.exp(1j * D["qw"][None, :] * (zs - self.zt))
                    hc = D["c"][None, :] * dn + D["ur"][None, :] * up
                    if need_q:
                        qc = (1j * D["qw"] / self.ew)[None, :] * (
                            -D["c"][None, :] * dn + D["ur"][None, :] * up
                        )
                    if "Hzz" in base:
                        q2 = (-D["qw"] ** 2)[None, :] * hc
                    del dn, up
                else:
                    lay = self.layers[rj - 1]
                    E1 = xp.exp(1j * D["q%d" % rj][None, :] * (zs - lay["lo_m"]))
                    E2 = 1.0 / E1
                    hc = D["ut"][None, :] * (D["A%d" % rj][None, :] * E1 + D["B%d" % rj][None, :] * E2)
                    if need_q:
                        qc = D["ut"][None, :] * (D["Cq%d" % rj][None, :] * E1 + D["Dq%d" % rj][None, :] * E2)
                    if "Hzz" in base:
                        q2 = (-D["q%d" % rj] ** 2)[None, :] * hc
                    del E1, E2
                vals = {}
                if need_h:
                    Eh = EX * hc
                    if "H" in base:
                        vals["H"] = total(Eh)
                    if "Hx" in base:
                        vals["Hx"] = total(Eh * D["ik"][None, :])
                    if "Hxx" in base:
                        vals["Hxx"] = total(Eh * D["mk2"][None, :])
                    del Eh
                if "Hzz" in base:
                    vals["Hzz"] = total(EX * q2)
                if need_q:
                    Eq = EX * qc
                    if "Hz" in base:
                        vals["Hz"] = er * total(Eq)
                    if "Hxz" in base:
                        vals["Hxz"] = er * total(Eq * D["ik"][None, :])
                    del Eq
                for key, v in vals.items():
                    out[key][ii] = v if am is xp else v.get()
                del EX, hc, qc, q2, vals
        ed = am.asarray(self.eps)[am.asarray(self.region_mat)[reg]]
        den = self.omega_eps0 * ed
        if "Hz" in out:
            out["Ex"] = -1j * out["Hz"] / den
        if "Hx" in out:
            out["Ez"] = 1j * out["Hx"] / den
        if "Hxz" in out:
            out["dxEx"] = -1j * out["Hxz"] / den
        if "Hxx" in out:
            out["dxEz"] = 1j * out["Hxx"] / den
        if "Ex" in out and "Ez" in out:
            out["F"] = am.abs(out["Ex"]) ** 2 + am.abs(out["Ez"]) ** 2
        out["eps"] = ed
        if "Hz" in out:
            out["q"] = out["Hz"] / ed
        out["region"] = reg
        return out

    def fields_by_z(self, x_nm, z_nm):
        return self.fields(x_nm, z_nm, self.region_of_z(z_nm))

    def free_device(self):
        if self.xp is not np:
            self.dev = {}
            self.xp.get_default_memory_pool().free_all_blocks()
