"""Fourth-order triangular finite elements for the MDM optical fields."""

import contextlib
import itertools
import numpy as np
import scipy.sparse as sp

EXPS = tuple((a, b) for a in range(5) for b in range(5 - a))  # degree-4 monomials xi^a eta^b
LATTICE = np.array([(i / 4.0, j / 4.0) for i in range(5) for j in range(5 - i)], dtype=np.float64)
PERMS = tuple(itertools.permutations(range(3)))  # orientation classes (vertex-number order)
EDGE_DOFS = {
    (0, 1): (9, 10, 11),
    (1, 2): (6, 7, 8),
    (0, 2): (3, 4, 5),
}  # local vertices -> local edge DOF block
RULES = {"stiffness": ("TRIG", 6), "mass": ("TRIG", 8), "sheet": ("SEGM", 8)}
CHUNK_ELEMENTS = 8192
CHUNK_POINTS = 65536
POINT_KEYS = (
    "H",
    "Hx",
    "Hz",
    "Hxx",
    "Hxz",
    "Hzz",
)  # H, gradient and Hessian components of the point evaluator


def monomials(xp, xi, eta, order=2):
    """The 15 monomials xi^a eta^b (EXPS order) at the points with their exact first derivatives (d/dxi, d/deta) and
    second derivatives (d2/dxi2, d2/dxi deta, d2/deta2): arrays (..., 15), (..., 15, 2), (..., 15, 3); derivatives
    above order are not formed (None)."""
    xi = xp.asarray(xi, dtype=xp.float64)
    eta = xp.asarray(eta, dtype=xp.float64)
    one, zero = xp.ones_like(xi), xp.zeros_like(xi)
    px, py = [one], [one]
    for _ in range(4):
        px.append(px[-1] * xi)
        py.append(py[-1] * eta)

    def p(pw, k):
        return pw[k] if k >= 0 else zero

    m0 = xp.stack([p(px, a) * p(py, b) for a, b in EXPS], axis=-1)
    if order < 1:
        return m0, None, None
    m1 = xp.stack(
        [xp.stack([a * p(px, a - 1) * p(py, b), b * p(px, a) * p(py, b - 1)], axis=-1) for a, b in EXPS],
        axis=-2,
    )
    if order < 2:
        return m0, m1, None
    m2 = xp.stack(
        [
            xp.stack(
                [
                    a * (a - 1) * p(px, a - 2) * p(py, b),
                    a * b * p(px, a - 1) * p(py, b - 1),
                    b * (b - 1) * p(px, a) * p(py, b - 2),
                ],
                axis=-1,
            )
            for a, b in EXPS
        ],
        axis=-2,
    )
    return m0, m1, m2


# The two fixed 15-term sums of the GPU evaluation, compiled once per process on first use with --fmad=false (every
# product and every sum rounded on its own, as by the separate array operations they replace). p4_basis: per point and
# shape function i, the monomials of monomials() (powers by repeated multiplication, integer factors first) and
# sum_k monomial_k x C[class, k, i] over k = 0..14 in that order, for the values and the first and second reference
# derivatives as requested. p4_accumulate: per point and coefficient column j, sum_i f_i x U[D[element, i], j] over the
# local DOFs i = 0..14 in that order (complex values as re/im pairs; a real f times a complex U rounds each part once).
_KERNEL_SOURCE = r"""
#define P(p, e) ((e) >= 0 ? (p)[(e)] : 0.0)
__constant__ int EA[15] = {@EA@};
__constant__ int EB[15] = {@EB@};

extern "C" __global__ void p4_basis(const double* xi, const double* eta, const long long* E, const long long* cls,
                                    const double* C, const long long m, const int want0, const int order,
                                    double* f0, double* d1, double* d2)
{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= 15 * m) return;
    const long long n = t / 15;
    const double* c = C + 225 * cls[E[n]] + (t - 15 * n);
    double px[5], py[5];
    px[0] = 1.0;
    py[0] = 1.0;
    for (int q = 1; q < 5; ++q) {
        px[q] = px[q - 1] * xi[n];
        py[q] = py[q - 1] * eta[n];
    }
    if (want0) {
        double s = (px[EA[0]] * py[EB[0]]) * c[0];
        for (int k = 1; k < 15; ++k) s = s + (px[EA[k]] * py[EB[k]]) * c[15 * k];
        f0[t] = s;
    }
    for (int d = 0; d < 2 && order >= 1; ++d) {
        double s = 0.0;
        for (int k = 0; k < 15; ++k) {
            const int a = EA[k], b = EB[k];
            const double v = d == 0 ? ((double)a * P(px, a - 1)) * py[b] : ((double)b * px[a]) * P(py, b - 1);
            s = k == 0 ? v * c[0] : s + v * c[15 * k];
        }
        d1[2 * t + d] = s;
    }
    for (int d = 0; d < 3 && order >= 2; ++d) {
        double s = 0.0;
        for (int k = 0; k < 15; ++k) {
            const int a = EA[k], b = EB[k];
            const double v = d == 0 ? ((double)(a * (a - 1)) * P(px, a - 2)) * py[b]
                           : d == 1 ? ((double)(a * b) * P(px, a - 1)) * P(py, b - 1)
                                    : ((double)(b * (b - 1)) * px[a]) * P(py, b - 2);
            s = k == 0 ? v * c[0] : s + v * c[15 * k];
        }
        d2[3 * t + d] = s;
    }
}

extern "C" __global__ void p4_accumulate(const double* f, const long long* E, const long long* D, const double* U,
                                         const long long m, const long long k, double* out)
{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= m * k) return;
    const long long n = t / k, j = t - k * n;
    const long long* dof = D + 15 * E[n];
    const double* fn = f + 15 * n;
    const double* u = U + 2 * (k * dof[0] + j);
    double re = fn[0] * u[0], im = fn[0] * u[1];
    for (int i = 1; i < 15; ++i) {
        u = U + 2 * (k * dof[i] + j);
        re = re + fn[i] * u[0];
        im = im + fn[i] * u[1];
    }
    out[2 * t] = re;
    out[2 * t + 1] = im;
}
"""
_KERNELS = {}


def _kernels(cp):
    """p4_basis and p4_accumulate (compiled on first use, then reused by the process)."""
    if not _KERNELS:
        src = _KERNEL_SOURCE.replace("@EA@", ", ".join(str(a) for a, _ in EXPS))
        mod = cp.RawModule(
            code=src.replace("@EB@", ", ".join(str(b) for _, b in EXPS)), options=("--fmad=false",)
        )
        _KERNELS.update(basis=mod.get_function("p4_basis"), accumulate=mod.get_function("p4_accumulate"))
    return _KERNELS["basis"], _KERNELS["accumulate"]


def _launch(kernel, threads, args, block=256):
    if threads:
        kernel(((threads + block - 1) // block,), (block,), args)


class P4Setup:
    """NGSolve basis tables and edge quadrature for affine P4 triangles."""

    def __init__(self, mesh, fes):
        import ngsolve as ng

        self.ngsolve_version = ng.__version__
        self.nv, self.ne, self.ndof = int(mesh.nv), int(mesh.ne), int(fes.ndof)
        self.P = np.array([v.point for v in mesh.vertices], dtype=np.float64)[:, :2]
        T = np.empty((self.ne, 3), dtype=np.int64)
        D = np.empty((self.ne, 15), dtype=np.int64)
        mats = []
        for i, el in enumerate(mesh.Elements(ng.VOL)):
            T[i] = [v.nr for v in el.vertices]
            d = list(fes.GetDofNrs(el))
            D[i] = d
            mats.append(el.mat)
        self.T, self.D, self.mat_names = T, D, np.array(mats)
        index = {p: k for k, p in enumerate(PERMS)}
        self.cls = np.array([index[tuple(r)] for r in np.argsort(T, axis=1, kind="stable")], dtype=np.int64)
        V0, _, _ = monomials(np, LATTICE[:, 0], LATTICE[:, 1])
        self.C = np.zeros((6, 15, 15), dtype=np.float64)
        self.representatives = {}
        for c in np.unique(self.cls):
            e = int(np.flatnonzero(self.cls == c)[0])
            fe = fes.GetFE(ng.ElementId(ng.VOL, e))
            S = np.array([np.array(fe.CalcShape(x, y)) for x, y in LATTICE])
            self.C[c] = np.linalg.solve(V0, S)
            self.representatives[int(c)] = e
        self.classes_present = [int(c) for c in np.unique(self.cls)]
        # rules and reference tables at their points (CPU reference tables)
        self.rules = {}
        for name, (et, order) in RULES.items():
            ir = ng.IntegrationRule(getattr(ng, et), order)
            pts = np.array([p[:2] if et == "TRIG" else p[:1] for p in ir.points], dtype=np.float64)
            self.rules[name] = {
                "element": et,
                "order": order,
                "points": pts,
                "weights": np.array(list(ir.weights), dtype=np.float64),
            }
        xs = self.rules["stiffness"]["points"]
        xm = self.rules["mass"]["points"]
        self.dphi_s = np.einsum(
            "qmd,cmi->cqid", monomials(np, xs[:, 0], xs[:, 1])[1], self.C
        )  # (6, 12, 15, 2)
        self.phi_m = np.einsum("qm,cmi->cqi", monomials(np, xm[:, 0], xm[:, 1])[0], self.C)  # (6, 25, 15)

    def sheet_edges(self, mesh, fes, bc_name, side):
        """The boundary segments named bc_name (an internal source line): vertices, the adjacent element on the given
        side (-1 below, +1 above, by the element centroid's z), local vertex indices and the 5 local DOFs of the edge
        trace, the SEGM order-8 points as reference coordinates of that element, the weights times the edge length and
        the physical points."""
        import ngsolve as ng

        segs = []
        for el in mesh.Elements(ng.BND):
            if el.mat == bc_name:
                segs.append([v.nr for v in el.vertices])
        segs = np.array(segs, dtype=np.int64)
        flat = self.T.ravel()
        order = np.argsort(flat, kind="stable")
        ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=self.nv))])
        v2t = order // 3
        cen = self.P[self.T].mean(axis=1)
        t, w = self.rules["sheet"]["points"][:, 0], self.rules["sheet"]["weights"]
        es, loc5, ref, wl, xq, zq = [], [], [], [], [], []
        for va, vb in segs:
            ea = set(v2t[ptr[va] : ptr[va + 1]].tolist()) & set(v2t[ptr[vb] : ptr[vb + 1]].tolist())
            zs = 0.5 * (self.P[va, 1] + self.P[vb, 1])
            cand = [e for e in ea if (cen[e, 1] - zs) * side > 0]
            e = cand[0]
            la, lb = int(np.flatnonzero(self.T[e] == va)[0]), int(np.flatnonzero(self.T[e] == vb)[0])
            blk = EDGE_DOFS[(min(la, lb), max(la, lb))]
            l5 = [la, lb] + list(blk)
            lam = np.zeros((t.size, 3))
            lam[:, la], lam[:, lb] = t, 1.0 - t
            L = float(np.hypot(*(self.P[va] - self.P[vb])))
            X = np.outer(t, self.P[va]) + np.outer(1.0 - t, self.P[vb])
            es.append(e)
            loc5.append(l5)
            ref.append(lam[:, :2])
            wl.append(w * L)
            xq.append(X[:, 0])
            zq.append(X[:, 1])
        return {
            "segments": segs,
            "elements": np.array(es, dtype=np.int64),
            "local5": np.array(loc5, dtype=np.int64),
            "ref": np.array(ref),
            "weights_length": np.array(wl),
            "x_q": np.array(xq),
            "z_q": np.array(zq),
            "t": np.asarray(t, dtype=np.float64),
            "side": int(side),
            "length": float(np.sum(np.array(wl)) / np.sum(w)),
        }


def _deterministic_sum(cp, keys, values):
    """Gather sparse contributions in reproducible key order."""
    order = cp.lexsort(cp.stack((cp.arange(values.size),) + tuple(keys)))
    sk = [k[order] for k in keys]
    sv = values[order]
    begins = cp.empty(sv.size, dtype=cp.bool_)
    begins[0] = True
    diff = cp.zeros(sv.size - 1, dtype=cp.bool_)
    for k in sk:
        diff |= k[1:] != k[:-1]
    begins[1:] = diff
    starts = cp.nonzero(begins)[0]
    reduced = cp.empty(starts.size, dtype=cp.complex128)
    cp.ElementwiseKernel(
        "raw int64 starts, raw complex128 values, int64 count, int64 length",
        "complex128 result",
        "long long end = i+1 < count ? starts[i+1] : length; "
        "result = complex<double>(0.0, 0.0); "
        "for (long long k=starts[i]; k<end; ++k) result += values[k];",
        "mdm_deterministic_entry_sum",
    )(starts, sv, starts.size, sv.size, reduced)
    return [k[starts] for k in sk], reduced


class GPUTri4:
    """P4 element integration, source loads, and native field derivatives on the GPU."""

    def __init__(
        self,
        setup,
        eps_elem,
        label,
        assembly=True,
        columns=1,
        chunk_elements=CHUNK_ELEMENTS,
        chunk_points=CHUNK_POINTS,
    ):
        import cupy as cp

        self.cp, self.setup, self.label = cp, setup, label
        self.N, self.ne = int(setup.ndof), int(setup.ne)
        self.chunk_elements, self.chunk_points = int(chunk_elements), int(chunk_points)
        self.scratch_pool = cp.cuda.MemoryPool()
        self.res_pool = cp.cuda.MemoryPool()
        self._res = {}
        self.K = self.b = self.U = None
        with self._resident():
            self._put("P", setup.P)
            self._put("T", setup.T)
            self._put("D", setup.D)
            self._put("cls", setup.cls)
            self._put("eps", np.asarray(eps_elem, dtype=np.complex128))
            self._put("C", setup.C)
            self._put("dphi_s", setup.dphi_s)
            self._put("phi_m", setup.phi_m)
            for name in ("stiffness", "mass"):
                self._put("x_" + name, setup.rules[name]["points"])
                self._put("w_" + name, setup.rules[name]["weights"])

    def _resident(self):
        return self.cp.cuda.using_allocator(self.res_pool.malloc)

    def _h2d(self, values):
        return self.cp.asarray(np.ascontiguousarray(values))

    def _d2h(self, values):
        return self.cp.asnumpy(values)

    @contextlib.contextmanager
    def _stage(self):
        try:
            with self.cp.cuda.using_allocator(self.scratch_pool.malloc):
                yield
        finally:
            self.scratch_pool.free_all_blocks()

    def _put(self, name, host):
        dev = self._h2d(host)
        self._res[name] = dev
        setattr(self, name, dev)

    def _geometry(self, E):
        """Vertex coordinates and the affine map of the elements E: (V2, J, det, J^-1) with J = [V0 - V2, V1 - V2]."""
        V = self.P[self.T[E]]  # (m, 3, 2)
        x2, z2 = V[:, 2, 0], V[:, 2, 1]
        J00, J01 = V[:, 0, 0] - x2, V[:, 1, 0] - x2
        J10, J11 = V[:, 0, 1] - z2, V[:, 1, 1] - z2
        det = J00 * J11 - J01 * J10
        return (x2, z2), (J00, J01, J10, J11), det, (J11 / det, -J01 / det, -J10 / det, J00 / det)

    @staticmethod
    def _stretch(cp, v, lo, hi, t, s):
        """route_a_forms stretch: 1 + i s q^2, q = (v - hi)/t above hi, (lo - v)/t below lo, 0 inside."""
        q = cp.where(v - hi > 0, (v - hi) / t, cp.where(lo - v > 0, (lo - v) / t, 0.0))
        return 1 + 1j * s * q * q

    # ---------------- assembly ----------------

    def assemble(self, k0, pml):
        """K on the GPU (resident canonical CSR, cupyx complex128/int32). k0 in 1/nm; pml = {"thickness_nm",
        "strength", "physical_x_nm", "physical_z_nm"} (the shared record's PML, route_a_forms)."""
        with self._stage():
            self.K = self._assemble(float(k0), pml)
        return self.K

    def _assemble(self, k0, pml):
        cp = self.cp
        import cupyx.scipy.sparse as csp

        t, s = float(pml["thickness_nm"]), float(pml["strength"])
        xw, zw = [float(v) for v in pml["physical_x_nm"]], [float(v) for v in pml["physical_z_nm"]]
        k2 = k0 * k0
        vals = cp.empty((self.ne, 15, 15), dtype=cp.complex128)
        for c0 in range(0, self.ne, self.chunk_elements):
            E = cp.arange(c0, min(self.ne, c0 + self.chunk_elements))
            (x2, z2), (J00, J01, J10, J11), det, (I00, I01, I10, I11) = self._geometry(E)
            adet = cp.abs(det)
            ie = 1.0 / self.eps[E]
            cl = self.cls[E]
            Ke = cp.zeros((E.size, 15, 15), dtype=cp.complex128)
            xs, ws = self.x_stiffness, self.w_stiffness
            for q in range(int(ws.size)):
                X = x2 + J00 * xs[q, 0] + J01 * xs[q, 1]
                Z = z2 + J10 * xs[q, 0] + J11 * xs[q, 1]
                sx, sz = self._stretch(cp, X, xw[0], xw[1], t, s), self._stretch(cp, Z, zw[0], zw[1], t, s)
                d = self.dphi_s[cl, q]  # (m, 15, 2) reference gradients
                gx = I00[:, None] * d[:, :, 0] + I10[:, None] * d[:, :, 1]
                gz = I01[:, None] * d[:, :, 0] + I11[:, None] * d[:, :, 1]
                c1 = (ie * (sz / sx) * ws[q] * adet)[:, None, None]
                c2 = (ie * (sx / sz) * ws[q] * adet)[:, None, None]
                Ke += c1 * (gx[:, :, None] * gx[:, None, :]) + c2 * (gz[:, :, None] * gz[:, None, :])
            xm, wm = self.x_mass, self.w_mass
            for q in range(int(wm.size)):
                X = x2 + J00 * xm[q, 0] + J01 * xm[q, 1]
                Z = z2 + J10 * xm[q, 0] + J11 * xm[q, 1]
                sx, sz = self._stretch(cp, X, xw[0], xw[1], t, s), self._stretch(cp, Z, zw[0], zw[1], t, s)
                ph = self.phi_m[cl, q]  # (m, 15)
                c3 = (k2 * (sx * sz) * wm[q] * adet)[:, None, None]
                Ke -= c3 * (ph[:, :, None] * ph[:, None, :])
            vals[c0 : c0 + int(E.size)] = Ke
            del Ke
        rows = cp.broadcast_to(self.D[:, :, None], vals.shape).ravel()  # test function i
        cols = cp.broadcast_to(self.D[:, None, :], vals.shape).ravel()  # trial function j
        v = vals.ravel()
        del vals
        (uc, ur), red = _deterministic_sum(cp, (cols, rows), v)
        del rows, cols, v
        coo = csp.coo_matrix((red, (ur.astype(cp.int32), uc.astype(cp.int32))), shape=(self.N, self.N))
        coo.has_canonical_format = True
        A = coo.tocsr()
        del coo, ur, uc, red
        with self._resident():
            K = csp.csr_matrix((A.data.copy(), A.indices.copy(), A.indptr.copy()), shape=A.shape)
        return K

    def sheet_load_table(self, edges, table_x_nm, table_sigma, window_nm, factor):
        """The same sheet load with sigma from a source table (the launch tables, sigma_SI in A/m^2 times `factor`,
        1e-9 for A/m per nm): on the GPU the physical rule points x_q = t P_a + (1 - t) P_b of every sheet edge and the
        linear interpolation of the uniform table, zero outside the physical window (as sources.sheet_density).
        Returns the host load; the device load stays resident."""
        tx = np.asarray(table_x_nm, dtype=np.float64)
        h = float(tx[1] - tx[0])
        tab = np.asarray(table_sigma, dtype=np.complex128) * float(factor)
        with self._stage():
            self.b = self._sheet_table(
                edges, float(tx[0]), h, tab, (float(window_nm[0]), float(window_nm[1]))
            )
            out = self._d2h(self.b)
        return out

    def _sheet_table(self, edges, x0, h, tab, window):
        cp = self.cp
        seg = self._h2d(edges["segments"])
        t = self._h2d(edges["t"])
        xq = self.P[seg[:, 0], 0][:, None] * t[None, :] + self.P[seg[:, 1], 0][:, None] * (1.0 - t)[None, :]
        td = self._h2d(tab)
        pos = (xq - x0) / h
        i = cp.clip(cp.floor(pos).astype(cp.int64), 0, int(td.size) - 2)
        f = pos - i
        sig = td[i] * (1.0 - f) + td[i + 1] * f
        sig = cp.where((xq >= window[0]) & (xq <= window[1]), sig, 0.0 + 0.0j)
        return self._sheet(edges, sig)

    def _sheet(self, edges, sig):
        cp = self.cp
        es = self._h2d(edges["elements"])
        l5 = self._h2d(edges["local5"])
        ref = self._h2d(edges["ref"])  # (k, 5, 2)
        wl = self._h2d(edges["weights_length"])
        sd = sig if isinstance(sig, cp.ndarray) else self._h2d(sig)
        k, nq = int(es.size), int(wl.shape[1])
        phi, _, _ = self._basis(cp.repeat(es, nq), ref[:, :, 0].ravel(), ref[:, :, 1].ravel(), True, 0)
        phi = phi.reshape(k, nq, 15)
        sel = cp.take_along_axis(phi, cp.broadcast_to(l5[:, None, :], (k, nq, 5)), axis=2)  # (k, nq, 5)
        contrib = (wl * sd)[:, :, None] * sel
        tot = contrib[:, 0, :]
        for q in range(1, nq):
            tot = tot + contrib[:, q, :]
        rows = cp.take_along_axis(self.D[es], l5, axis=1).ravel()
        (ur,), red = _deterministic_sum(cp, (rows,), tot.ravel())
        self.b = None
        with self._resident():
            b = cp.zeros(self.N, dtype=cp.complex128)
        b[ur] = red
        return b

    def host_K(self):
        K = self.K
        return sp.csr_matrix((self._d2h(K.data), self._d2h(K.indices), self._d2h(K.indptr)), shape=K.shape)

    def set_solution(self, U):
        """Upload the solution coefficients U (N, k) once; they stay resident for points and energy."""
        U = np.asarray(U, dtype=np.complex128).reshape(self.N, -1)
        with self._resident():
            self.U = self._h2d(U)

    def points(self, elems, xi, eta, keys=POINT_KEYS, use=None):
        """H and the requested gradient and Hessian components (lengths of the mesh, here nm; keys default to all six)
        for every column of the resident U at the reference points (xi, eta) = (lambda0, lambda1) of the given
        elements: dict of host arrays (n, k). use(values): a further consumer of the same device values (dict of (n, k)
        CuPy arrays, mesh lengths) inside the stage, before their download and release; it keeps none of them.
        """
        with self._stage():
            out = self._points(
                self.U,
                np.asarray(elems, dtype=np.int64),
                np.asarray(xi, dtype=np.float64),
                np.asarray(eta, dtype=np.float64),
                keys,
            )
            if use is not None:
                use(out)
            out = {key: self._d2h(v) for key, v in out.items()}
        return out

    def _basis(self, E, xi, eta, want0, order):
        """Shape functions (m, 15) and their first (m, 15, 2) and second (m, 15, 3) reference derivatives at the
        reference points (xi, eta) of the elements E, as requested (else None): the compiled p4_basis."""
        cp = self.cp
        m = int(E.size)
        f0 = cp.empty((m, 15), dtype=cp.float64) if want0 else None
        d1 = cp.empty((m, 15, 2), dtype=cp.float64) if order >= 1 else None
        d2 = cp.empty((m, 15, 3), dtype=cp.float64) if order >= 2 else None
        none = cp.empty(0, dtype=cp.float64)
        basis, _ = _kernels(cp)
        args = (
            cp.ascontiguousarray(xi, dtype=cp.float64),
            cp.ascontiguousarray(eta, dtype=cp.float64),
            cp.ascontiguousarray(E, dtype=cp.int64),
            self.cls,
            self.C,
            np.int64(m),
            np.int32(bool(want0)),
            np.int32(order),
            none if f0 is None else f0,
            none if d1 is None else d1,
            none if d2 is None else d2,
        )
        _launch(basis, 15 * m, args)
        return f0, d1, d2

    def _accumulate(self, f, E, Ud, out):
        """out (m, k) = sum over the local DOFs i = 0..14, in order, of f[:, i] Ud[D[E, i], :]: the compiled
        p4_accumulate; out is a C-contiguous complex128 array (m, k) or row slice of one."""
        cp = self.cp
        _, accumulate = _kernels(cp)
        m, k = int(E.size), int(Ud.shape[1])
        args = (
            cp.ascontiguousarray(f, dtype=cp.float64),
            cp.ascontiguousarray(E, dtype=cp.int64),
            self.D,
            Ud,
            np.int64(m),
            np.int64(k),
            out,
        )
        _launch(accumulate, m * k, args)

    def _points(self, Ud, elems, xi, eta, keys=POINT_KEYS):
        """The requested quantities only: no Hessian transforms or second reference derivatives without a second
        derivative, no geometry or first derivatives without any derivative. The basis and coefficient sums are the
        compiled p4_basis and p4_accumulate."""
        cp = self.cp
        n, k = elems.size, int(Ud.shape[1])
        keys = [key for key in POINT_KEYS if key in keys]
        order = 2 if {"Hxx", "Hxz", "Hzz"} & set(keys) else 1 if {"Hx", "Hz"} & set(keys) else 0
        out = {key: cp.empty((n, k), dtype=cp.complex128) for key in keys}
        for s0 in range(0, n, self.chunk_points):
            sl = slice(s0, min(n, s0 + self.chunk_points))
            E = self._h2d(elems[sl])
            phi, d1, d2 = self._basis(E, self._h2d(xi[sl]), self._h2d(eta[sl]), "H" in keys, order)
            f = {}
            if "H" in keys:
                f["H"] = phi  # (m, 15)
            if order:
                _, _, _, (I00, I01, I10, I11) = self._geometry(E)
                a, b_, c, d = (v[:, None] for v in (I00, I10, I01, I11))  # J^-T = [[I00, I10], [I01, I11]]
                f["Hx"] = a * d1[:, :, 0] + b_ * d1[:, :, 1]
                f["Hz"] = c * d1[:, :, 0] + d * d1[:, :, 1]
            if order == 2:
                f["Hxx"] = a * a * d2[:, :, 0] + 2.0 * a * b_ * d2[:, :, 1] + b_ * b_ * d2[:, :, 2]
                f["Hxz"] = a * c * d2[:, :, 0] + (a * d + b_ * c) * d2[:, :, 1] + b_ * d * d2[:, :, 2]
                f["Hzz"] = c * c * d2[:, :, 0] + 2.0 * c * d * d2[:, :, 1] + d * d * d2[:, :, 2]
            for key in keys:
                self._accumulate(f[key], E, Ud, out[key][sl])
        return out

    def close(self):
        for name in self._res:
            setattr(self, name, None)
        self._res.clear()
        self.K = self.b = self.U = None
        self.scratch_pool.free_all_blocks()
        self.res_pool.free_all_blocks()


def batch_views(setup, n_batches):
    import types

    ne = int(setup.ne)
    bounds = np.linspace(0, ne, n_batches + 1).astype(np.int64)
    views = []
    for b in range(n_batches):
        idx = np.arange(bounds[b], bounds[b + 1])
        v = types.SimpleNamespace(**{k: val for k, val in vars(setup).items()})
        v.T, v.D, v.cls, v.ne = setup.T[idx], setup.D[idx], setup.cls[idx], int(idx.size)
        views.append((idx, v))
    return views


def _element_matrix(setup, eps_el, k0_nm, pml):
    gpu = GPUTri4(setup, eps_el, "TM element matrices")
    try:
        gpu.assemble(k0_nm, pml)
        return gpu.host_K()
    finally:
        gpu.close()


def assemble_operator(setup, eps_el, k0_nm, pml):
    """Use global element order for the deterministic sparse gather of the paper jobs."""
    import cupy as cp

    try:
        return _element_matrix(setup, eps_el, k0_nm, pml)
    except cp.cuda.memory.OutOfMemoryError:
        cp.get_default_memory_pool().free_all_blocks()

    # A smaller device can assemble partial matrices and combine them in host memory.
    rows, columns, values = [], [], []
    batch_count = max(2, (setup.ne + 65535) // 65536)
    for indices, view in batch_views(setup, batch_count):
        batch = _element_matrix(view, eps_el[indices], k0_nm, pml).tocoo()
        rows.append(batch.row)
        columns.append(batch.col)
        values.append(batch.data)
    matrix = sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(setup.ndof, setup.ndof),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def contrast_load_p4(setup, mat_elem, bg_elem, eps_codes, stack, order, chunk=512):
    """b_i = int (eps_b^-1 - eps^-1) grad H_b . grad N_i dA on the P4 mesh (lengths nm, as the GPU assembly) with the
    NGSolve TRIG rule of the given order (6 = the stiffness rule of GPUTri4), eps_b the permittivity of the element's
    background region of the stack; elements with eps = eps_b contribute exactly zero. The background gradients are
    converted from per m to per nm (1e-9) once. Returns the load (ndof,), the rule record and the contrast elements.
    """
    import ngsolve as ng

    eps_c = np.asarray(eps_codes, dtype=np.complex128)
    mat_elem = np.asarray(mat_elem, dtype=np.int64)
    bg_elem = np.asarray(bg_elem, dtype=np.int64)
    bg_mat = np.asarray(stack.region_mat)[bg_elem]
    coef = 1.0 / eps_c[bg_mat] - 1.0 / eps_c[mat_elem]
    E = np.flatnonzero(coef != 0)
    ir = ng.IntegrationRule(ng.TRIG, int(order))
    xs = np.array([p[:2] for p in ir.points], dtype=np.float64)
    ws = np.array(list(ir.weights), dtype=np.float64)
    nq = xs.shape[0]
    _, m1, _ = monomials(np, xs[:, 0], xs[:, 1])  # (nq, 15, 2)
    dphi = np.einsum("qmd,cmi->cqid", m1, setup.C)  # (6, nq, 15, 2)
    b = np.zeros(setup.ndof, dtype=np.complex128)
    for s0 in range(0, E.size, chunk):
        Ec = E[s0 : s0 + chunk]
        V = setup.P[setup.T[Ec]]  # (c, 3, 2)
        J00, J01 = V[:, 0, 0] - V[:, 2, 0], V[:, 1, 0] - V[:, 2, 0]
        J10, J11 = V[:, 0, 1] - V[:, 2, 1], V[:, 1, 1] - V[:, 2, 1]
        det = J00 * J11 - J01 * J10
        I00, I01, I10, I11 = J11 / det, -J01 / det, -J10 / det, J00 / det
        X = V[:, 2, 0][:, None] + J00[:, None] * xs[None, :, 0] + J01[:, None] * xs[None, :, 1]  # (c, nq)
        Z = V[:, 2, 1][:, None] + J10[:, None] * xs[None, :, 0] + J11[:, None] * xs[None, :, 1]
        d = dphi[setup.cls[Ec]]  # (c, nq, 15, 2)
        gx = I00[:, None, None] * d[..., 0] + I10[:, None, None] * d[..., 1]  # (c, nq, 15)
        gz = I01[:, None, None] * d[..., 0] + I11[:, None, None] * d[..., 1]
        F = stack.fields(X.ravel(), Z.ravel(), np.repeat(bg_elem[Ec], nq), keys=("Hx", "Hz"))
        hx = (F["Hx"] * 1e-9).reshape(Ec.size, nq)  # per nm
        hz = (F["Hz"] * 1e-9).reshape(Ec.size, nq)
        be = (coef[Ec] * np.abs(det))[:, None] * np.einsum(
            "q,cqi->ci", ws, gx * hx[:, :, None] + gz * hz[:, :, None]
        )
        np.add.at(b, setup.D[Ec].ravel(), be.ravel())
    return (
        b,
        {
            "elements": int(E.size),
            "rule": "NGSolve TRIG order %d (%d points)" % (int(order), nq),
            "points": int(E.size * nq),
        },
        E,
    )


P4_SI = {"H": None, "Hx": 1e9, "Hz": 1e9, "Hxx": 1e18, "Hxz": 1e18, "Hzz": 1e18}  # nm -> m factors


def p4_vals(v, col=0):
    """SI values of one coefficient column from GPUTri4.points (lengths of the mesh, nm) for the quantities present: H
    unchanged, H_x and H_z times 1e9, the second derivatives times 1e18 (applied exactly once)."""
    return {key: v[key][:, col] if s is None else v[key][:, col] * s for key, s in P4_SI.items() if key in v}
