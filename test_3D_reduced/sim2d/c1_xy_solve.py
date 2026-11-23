"""GPU assembly of the corner operator with its equivalent line feed."""

import numpy as np
import scipy.sparse as sp
from sim2d import fem_p4
from . import c1_xy_common as cc

RULE = "mass"
CHUNK_ELEMENTS = 4096


def reference_gradients(tab):
    """Reference gradients (6, nq, 15, 2) of the P4 class expansions at the points of the assembly rule (tab.phi_m holds
    the values of the same rule), from fem_p4.monomials and tab.C as P4Setup forms dphi_s for the stiffness rule.
    """
    xm = np.asarray(tab.rules[RULE]["points"], dtype=np.float64)
    m1 = fem_p4.monomials(np, xm[:, 0], xm[:, 1])[1]
    return np.einsum("qmd,cmi->cqid", m1, tab.C)


class XYAssembler:
    """P4 coefficient integration and sparse gather over one batch of elements."""

    def __init__(self, tab, label, chunk_elements=CHUNK_ELEMENTS, elements=None):
        import cupy as cp

        self.cp, self.tab, self.label = cp, tab, label
        self.N = int(tab.ndof)
        if elements is None:
            self.elements = np.arange(int(tab.ne), dtype=np.int64)
        else:
            self.elements = np.asarray(elements, dtype=np.int64)
        self.ne = int(self.elements.size)
        self.chunk = max(1, min(int(chunk_elements), self.ne))
        self.xm = np.asarray(tab.rules[RULE]["points"], dtype=np.float64)
        self.wm = np.asarray(tab.rules[RULE]["weights"], dtype=np.float64)
        self.nq = int(self.wm.size)
        self.dphi_m = reference_gradients(tab)

    def _chunk_host(self, E, coeff_fn):
        """Host part of one chunk: the affine maps (|det| and J^-1), the physical rule points and the coefficients."""
        tab = self.tab
        V = tab.P[tab.T[E]]  # (m, 3, 2)
        x2, y2 = V[:, 2, 0], V[:, 2, 1]
        J00, J01 = V[:, 0, 0] - x2, V[:, 1, 0] - x2
        J10, J11 = V[:, 0, 1] - y2, V[:, 1, 1] - y2
        det = J00 * J11 - J01 * J10
        xm = self.xm
        X = x2[:, None] + J00[:, None] * xm[None, :, 0] + J01[:, None] * xm[None, :, 1]  # (m, nq)
        Y = y2[:, None] + J10[:, None] * xm[None, :, 0] + J11[:, None] * xm[None, :, 1]
        coef = coeff_fn(X, Y)
        out = []
        for v in coef:
            a = np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.complex128), X.shape))
            out.append(a)
        inv = np.stack([J11 / det, -J01 / det, -J10 / det, J00 / det], axis=1)  # I00, I01, I10, I11
        return np.abs(det), inv, out

    def assemble(self, coeff_fn):
        import cupyx.scipy.sparse as csp

        cp, tab = self.cp, self.tab
        D = cp.asarray(tab.D[self.elements])
        cls_d = cp.asarray(tab.cls[self.elements])
        phi_m = cp.asarray(np.ascontiguousarray(tab.phi_m))
        dphi_m = cp.asarray(self.dphi_m)
        vals = cp.empty((self.ne, 15, 15), dtype=cp.complex128)
        for c0 in range(0, self.ne, self.chunk):
            E = self.elements[c0 : c0 + self.chunk]
            m = int(E.size)
            adet_h, inv_h, (cx_h, cy_h, mm_h) = self._chunk_host(E, coeff_fn)
            cl = cls_d[c0 : c0 + m]
            dph = dphi_m[cl]  # (m, nq, 15, 2) reference gradients
            ph = phi_m[cl]  # (m, nq, 15) values
            inv = cp.asarray(inv_h)
            adet = cp.asarray(adet_h)
            cxd, cyd, mmd = cp.asarray(cx_h), cp.asarray(cy_h), cp.asarray(mm_h)
            I00, I01, I10, I11 = (inv[:, k][:, None] for k in range(4))
            Ke = cp.zeros((m, 15, 15), dtype=cp.complex128)
            for q in range(self.nq):
                gx = I00 * dph[:, q, :, 0] + I10 * dph[:, q, :, 1]  # d/dx (per nm), (m, 15)
                gy = I01 * dph[:, q, :, 0] + I11 * dph[:, q, :, 1]  # d/dy
                p = ph[:, q, :]
                wq = float(self.wm[q]) * adet
                c1 = (cxd[:, q] * wq)[:, None, None]
                c2 = (cyd[:, q] * wq)[:, None, None]
                c3 = (mmd[:, q] * wq)[:, None, None]
                Ke += (
                    c1 * (gx[:, :, None] * gx[:, None, :])
                    + c2 * (gy[:, :, None] * gy[:, None, :])
                    + c3 * (p[:, :, None] * p[:, None, :])
                )
            vals[c0 : c0 + m] = Ke
            del Ke, dph, ph, inv, adet, cxd, cyd, mmd, cl, I00, I01, I10, I11, gx, gy, p, wq, c1, c2, c3
        rows = cp.broadcast_to(D[:, :, None], vals.shape).ravel()  # test function i
        cols = cp.broadcast_to(D[:, None, :], vals.shape).ravel()  # trial function j
        v = vals.ravel()
        del vals
        (uc, ur), red = fem_p4._deterministic_sum(cp, (cols, rows), v)
        del rows, cols, v
        coo = csp.coo_matrix((red, (ur.astype(cp.int32), uc.astype(cp.int32))), shape=(self.N, self.N))
        coo.has_canonical_format = True
        A = coo.tocsr()
        del coo, ur, uc, red
        K = sp.csr_matrix(
            (cp.asnumpy(A.data), cp.asnumpy(A.indices), cp.asnumpy(A.indptr)), shape=(self.N, self.N)
        )
        return K


def assemble_operator(tab, coeff_fn):
    import cupy as cp

    parts = []
    for elements in np.array_split(np.arange(tab.ne), max(1, (tab.ne + 32767) // 32768)):
        parts.append(XYAssembler(tab, "corner", elements=elements).assemble(coeff_fn))
        cp.get_default_memory_pool().free_all_blocks()
    matrix = sum(parts[1:], parts[0]).tocsr()
    matrix.sort_indices()
    return matrix


def line_load(tab, a_bar_p):
    """b (ndof,) complex128 of (10): b_i = -2 a_bar_p sum over the source edges k and rule points q of
    weights_length[k, q] s_y(y_q) N_i(ref_q) over the five trace DOFs of each edge (zero elsewhere); the basis values
    from fem_p4.monomials and tab.C at the stored reference points, s_y from c1_xy_common.pml_stretch at the edge
    points (s_x = 1 on the feed line, inside the real domain in x). a_bar_p in A/m; b in A/m nm."""
    se = tab.source_edges
    es = np.asarray(se["elements"], dtype=np.int64)
    l5 = np.asarray(se["local5"], dtype=np.int64)
    ref = np.asarray(se["ref"], dtype=np.float64)
    wl = np.asarray(se["weights_length"], dtype=np.float64)
    xq = np.asarray(se["x_q"], dtype=np.float64)
    yq = np.asarray(se["y_q"], dtype=np.float64)
    k, nq = wl.shape
    m0, _, _ = fem_p4.monomials(np, ref[..., 0].ravel(), ref[..., 1].ravel())  # (k nq, 15)
    Cc = tab.C[tab.cls[es]]  # (k, 15, 15)
    phi = np.einsum("kqm,kmi->kqi", m0.reshape(k, nq, 15), Cc)  # (k, nq, 15)
    sel = np.take_along_axis(phi, np.broadcast_to(l5[:, None, :], (k, nq, 5)), axis=2)  # the trace DOFs
    _, sy = cc.pml_stretch(xq, yq)
    contrib = -2.0 * complex(a_bar_p) * np.einsum("kq,kqi->ki", wl * sy, sel)  # (k, 5)
    b = np.zeros(int(tab.ndof), dtype=np.complex128)
    np.add.at(b, np.take_along_axis(tab.D[es], l5, axis=1).ravel(), contrib.ravel())
    return b
