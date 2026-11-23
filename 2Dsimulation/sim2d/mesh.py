"""Build the Netgen mesh, define the material-side point search, and prepare P4 basis tables."""

from pathlib import Path
import numpy as np
from . import fem_p4, geometry, materials, storage

TOL_NM = 1e-9
LOCATE_CHUNK = 16384


class Locator:
    """Material-side element choice on the mesh triangles (C1 nm): NGSolve's point search gives one containing
    triangle; points on an edge or a vertex (or in a triangle of another material) are resolved among all triangles
    sharing a vertex of it, within the declared material, then the +x triangle (centroid), the +z triangle on a
    horizontal edge. Returns element numbers and barycentric (lambda0, lambda1) as NGSolve's MeshPoint stores them.
    """

    def __init__(self, mesh, npz):
        with np.load(npz, allow_pickle=False) as d:
            self.P, self.T, self.mat = d["points_nm"], d["triangles"], d["material"]
        self.mesh = mesh
        nv = self.P.shape[0]
        flat = self.T.ravel()
        order = np.argsort(flat, kind="stable")
        self.v2t_ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=nv))])
        self.v2t = (order // 3).astype(np.int64)
        self.cen = self.P[self.T].mean(axis=1)

    def bary(self, e, x, z):
        V = self.P[self.T[e]]
        a = V[..., 0, 0] - V[..., 2, 0]
        b = V[..., 1, 0] - V[..., 2, 0]
        c = V[..., 0, 1] - V[..., 2, 1]
        d = V[..., 1, 1] - V[..., 2, 1]
        det = a * d - b * c
        rx, rz = x - V[..., 2, 0], z - V[..., 2, 1]
        l0 = (d * rx - b * rz) / det
        l1 = (-c * rx + a * rz) / det
        return l0, l1

    def locate(self, x, z, mat, tol=1e-10):
        x, z, mat = (np.asarray(v).ravel() for v in (x, z, mat))
        mp = self.mesh(x, z)
        e0 = np.asarray(mp["nr"], dtype=np.int64)
        l0, l1 = self.bary(e0, x, z)
        l2 = 1.0 - l0 - l1
        interior = (np.minimum(np.minimum(l0, l1), l2) > tol) & (self.mat[e0] == mat) & (e0 >= 0)
        el = np.where(interior, e0, -1)
        rest = np.flatnonzero(~interior)
        for s0 in range(0, rest.size, LOCATE_CHUNK):
            k = rest[s0 : s0 + LOCATE_CHUNK]
            el[k] = self._choose(k, x, z, mat, e0, tol)
        l0, l1 = self.bary(el, x, z)
        mp["nr"] = el
        mp["x"] = np.clip(l0, 0.0, 1.0)
        mp["y"] = np.clip(l1, 0.0, 1.0)
        return mp, el

    def _choose(self, k, x, z, mat, e0, tol):
        """The material-side choice for the points k that are not strictly inside their found triangle, all at once:
        per point the sorted unique triangles sharing a vertex of the found triangle (rows padded with -1, never
        selected), the closed barycentric test and the declared material; then, among the remaining candidates in that
        order, the largest centroid z when every centroid lies within |dz| + 1e-12 of the first one in x (vertical
        edge), else the right-most ones (centroid x within 1e-9 max(1, ptp x + ptp z) of the largest) and among them the
        largest centroid z; the first index on ties. A point without candidate stops with the error of the first such
        point."""
        n, rows = k.size, np.arange(k.size)
        ek = e0[k]
        has = ek >= 0
        seeds = self.T[np.where(has, ek, 0)]  # (n, 3)
        lo, hi = self.v2t_ptr[seeds], self.v2t_ptr[seeds + 1]
        deg = np.where(has[:, None], hi - lo, 0)
        j = np.arange(int(deg.max()) if n else 0)
        slot = np.minimum(lo[:, :, None] + j, self.v2t.size - 1)
        cand = np.where(j < deg[:, :, None], self.v2t[slot], -1).reshape(n, -1)
        cand.sort(axis=1)
        valid = cand >= 0
        valid[:, 1:] &= cand[:, 1:] != cand[:, :-1]
        c = np.where(valid, cand, 0)
        a0, a1 = self.bary(c, x[k][:, None], z[k][:, None])
        a2 = 1.0 - a0 - a1
        ok = valid & (np.minimum(np.minimum(a0, a1), a2) >= -tol) & (self.mat[c] == mat[k][:, None])
        none_ = ~ok.any(axis=1)
        if np.any(none_):
            i = k[int(np.argmax(none_))]
            raise ValueError(
                "material-side point location: no triangle of material %d contains (%r, %r)"
                % (mat[i], x[i], z[i])
            )
        cx, cz = self.cen[c, 0], self.cen[c, 1]
        f = np.argmax(ok, axis=1)
        near = np.abs(cx - cx[rows, f][:, None]) <= np.abs(cz - cz[rows, f][:, None]) + 1e-12
        vertical = (ok.sum(axis=1) > 1) & np.all(~ok | near, axis=1)
        cx_max, cx_min = np.where(ok, cx, -np.inf).max(axis=1), np.where(ok, cx, np.inf).min(axis=1)
        cz_max, cz_min = np.where(ok, cz, -np.inf).max(axis=1), np.where(ok, cz, np.inf).min(axis=1)
        right = cx >= (cx_max - 1e-9 * np.maximum(1.0, (cx_max - cx_min) + (cz_max - cz_min)))[:, None]
        sel = np.where(vertical[:, None], ok, ok & right)
        return c[rows, np.argmax(np.where(sel, cz, -np.inf), axis=1)]


def bary(P, T, el, x, z):
    """NGSolve reference coordinates (lambda0, lambda1) of the points in the given triangles (as Locator.bary)."""
    V = P[T[el]]
    a = V[..., 0, 0] - V[..., 2, 0]
    b = V[..., 1, 0] - V[..., 2, 0]
    c = V[..., 0, 1] - V[..., 2, 1]
    d = V[..., 1, 1] - V[..., 2, 1]
    det = a * d - b * c
    rx, rz = x - V[..., 2, 0], z - V[..., 2, 1]
    return (d * rx - b * rz) / det, (-c * rx + a * rz) / det


def locate_p4(loc, x, z, mat, side):
    """Element and reference coordinates of every point: Locator (material-side choice) for points without a declared
    side; for trace points, among all triangles containing the exact point (barycentric >= -1e-9), those of the
    declared material whose centroid lies on the declared side, then the one containing the point shifted 1e-3 nm to
    that side most deeply; reference coordinates of the exact point, clipped to [0, 1]."""
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    mat, side = np.asarray(mat, dtype=np.int64), np.asarray(side, dtype=np.int64)
    el = np.full(x.size, -1, dtype=np.int64)
    l0, l1 = np.zeros(x.size), np.zeros(x.size)
    m0 = side == 0
    if np.any(m0):
        mp, e = loc.locate(x[m0], z[m0], mat[m0])
        el[m0], l0[m0], l1[m0] = e, np.asarray(mp["x"]), np.asarray(mp["y"])
    allt = np.arange(loc.T.shape[0])
    for k in np.flatnonzero(~m0):
        a0, a1 = bary(loc.P, loc.T, allt, np.full(allt.size, x[k]), np.full(allt.size, z[k]))
        inside = np.minimum(np.minimum(a0, a1), 1.0 - a0 - a1) >= -1e-9
        cand = allt[inside & (loc.mat == mat[k]) & ((loc.cen[:, 1] - z[k]) * side[k] > 0)]
        if cand.size == 0:
            raise ValueError(
                "trace point (%r, %r) has no triangle of material %d on side %+d"
                % (x[k], z[k], mat[k], side[k])
            )
        s0, s1 = bary(loc.P, loc.T, cand, np.full(cand.size, x[k]), np.full(cand.size, z[k] + 1e-3 * side[k]))
        e = int(cand[int(np.argmax(np.minimum(np.minimum(s0, s1), 1.0 - s0 - s1)))])
        b0, b1 = bary(loc.P, loc.T, np.array([e]), np.array([x[k]]), np.array([z[k]]))
        el[k], l0[k], l1[k] = e, float(np.clip(b0[0], 0.0, 1.0)), float(np.clip(b1[0], 0.0, 1.0))
    return el, l0, l1


def line_rule_p4(P, T, axis, c, lo, hi, inward, n):
    """Exact intersections of the line coord[axis] = c (other coordinate in [lo, hi]) with the triangles, the triangle
    on the inward side (inward = +1: larger coord[axis]) where the line runs along an edge; n Gauss-Legendre points
    per intersection segment: elements, (lambda0, lambda1), physical points and weights (nm). All candidate triangles
    are treated together with masks (on-line vertices, edge crossings, the inward-edge rule), in triangle order.
    """
    other = 1 - axis
    V = P[T]
    a = V[:, :, axis] - c
    on = np.abs(a) <= TOL_NM
    cand = np.flatnonzero(~(np.all(a > TOL_NM, axis=1) | np.all(a < -TOL_NM, axis=1)))
    t, w = np.polynomial.legendre.leggauss(n)
    ac, bc, oc = a[cand], V[cand, :, other], on[cand]
    non = oc.sum(axis=1)
    lo_pts = np.where(oc, bc, np.inf)  # the on-line vertices
    hi_pts = np.where(oc, bc, -np.inf)
    npts = non.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            cross = ~oc[:, i] & ~oc[:, j] & (ac[:, i] * ac[:, j] < 0)
            p = bc[:, i] - ac[:, i] * (bc[:, j] - bc[:, i]) / (ac[:, j] - ac[:, i])
            lo_pts = np.concatenate([lo_pts, np.where(cross, p, np.inf)[:, None]], axis=1)
            hi_pts = np.concatenate([hi_pts, np.where(cross, p, -np.inf)[:, None]], axis=1)
            npts += cross
    off = np.argmax(~oc, axis=1)  # the off-line vertex of an edge on the line
    edge_ok = (non == 2) & (ac[np.arange(cand.size), off] * inward > 0)
    keep = np.where(non >= 2, edge_ok, npts >= 2)
    s0 = np.maximum(lo_pts.min(axis=1), lo)
    s1 = np.minimum(hi_pts.max(axis=1), hi)
    keep &= s1 - s0 > TOL_NM
    e, s0, s1 = cand[keep], s0[keep], s1[keep]
    s = 0.5 * (s1 - s0)[:, None] * (t + 1.0)[None, :] + s0[:, None]  # (m, n)
    X = np.empty(s.shape + (2,))
    X[..., axis], X[..., other] = c, s
    el = np.repeat(e, n)
    l0, l1 = bary(P, T, el, X[..., 0].ravel(), X[..., 1].ravel())
    return {
        "el": el,
        "l0": np.clip(l0, 0.0, 1.0),
        "l1": np.clip(l1, 0.0, 1.0),
        "x": X[..., 0].ravel(),
        "z": X[..., 1].ravel(),
        "w": (0.5 * (s1 - s0)[:, None] * w[None, :]).ravel(),
    }


def outer_boundary_dofs(mesh, fes, P, L):
    """DOFs of the H1 space on the outer boundary of the computational domain (the four sides of [X0, X1] x [Z0, Z1]),
    from the boundary elements whose vertices lie on one side (the unnamed 'default' boundary also covers internal
    face-to-face edges, so a geometric filter is used)."""
    import ngsolve as ng

    X0, X1, Z0, Z1 = L["X0"], L["X1"], L["Z0"], L["Z1"]
    dofs, n_edges = set(), 0
    for el in mesh.Elements(ng.BND):
        p = P[[v.nr for v in el.vertices]]
        on = (
            np.all(np.abs(p[:, 0] - X0) <= TOL_NM)
            or np.all(np.abs(p[:, 0] - X1) <= TOL_NM)
            or np.all(np.abs(p[:, 1] - Z0) <= TOL_NM)
            or np.all(np.abs(p[:, 1] - Z1) <= TOL_NM)
        )
        if on:
            n_edges += 1
            dofs.update(int(d) for d in fes.GetDofNrs(el))
    bnd = np.zeros(fes.ndof, dtype=bool)
    bnd[np.array(sorted(dofs), dtype=np.int64)] = True
    nv_on = int(
        np.sum(
            (np.abs(P[:, 0] - X0) <= TOL_NM)
            | (np.abs(P[:, 0] - X1) <= TOL_NM)
            | (np.abs(P[:, 1] - Z0) <= TOL_NM)
            | (np.abs(P[:, 1] - Z1) <= TOL_NM)
        )
    )
    return bnd, n_edges, nv_on


def construct(cfg, out_dir, desc, name=None):
    """One Netgen build of the configured device from the faces of its geometry description (geometry.description: C1
    faces with region maxh, the named edges and the background partitions; Glue, OCCGeometry(dim=2).GenerateMesh(maxh
    = h_free x h_scale, grading)); writes mesh.vol/.npz and its manifest into the new out_dir."""
    import ngsolve
    from netgen.occ import Glue, MoveTo, OCCGeometry, Y

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = cfg["parameters"]["mesh"]
    specs = desc["faces"]
    faces = []
    for s in specs:
        wire = MoveTo(*s["points"][0])
        for q in s["points"][1:]:
            wire = wire.LineTo(*q)
        f = wire.Close().Face()
        f.name = s["name"]
        if s["maxh"]:
            f.maxh = s["maxh"]
        if s["top"]:
            f.edges.Max(Y).name = s["top"]
        if s["bottom"]:
            f.edges.Min(Y).name = s["bottom"]
        faces.append(f)
    geo = OCCGeometry(Glue(faces), dim=2)
    ngmesh = geo.GenerateMesh(maxh=float(mp["h_free"]) * float(mp["h_scale"]), grading=float(mp["grading"]))
    mesh = ngsolve.Mesh(ngmesh)
    pts = np.array([v.point for v in mesh.vertices], dtype=np.float64)[:, :2]
    tri = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngsolve.VOL)], dtype=np.int64)
    mat = np.array([materials.FACE_NAMES[el.mat] for el in mesh.Elements(ngsolve.VOL)], dtype=np.int64)
    seg = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngsolve.BND)], dtype=np.int64)
    segname = [el.mat for el in mesh.Elements(ngsolve.BND)]
    ngmesh.Save(str(out_dir / "mesh.vol"))
    np.savez(
        out_dir / "mesh.npz",
        points_nm=pts,
        triangles=tri,
        material=mat,
        boundary_segments=seg,
        boundary_names=np.array(segname, dtype=str),
        domain_materials=np.array(mesh.GetMaterials(), dtype=str),
    )


def prepare(cfg, directory):
    import ngsolve as ng

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    volume = directory / "mesh.vol"
    if not volume.is_file():
        desc = geometry.description(cfg)
        storage.write_json(directory.parent / "geometry" / "geometry.json", desc)
        construct(cfg, directory, desc)
    mesh = ng.Mesh(str(volume))
    fes = ng.H1(mesh, order=4, complex=True)
    tables = fem_p4.P4Setup(mesh, fes)
    tables.material = np.array([materials.FACE_NAMES[m] for m in tables.mat_names], dtype=np.int64)
    tables.source_edges = tables.sheet_edges(mesh, fes, "source", -1)
    tables.boundary = outer_boundary_dofs(mesh, fes, tables.P, cfg["layout"])[0]
    return tables, Locator(mesh, directory / "mesh.npz")


def write_tables(tab, stem):
    arrays = {
        key: getattr(tab, key)
        for key in ("P", "T", "D", "C", "cls", "dphi_s", "phi_m", "material", "boundary")
    }
    for name, rule in tab.rules.items():
        arrays["rule_" + name + "_points"] = rule["points"]
        arrays["rule_" + name + "_weights"] = rule["weights"]
    for key, value in tab.source_edges.items():
        if isinstance(value, np.ndarray):
            arrays["source__" + key] = value
    np.savez(stem.with_suffix(".npz"), **arrays)
    storage.write_json(
        stem.with_suffix(".json"),
        {
            "nv": tab.nv,
            "ne": tab.ne,
            "ndof": tab.ndof,
            "classes_present": tab.classes_present,
            "ngsolve_version": tab.ngsolve_version,
            "rules": {k: {"element": v["element"], "order": v["order"]} for k, v in tab.rules.items()},
            "source_edges_scalars": {
                k: v for k, v in tab.source_edges.items() if not isinstance(v, np.ndarray)
            },
        },
    )
