"""Netgen P4 mesh for the local corner and its X-directed feed."""

from pathlib import Path
import time
import numpy as np
from sim2d import mesh as meshmod, fem_p4
from . import c1_xy_common as cc

BRANCH_NAMES = {
    0: "central",
    1: "xtaper",
    2: "ytaper",
    3: "plateau",
}  # face names by c1_xy_common branch code
FACE_BRANCH = {v: k for k, v in BRANCH_NAMES.items()}
PLATEAU = 3
FEED_BOX_NM = 100.0  # half-size of the refined box around the feed edge point (x_s, b)
GRADING = 0.25
ORDER = 4
SOURCE = "source"
TOL_NM = meshmod.TOL_NM  # 1e-9 nm: on-line tests of vertices and rule points
MESH_FILES = ("mesh.vol", "mesh.npz", "mesh.json")


def grid_lines():
    """The x and y lines of the face grid (nm): outer box, PML starts, the feed line (x only), a and b."""
    xs = sorted({cc.OUTER_X[0], cc.REAL_X[0], cc.XS_NM, cc.A_NM, cc.B_NM, cc.REAL_X[1], cc.OUTER_X[1]})
    ys = sorted({cc.OUTER_Y[0], cc.REAL_Y[0], cc.A_NM, cc.B_NM, cc.REAL_Y[1], cc.OUTER_Y[1]})
    return xs, ys


def feed_box():
    """(x0, x1, y0, y1) of the refined box around the feed edge point (x_s, b)."""
    return (cc.XS_NM - FEED_BOX_NM, cc.XS_NM + FEED_BOX_NM, cc.B_NM - FEED_BOX_NM, cc.B_NM + FEED_BOX_NM)


def polygon_area(pts):
    """Signed area (positive counter-clockwise)."""
    P = np.asarray(pts, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(pts):
    P = np.asarray(pts, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * float(np.sum(cr))
    return float(np.sum((x + np.roll(x, -1)) * cr) / (6.0 * a)), float(
        np.sum((y + np.roll(y, -1)) * cr) / (6.0 * a)
    )


def _rect(R):
    x0, x1, y0, y1 = R
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _overlap(R, S):
    x0, x1 = max(R[0], S[0]), min(R[1], S[1])
    y0, y1 = max(R[2], S[2]), min(R[3], S[3])
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        return None
    return (x0, x1, y0, y1)


def _rect_minus(R, S):
    """The rectangle R minus the sub-rectangle S (S inside R, S != R) as one counter-clockwise polygon: S sharing a
    full side of R leaves a rectangle, S sharing a corner leaves an L-shaped hexagon."""
    x0, x1, y0, y1 = R
    u0, u1, v0, v1 = S
    left, right, bottom, top = u0 == x0, u1 == x1, v0 == y0, v1 == y1
    if left and right and top:
        return [(x0, y0), (x1, y0), (x1, v0), (x0, v0)]
    if left and right and bottom:
        return [(x0, v1), (x1, v1), (x1, y1), (x0, y1)]
    if top and bottom and left:
        return [(u1, y0), (x1, y0), (x1, y1), (u1, y1)]
    if top and bottom and right:
        return [(x0, y0), (u0, y0), (u0, y1), (x0, y1)]
    if right and top:
        return [(x0, y0), (x1, y0), (x1, v0), (u0, v0), (u0, y1), (x0, y1)]
    if left and top:
        return [(x0, y0), (x1, y0), (x1, y1), (u1, y1), (u1, v0), (x0, v0)]
    if left and bottom:
        return [(u1, y0), (x1, y0), (x1, y1), (x0, y1), (x0, v1), (u1, v1)]
    if right and bottom:
        return [(x0, y0), (u0, y0), (u0, v1), (x1, v1), (x1, y1), (x0, y1)]
    raise ValueError(
        "the feed box must share a corner or a full side of every grid cell it cuts: cell %r, box %r" % (R, S)
    )


def face_specs(h_nm=25.0, h_fine_nm=10.0):
    """The faces in construction order: dicts name (branch at the centroid), tag, maxh, points (nm, counter-clockwise).
    Grid cells of grid_lines; the ridge square as its two triangles (maxh h_fine_nm); the feed box cut out of the other
    cells it crosses as its own faces (maxh h_fine_nm); every other face maxh h_nm."""
    xs, ys = grid_lines()
    box = feed_box()
    a, b = cc.A_NM, cc.B_NM
    faces = []

    def add(pts, maxh, tag):
        cx, cy = polygon_centroid(pts)
        name = BRANCH_NAMES[int(cc.branch_xy(cx, cy))]
        faces.append(
            {"name": name, "tag": tag, "maxh": float(maxh), "points": [tuple(map(float, q)) for q in pts]}
        )

    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            R = (x0, x1, y0, y1)
            if R == (a, b, a, b):
                add([(a, a), (b, b), (a, b)], h_fine_nm, "x-taper ridge triangle (y >= x)")
                add([(a, a), (b, a), (b, b)], h_fine_nm, "y-taper ridge triangle (y <= x)")
                continue
            ov = _overlap(R, box)
            if ov is None:
                add(_rect(R), h_nm, "grid cell")
            elif ov == R:
                add(_rect(R), h_fine_nm, "grid cell inside the feed box")
            else:
                add(_rect_minus(R, ov), h_nm, "grid cell minus the feed box")
                add(_rect(ov), h_fine_nm, "feed box part")
    return faces


def _is_source_edge(e):
    p, q = tuple(e.start)[:2], tuple(e.end)[:2]
    return (
        abs(p[0] - cc.XS_NM) <= TOL_NM
        and abs(q[0] - cc.XS_NM) <= TOL_NM
        and min(p[1], q[1]) >= cc.B_NM - TOL_NM
        and max(p[1], q[1]) <= cc.OUTER_Y[1] + TOL_NM
    )


def construct(out_dir, faces, h_nm, grading=GRADING):
    """Construct branch-aligned surfaces and save their element connectivity."""
    import ngsolve
    from netgen.occ import Glue, MoveTo, OCCGeometry

    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    occ, n_named = [], 0
    for s in faces:
        wire = MoveTo(*s["points"][0])
        for q in s["points"][1:]:
            wire = wire.LineTo(*q)
        f = wire.Close().Face()
        f.name = s["name"]
        f.maxh = s["maxh"]
        for e in f.edges:
            if _is_source_edge(e):
                e.name = SOURCE
                n_named += 1
        occ.append(f)
    geo = OCCGeometry(Glue(occ), dim=2)
    ngmesh = geo.GenerateMesh(maxh=float(h_nm), grading=float(grading))
    mesh = ngsolve.Mesh(ngmesh)
    pts = np.array([v.point for v in mesh.vertices], dtype=np.float64)[:, :2]
    els = list(mesh.Elements(ngsolve.VOL))
    tri = np.array([[v.nr for v in el.vertices] for el in els], dtype=np.int64)
    branch = np.array([FACE_BRANCH[el.mat] for el in els], dtype=np.int64)
    cr = branch == PLATEAU
    seg = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngsolve.BND)], dtype=np.int64)
    segname = [el.mat for el in mesh.Elements(ngsolve.BND)]
    ndof = int(ngsolve.H1(mesh, order=ORDER, complex=True).ndof)
    ngmesh.Save(str(out_dir / MESH_FILES[0]))
    np.savez(
        out_dir / MESH_FILES[1],
        **{
            "points_nm": pts,
            "triangles": tri,
            "branch": branch,
            "material": branch,  # the same codes under the key mesh.Locator reads
            "cr": cr,
            "boundary_segments": seg,
            "boundary_names": np.array(segname, dtype=str),
            "domain_materials": np.array(mesh.GetMaterials(), dtype=str),
        },
    )
    counts = {
        "vertices": int(mesh.nv),
        "triangles": int(mesh.ne),
        "edges": int(mesh.nedge),
        "H1_complex_ndof": ndof,
        "source_segments": int(sum(1 for n in segname if n == SOURCE)),
        "boundary_segments": int(len(segname)),
    }
    return counts, time.perf_counter() - t0, ngsolve.__version__, n_named


def outer_x_boundary_dofs(mesh, fes, P):
    """DOFs of the H1 space on the outer X sides only (x = OUTER_X0 or x = OUTER_X1), from the boundary elements whose
    vertices all lie on one of them (mesh.outer_boundary_dofs filtered by x); the Y sides are left unconstrained.
    Returns (mask, number of such boundary elements, number of vertices on the two sides)."""
    import ngsolve as ng

    X0, X1 = cc.OUTER_X
    dofs, n_edges = set(), 0
    for el in mesh.Elements(ng.BND):
        p = P[[v.nr for v in el.vertices]]
        if np.all(np.abs(p[:, 0] - X0) <= TOL_NM) or np.all(np.abs(p[:, 0] - X1) <= TOL_NM):
            n_edges += 1
            dofs.update(int(d) for d in fes.GetDofNrs(el))
    bnd = np.zeros(fes.ndof, dtype=bool)
    if dofs:
        bnd[np.array(sorted(dofs), dtype=np.int64)] = True
    nv_on = int(np.sum((np.abs(P[:, 0] - X0) <= TOL_NM) | (np.abs(P[:, 0] - X1) <= TOL_NM)))
    return bnd, n_edges, nv_on


def source_edges(s, mesh, fes, axis=0, coordinate=cc.XS_NM, side=1, name=SOURCE):
    """Axis-aware version of fem_p4.P4Setup.sheet_edges: the boundary segments named `name` (the feed line coord[axis]
    = coordinate), the adjacent element on the given side of that coordinate (by the element centroid), local vertex
    indices and the 5 local DOFs of the edge trace (fem_p4.EDGE_DOFS), the SEGM order-8 points as reference
    coordinates (lambda0, lambda1) of that element, the weights times the edge length (nm) and the physical points
    (x_q, y_q)."""
    import ngsolve as ng

    segs = []
    for el in mesh.Elements(ng.BND):
        if el.mat == name:
            segs.append([v.nr for v in el.vertices])
    segs = np.array(segs, dtype=np.int64).reshape(-1, 2)
    flat = s.T.ravel()
    order = np.argsort(flat, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=s.nv))])
    v2t = order // 3
    cen = s.P[s.T].mean(axis=1)
    t, w = s.rules["sheet"]["points"][:, 0], s.rules["sheet"]["weights"]
    es, loc5, ref, wl, xq, yq = [], [], [], [], [], []
    for va, vb in segs:
        ea = set(v2t[ptr[va] : ptr[va + 1]].tolist()) & set(v2t[ptr[vb] : ptr[vb + 1]].tolist())
        c = 0.5 * (s.P[va, axis] + s.P[vb, axis])
        cand = [e for e in ea if (cen[e, axis] - c) * side > 0]
        e = cand[0]
        la, lb = int(np.flatnonzero(s.T[e] == va)[0]), int(np.flatnonzero(s.T[e] == vb)[0])
        blk = fem_p4.EDGE_DOFS[(min(la, lb), max(la, lb))]
        l5 = [la, lb] + list(blk)
        lam = np.zeros((t.size, 3))
        lam[:, la], lam[:, lb] = t, 1.0 - t
        L = float(np.hypot(*(s.P[va] - s.P[vb])))
        X = np.outer(t, s.P[va]) + np.outer(1.0 - t, s.P[vb])
        es.append(e)
        loc5.append(l5)
        ref.append(lam[:, :2])
        wl.append(w * L)
        xq.append(X[:, 0])
        yq.append(X[:, 1])
    k = len(es)
    nq = int(t.size)
    out = {
        "segments": segs,
        "elements": np.array(es, dtype=np.int64),
        "local5": np.array(loc5, dtype=np.int64).reshape(k, 5),
        "ref": np.array(ref, dtype=np.float64).reshape(k, nq, 2),
        "weights_length": np.array(wl, dtype=np.float64).reshape(k, nq),
        "x_q": np.array(xq, dtype=np.float64).reshape(k, nq),
        "y_q": np.array(yq, dtype=np.float64).reshape(k, nq),
        "t": np.asarray(t, dtype=np.float64),
        "side": int(side),
        "axis": int(axis),
        "coordinate_nm": float(coordinate),
        "length": float(np.sum(np.array(wl)) / np.sum(w)) if k else 0.0,
        "rule": "SEGM order 8 (%d points), reference coordinates (lambda0, lambda1) of the +x element" % nq,
    }
    return out


def prepare(directory, h_nm, h_fine_nm):
    import ngsolve as ng

    directory.mkdir(parents=True, exist_ok=True)
    volume, arrays = directory / "mesh.vol", directory / "mesh.npz"
    construct(directory, face_specs(h_nm, h_fine_nm), h_nm)
    mesh = ng.Mesh(str(volume))
    fes = ng.H1(mesh, order=ORDER, complex=True)
    tab = fem_p4.P4Setup(mesh, fes)
    tab.material = np.array([FACE_BRANCH[m] for m in tab.mat_names], dtype=np.int64)
    tab.source_edges = source_edges(tab, mesh, fes)
    tab.boundary = outer_x_boundary_dofs(mesh, fes, tab.P)[0]
    return tab, volume, arrays
