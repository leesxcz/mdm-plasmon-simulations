"""Material regions, surface polygons and internal source planes."""

import numpy as np

from . import materials as mt
from .config import mesh_inputs, surface

COLORS = {
    mt.WATER: "#d6ecfa",
    mt.AG: "#a8a8a8",
    mt.GAP: "#f2a93b",
    mt.GLASS: "#e4def3",
    mt.CR: "#3f4b63",
    mt.SLIT: "#f7d48a",
}
FACE_CODE = mt.FACE_NAMES
SLIT_GLASS_TOL = (
    1e-12  # relative agreement of the resolved slit and glass permittivities for the glass slit colour
)


def slit_is_glass(cfg):
    """True when the resolved slit permittivity equals the glass permittivity (relative SLIT_GLASS_TOL): the slits are
    filled with the substrate glass and the views draw and label them as glass."""
    es, eg = complex(cfg["eps"][mt.SLIT]), complex(cfg["eps"][mt.GLASS])
    return abs(es - eg) <= SLIT_GLASS_TOL * abs(eg)


def palette(cfg):
    """The material colours of the configuration's views: a copy of COLORS whose slit colour is the glass colour when
    slit_is_glass(cfg); COLORS itself is unchanged."""
    out = dict(COLORS)
    if slit_is_glass(cfg):
        out[mt.SLIT] = COLORS[mt.GLASS]
    return out


def _dedupe(pts):
    out = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) > 1e-12 or abs(p[1] - out[-1][1]) > 1e-12):
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= 1e-12 and abs(out[0][1] - out[-1][1]) <= 1e-12:
        out.pop()
    return out


def slit_wall(L, x, side):
    """The boundary between a slit and the lower Ag at the nominal wall position x (C1 nm), as points from the glass
    surface z = zg up to the gap surface z = z_gb; side = -1: the Ag lies on the left of the wall (a slit start), +1: on
    the right (a slit end). Sharp rims: the straight wall. Rounded rims (layout slit_rim_profile, config.rim_profile):
    the lower rim from the glass surface to the wall, the straight wall, the upper rim to the gap surface. The slit
    face, the Ag face beside it, classify and the mesh use the same rim points."""
    zg, zb = L["zg"], L["z_gb"]
    prof = L.get("slit_rim_profile")
    if not prof:
        return [(x, zg), (x, zb)]
    lower = [(x + side * u, zg + v) for u, v in prof]
    upper = [(x + side * u, zb - v) for u, v in reversed(prof)]
    if (
        abs(upper[0][1] - lower[-1][1]) <= 1e-12
    ):  # 2 R equals the film thickness: the two rims meet on the wall
        upper = upper[1:]
    return lower + upper


def rim_area(L):
    """Area (nm^2) of one rounded slit rim, the Ag removed between the wall, the film surface and the rim profile
    (R^2 minus the polygon of the profile and its centre (R, R)); 0.0 for sharp rims."""
    prof = L.get("slit_rim_profile")
    if not prof:
        return 0.0
    R = float(L["slit_rim_radius"])
    return R * R - face_area([(R, R)] + [tuple(q) for q in prof])


def face_specs(L, mesh):
    """Material faces (C1 nm) in construction order: dicts name/tag/maxh/points/top/bottom, plus the list of added
    background-partition faces. maxh values are the regional sizes times h_scale (None: the global free size). With
    rounded slit rims the slit faces and the Ag faces beside them are bounded by slit_wall (the same points on both
    sides of a wall)."""
    hs = float(mesh["h_scale"])
    h = {k: float(mesh[k]) * hs for k in ("h_free", "h_metal", "h_gap", "h_slit", "h_skin")}
    eps = 1e-9 * L["span"]
    faces, added = [], []

    def face(pts, name, maxh, tag, top=None, bottom=None):
        faces.append(
            {
                "name": name,
                "tag": tag,
                "maxh": maxh,
                "points": [tuple(map(float, q)) for q in pts],
                "top": top,
                "bottom": bottom,
            }
        )

    breaks = [L["xts_l"], L["xte_l"], L["xte_r"], L["xts_r"]]
    cuts = L["cuts_f_l"] + L["cuts_f_r"] + L["cuts_b"]

    def xsplits(z0, z1, extra=()):
        xs = {L["xmin"], L["xmax"]} | set(extra)
        if z0 >= L["zmin"] - eps and z1 <= L["zmax"] + eps:
            xs |= set(cuts)
        if z0 >= L["z_cvb"] - eps and z1 <= L["z_cvt"] + eps:
            xs |= set(L["cv_x"])
        return [L["X0"]] + sorted(x for x in xs if L["X0"] + eps < x < L["X1"] - eps) + [L["X1"]]

    def phys(a, b):
        return a >= L["xmin"] - eps and b <= L["xmax"] + eps

    def flat(z0, z1, name, maxh, tag, top=None, bottom=None):
        xs = xsplits(z0, z1)
        for a, b in zip(xs[:-1], xs[1:]):
            face(
                [(a, z0), (b, z0), (b, z1), (a, z1)],
                name,
                maxh,
                tag,
                top if top and phys(a, b) else None,
                bottom if bottom and phys(a, b) else None,
            )

    def levels(*zs):
        return sorted(set(float(z) for z in zs))

    # glass, with the launch sheet named 'source' on the physical x window
    flat(L["Z0"], L["zmin"], "glass", None, "glass PML")
    flat(L["zmin"], L["z_src"], "glass", None, "glass bulk", top="source")
    flat(L["z_src"], L["z_cvb"], "glass", None, "glass bulk", bottom="source")
    flat(L["z_cvb"], L["z_skin_g"], "glass", None, "glass bulk")
    flat(L["z_skin_g"], L["zg"], "glass", h["h_skin"], "glass skin")
    # lower Ag minus the slits of both banks, and the slits
    walls = [v for s in L["slits_l"] + L["slits_r"] for v in s]
    starts = {s[0] for s in L["slits_l"] + L["slits_r"]}
    ends = {s[1] for s in L["slits_l"] + L["slits_r"]}
    rim = L.get("slit_rim_radius")
    xs = xsplits(L["zg"], L["z_gb"], walls)
    for a, b in zip(xs[:-1], xs[1:]):
        m = 0.5 * (a + b)
        slit = any(s0 < m < s1 for s0, s1 in L["slits_l"] + L["slits_r"])
        pts = [(a, L["zg"]), (b, L["zg"]), (b, L["z_gb"]), (a, L["z_gb"])]
        if (
            rim
        ):  # rounded rims: the sides at slit walls are the walls with their rims (bottom to top, slit_wall)
            left = (
                slit_wall(L, a, -1 if a in starts else 1) if a in starts or a in ends else pts[:1] + pts[3:]
            )
            right = slit_wall(L, b, 1 if b in ends else -1) if b in starts or b in ends else pts[1:3]
            pts = left[:1] + right + left[:0:-1]
        face(
            pts,
            "slit" if slit else "ag",
            h["h_slit"] if slit else h["h_metal"],
            "slit" if slit else "lower Ag",
        )
    # dielectric gap
    flat(L["z_gb"], L["z_gt"], "diel", h["h_gap"], "gap")
    # upper Ag: plateaus, linear tapers and central film, mirrored
    xs = xsplits(L["z_gt"], L["z_pl"], breaks)
    for a, b in zip(xs[:-1], xs[1:]):
        face(
            [(a, L["z_gt"]), (b, L["z_gt"]), (b, float(surface(L, b))), (a, float(surface(L, a)))],
            "ag",
            h["h_metal"],
            "upper Ag",
        )
    # Cr caps over the plateaus, ending at the taper starts (configured layer list: none without Cr)
    if L["use_cr"]:
        xs = xsplits(L["z_pl"], L["z_cr"], breaks)
        for a, b in zip(xs[:-1], xs[1:]):
            m = 0.5 * (a + b)
            if m < L["xts_l"] or m > L["xts_r"]:
                face(
                    [(a, L["z_pl"]), (b, L["z_pl"]), (b, L["z_cr"]), (a, L["z_cr"])], "cr", h["h_metal"], "Cr"
                )
    # water skin band up to z = skin minus the solids; over the tapers and the taper-end interval the water faces are
    # split at the plateau-background interfaces (z_pl, z_cr) and, over the interval, at the observation line
    xs = xsplits(L["z_top_c"], L["z_skin_w"], breaks)
    for a, b in zip(xs[:-1], xs[1:]):
        m = 0.5 * (a + b)
        if m < L["xts_l"] or m > L["xts_r"]:
            face(
                [(a, L["z_cr"]), (b, L["z_cr"]), (b, L["z_skin_w"]), (a, L["z_skin_w"])],
                "water",
                h["h_skin"],
                "water skin",
            )
        elif L["xte_l"] < m < L["xte_r"]:
            lv = levels(L["z_top_c"], L["z_em"], L["z_pl"], L["z_cr"], L["z_skin_w"])
            for z0, z1 in zip(lv[:-1], lv[1:]):
                face([(a, z0), (b, z0), (b, z1), (a, z1)], "water", h["h_skin"], "water skin")
            added += [("taper-end interval", a, b, z) for z in levels(L["z_pl"], L["z_cr"])]
        else:
            face(
                _dedupe(
                    [(a, float(surface(L, a))), (b, float(surface(L, b))), (b, L["z_pl"]), (a, L["z_pl"])]
                ),
                "water",
                h["h_skin"],
                "water skin",
            )
            lv = levels(L["z_pl"], L["z_cr"], L["z_skin_w"])
            for z0, z1 in zip(lv[:-1], lv[1:]):
                face([(a, z0), (b, z0), (b, z1), (a, z1)], "water", h["h_skin"], "water skin")
            added += [("taper", a, b, z) for z in levels(L["z_pl"], L["z_cr"])]
    # water bulk: control-volume top, then the reference plane named 'ref_source' on the physical x window
    flat(L["z_skin_w"], L["z_cvt"], "water", None, "water bulk")
    flat(L["z_cvt"], L["z_ref"], "water", None, "water bulk", top="ref_source")
    flat(L["z_ref"], L["zmax"], "water", None, "water bulk", bottom="ref_source")
    flat(L["zmax"], L["Z1"], "water", None, "water PML")
    return faces, added


UNITS = "C1 frame, nm (x = 0 at the outer wall of the left bank's first slit, z = 0 at the central upper-Ag surface)"


def description(cfg):
    """The geometry description of the configuration (JSON-ready): the construction inputs of the mesh
    (config.mesh_inputs), the resolved C1 coordinates, the background stacks with their interfaces, the material faces
    of face_specs (polygons, material, tag, regional maxh, named top/bottom edges) with the added background-partition
    faces, the declared lines, the named source sheets on the physical window, the PML boxes and the effective regional
    mesh sizes (h times h_scale), grading and order."""
    from . import storage

    L, mp = cfg["layout"], cfg["parameters"]["mesh"]
    faces, added = face_specs(L, mp)
    hs = float(mp["h_scale"])
    return storage.to_jsonable(
        {
            "units": UNITS,
            "construction_inputs": mesh_inputs(cfg),
            "layout_C1_nm": L,
            "stacks": cfg["stacks"],
            "faces": [dict(f, material=mt.NAMES[FACE_CODE[f["name"]]]) for f in faces],
            "background_partitions": [list(a) for a in added],
            "lines": [
                {"name": n, "kind": k, "from_nm": [x0, z0], "to_nm": [x1, z1]}
                for n, k, x0, z0, x1, z1 in declared_lines(L)
            ],
            "named_sheets": {
                "source": {"z_nm": L["z_src"], "x_nm": [L["xmin"], L["xmax"]]},
                "ref_source": {"z_nm": L["z_ref"], "x_nm": [L["xmin"], L["xmax"]]},
            },
            "pml": cfg["pml"],
            "mesh_sizes_nm": {
                k: float(mp[k]) * hs for k in ("h_free", "h_metal", "h_gap", "h_slit", "h_skin")
            },
            "mesh_settings": {
                "skin_nm": float(mp["skin"]),
                "grading": float(mp["grading"]),
                "order": mp["order"],
            },
        }
    )


def classify(L, x, z):
    """Exact analytic material code of points strictly inside regions (configured Cr; with rounded slit rims the rim
    regions of slit_wall belong to the slits)."""
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    m = np.full(np.shape(x), mt.WATER, dtype=np.int64)
    m[z < L["zg"]] = mt.GLASS
    lower = (z > L["zg"]) & (z < L["z_gb"])
    slit = np.zeros(np.shape(x), dtype=bool)
    for a, b in L["slits_l"] + L["slits_r"]:
        slit |= (x > a) & (x < b)
    m[lower] = mt.AG
    m[lower & slit] = mt.SLIT
    rim = L.get("slit_rim_radius")
    if (
        rim
    ):  # rounded rims: beside a wall and a film surface, outside the Ag polygon of the rim profile and its centre
        u = np.full(
            np.shape(x), np.inf
        )  # distance from the nearest slit wall into the Ag (negative inside a slit)
        for a, b in L["slits_l"] + L["slits_r"]:
            u = np.minimum(u, np.maximum(a - x, x - b))
        v = np.minimum(z - L["zg"], L["z_gb"] - z)  # distance from the nearer film surface into the film
        near = lower & (u > 0.0) & (u < rim) & (v < rim)
        if np.any(near):
            un, vn = u[near], v[near]
            ag = np.ones(un.shape, dtype=bool)
            prof = np.asarray(L["slit_rim_profile"], dtype=np.float64)
            for (u0, v0), (u1, v1) in zip(prof[:-1], prof[1:]):  # on the centre's side of every segment: Ag
                centre = (u1 - u0) * (rim - v0) - (v1 - v0) * (rim - u0)
                ag &= ((u1 - u0) * (vn - v0) - (v1 - v0) * (un - u0)) * centre > 0.0
            near[near] = ~ag
            m[near] = mt.SLIT
    m[(z > L["z_gb"]) & (z < L["z_gt"])] = mt.GAP
    s = surface(L, x)
    m[(z > L["z_gt"]) & (z < s)] = mt.AG
    if L["use_cr"]:
        plateau = (x < L["xts_l"]) | (x > L["xts_r"])
        m[plateau & (z > s) & (z < s + L["t_cr"])] = mt.CR
    return m


def declared_lines(L):
    """Declared lines (name, kind, x0, z0, x1, z1) in the C1 frame (c1_full_mesh.declared_lines)."""
    out = [
        ("left PML start", "PML start", L["xmin"], L["Z0"], L["xmin"], L["Z1"]),
        ("right PML start", "PML start", L["xmax"], L["Z0"], L["xmax"], L["Z1"]),
        ("bottom PML start", "PML start", L["X0"], L["zmin"], L["X1"], L["zmin"]),
        ("top PML start", "PML start", L["X0"], L["zmax"], L["X1"], L["zmax"]),
        ("glass launch sheet", "launch sheet", L["xmin"], L["z_src"], L["xmax"], L["z_src"]),
        ("reference plane", "reference plane", L["xmin"], L["z_ref"], L["xmax"], L["z_ref"]),
        ("control-volume bottom", "control volume", L["cv_x"][0], L["z_cvb"], L["cv_x"][1], L["z_cvb"]),
        ("control-volume top", "control volume", L["cv_x"][0], L["z_cvt"], L["cv_x"][1], L["z_cvt"]),
        ("control-volume left face", "control volume", L["cv_x"][0], L["z_cvb"], L["cv_x"][0], L["z_cvt"]),
        ("control-volume right face", "control volume", L["cv_x"][1], L["z_cvb"], L["cv_x"][1], L["z_cvt"]),
        ("observation line", "observation line", L["xte_l"], L["z_em"], L["xte_r"], L["z_em"]),
    ]
    for lab, cuts in (
        ("left-launch forward cut", L["cuts_f_l"]),
        ("right-launch forward cut", L["cuts_f_r"]),
        ("backward cut", L["cuts_b"]),
    ):
        for c in cuts:
            out.append(("%s x_C1 = %g nm" % (lab, c), "projection cut", c, L["zmin"], c, L["zmax"]))
    return out


def face_area(pts):
    P = np.asarray(pts, dtype=np.float64)
    x, z = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))
