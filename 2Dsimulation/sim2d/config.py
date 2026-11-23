"""Device geometry, physical parameters and result folder names."""

import copy
import math

import numpy as np

from . import materials, storage

ROUNDING = (
    1e-12  # relative tolerance of derived positions (mdm_geometry ROUNDING, times the computational span)
)
CASES = ("left", "right", "counter", "all")
RIM_SEGMENTS = 12  # straight segments of one rounded slit rim (a quarter circle), lateral_nm.slit_rim_radius


def rim_profile(R, n=RIM_SEGMENTS):
    """One rounded slit rim as n straight segments (the mesh has straight triangles): the points (u, v) in nm from the
    film surface to the slit wall, u the distance from the nominal wall into the Ag, v the distance from the film
    surface into the film. The rim is the quarter circle of radius R about (R, R), tangent to the film surface at
    (R, 0) and to the wall at (0, R); these two end points are exact. The n - 1 inner points lie at the angles
    k pi/(2 n) on the radius rho, slightly above R, for which the polygon (centre, points) has the area pi R^2/4 of
    the quarter disk: (n - 2) rho^2 + 2 R rho = pi R^2/(2 sin(pi/(2 n))). The polygon is convex."""
    dphi = 0.5 * math.pi / n
    rho = R * (math.sqrt(1.0 + (n - 2) * 0.5 * math.pi / math.sin(dphi)) - 1.0) / (n - 2)
    pts = [(R, 0.0)]
    pts += [(R - rho * math.sin(k * dphi), R - rho * math.cos(k * dphi)) for k in range(1, n)]
    pts.append((0.0, R))
    return pts


def layout(p):
    """All positions in the C1 frame (nm) from the parameters (c1_full_mesh.layout with apply_lines, parameterized):
    slit banks, taper starts/ends, mirror plane, physical window, PML outer box, layer interfaces, launch sheet,
    control volume, skins, observation line, projection cuts and launch foci; with lateral_nm.slit_rim_radius also the
    rim radius and the rim profile (rim_profile)."""
    ly, la, dm, ln = p["layers_nm"], p["lateral_nm"], p["domain_nm"], p["lines_nm"]
    pml = p["pml"]["thickness_nm"]
    P, w, n = la["period"], la["slit_width"], int(la["n_slits"])
    L = {"slits_l": [(j * P, j * P + w) for j in range(n)]}
    L["xg0"], L["xg1"] = L["slits_l"][0][0], L["slits_l"][-1][1]
    L["xts_l"] = L["xg1"] + la["margin"]  # taper start
    L["xte_l"] = L["xts_l"] + la["taper"]  # taper end (s = 0)
    L["xte_r"] = L["xte_l"] + la["taper_end_separation"]  # right taper end (s = D)
    L["xm"] = 0.5 * (L["xte_l"] + L["xte_r"])  # mirror plane

    def mir(x):
        return 2.0 * L["xm"] - x

    L["slits_r"] = [(mir(b), mir(a)) for a, b in reversed(L["slits_l"])]
    L["xts_r"], L["xg1_r"], L["xg0_r"] = mir(L["xts_l"]), mir(L["xg1"]), mir(L["xg0"])
    L["xmin"] = L["xg0"] - la["outer_margin"]
    L["xmax"] = mir(L["xmin"])
    L["X0"], L["X1"] = L["xmin"] - pml, L["xmax"] + pml
    L["t_ag_bot"] = ly["tm"]  # lower Ag
    L["t_ag_top"] = ly["tm"]  # central upper Ag
    L["t_ag_thick"] = ly["tm"] + ly["plateau_excess"]  # plateau upper Ag
    L["t_cr"] = ly["cr"] if ly["use_cr"] else 0.0
    L["use_cr"] = bool(ly["use_cr"])
    L["z_gt"] = -L["t_ag_top"]  # gap top
    L["z_gb"] = L["z_gt"] - ly["td"]  # gap bottom
    L["zg"] = L["z_gb"] - L["t_ag_bot"]  # stack bottom (glass/Ag interface)
    L["z_top_c"] = L["z_gt"] + L["t_ag_top"]  # central upper-Ag top (= 0)
    L["z_pl"] = L["z_gt"] + L["t_ag_thick"]  # plateau Ag top
    L["z_cr"] = L["z_pl"] + L["t_cr"]  # Cr top (= z_pl without Cr)
    L["zmin"], L["zmax"] = L["zg"] - dm["glass_depth"], dm["water_height"]
    L["Z0"], L["Z1"] = L["zmin"] - pml, L["zmax"] + pml
    L["z_src"] = L["zg"] - ln["launch_sheet_below_stack"]  # glass launch sheet
    L["z_cvb"] = L["z_src"] + 0.5 * ln["launch_sheet_below_stack"]
    L["z_cvt"] = float(ln["cv_top"])  # settled control-volume top
    L["z_ref"] = float(ln["reference_plane"])  # settled reference plane z_T
    L["cv_x"] = [L["xmin"] + ln["cv_side_inset"], L["xmax"] - ln["cv_side_inset"]]
    L["z_skin_g"], L["z_skin_w"] = L["zg"] - ln["skin"], L["z_top_c"] + ln["skin"]
    L["z_em"] = L["z_top_c"] + ln["observation_height"]  # observation line z0
    L["z_gap_centre"] = L["z_gt"] - 0.5 * ly["td"]
    L["x_s0"] = L["xte_l"]  # s origin: left taper end
    L["cuts_f_l"] = [L["xte_l"] + c for c in ln["forward_cut_offsets"]]
    L["cuts_f_r"] = [mir(c) for c in L["cuts_f_l"]]
    L["cuts_b"] = [L["xg0"] - ln["backward_cut_offset"], mir(L["xg0"] - ln["backward_cut_offset"])]
    L["foci"] = [0.5 * (L["xg0"] + L["xg1"]), mir(0.5 * (L["xg0"] + L["xg1"]))]
    L["ref_window"] = [L["xmin"], L["xmax"]]
    L["span"] = L["X1"] - L["X0"]
    if la.get("slit_rim_radius") is not None:  # rounded slit rims (optional; absent: sharp rims, no entry)
        L["slit_rim_radius"] = float(la["slit_rim_radius"])
        L["slit_rim_profile"] = rim_profile(L["slit_rim_radius"])
    return L


def surface(L, x):
    """Upper-Ag top: plateau, linear taper, central film, mirrored (c1_full_mesh.surface, vectorized)."""
    return np.interp(
        x,
        [L["X0"], L["xts_l"], L["xte_l"], L["xte_r"], L["xts_r"], L["X1"]],
        [L["z_pl"], L["z_pl"], L["z_top_c"], L["z_top_c"], L["z_pl"], L["z_pl"]],
    )


def stacks(L):
    """Bottom-up layer lists (material, bottom, top in C1 nm) of the central and plateau background stacks, their
    interfaces and region materials (glass region 0 ... water region last)."""
    m = materials
    central = ((m.AG, L["zg"], L["z_gb"]), (m.GAP, L["z_gb"], L["z_gt"]), (m.AG, L["z_gt"], L["z_top_c"]))
    plateau = [(m.AG, L["zg"], L["z_gb"]), (m.GAP, L["z_gb"], L["z_gt"]), (m.AG, L["z_gt"], L["z_pl"])]
    if L["use_cr"]:
        plateau.append((m.CR, L["z_pl"], L["z_cr"]))
    out = {}
    for name, lay in (("central", central), ("plateau", tuple(plateau))):
        out[name] = {
            "layers": lay,
            "z_top_nm": lay[-1][2],
            "interfaces_nm": tuple(lo for _, lo, _ in lay) + (lay[-1][2],),
            "region_mat": (m.GLASS,) + tuple(mat for mat, _, _ in lay) + (m.WATER,),
        }
    return out


def read_parameters(path):
    """Load the parameter block from a packaged input or a saved run configuration."""
    parameters = storage.read_json(storage.ROOT / path)["parameters"]
    coefficients = parameters["launch"]["coefficients"]
    for side, value in coefficients.items():
        if isinstance(value, dict):
            value = (value["re"], value["im"])
        coefficients[side] = complex(*value)
    return parameters


def resolve(params):
    """The resolved configuration: parameters, frequency, materials, layout, stacks, sources, sampling rules and the
    the PML."""
    p = copy.deepcopy(params)
    f = materials.frequency(p["physics"]["energy_eV"])
    eps, eps_prov = materials.material_eps(p["physics"], storage.DATA)
    L = layout(p)
    st = stacks(L)
    ew = float(eps[materials.WATER].real)
    ref = dict(p["reference"][p["reference"]["preset"]])
    alpha = math.radians(ref["alpha_deg"])
    kw = f["k0_per_um"] * math.sqrt(ew)
    carrier = {"k_a_per_um": kw * math.sin(alpha), "q_a_per_um": kw * math.cos(alpha), "k_w_per_um": kw}
    D_um = p["lateral_nm"]["taper_end_separation"] / 1000.0
    window = {
        "s_a_um": (L["xmin"] - L["xte_l"]) / 1000.0,
        "s_b_um": (L["xmax"] - L["xte_l"]) / 1000.0,
        "L_um": D_um,
    }
    launch = dict(p["launch"])
    launch.update(
        {
            "focus_left_nm": (L["foci"][0], L["zg"]),
            "focus_right_nm": (L["foci"][1], L["zg"]),
            "sheet_z_nm": L["z_src"],
            "D_nm": L["zg"] - L["z_src"],
            "aperture_x_nm": [L["xmin"], L["xmax"]],
            "medium": "glass",
            "band": "full propagating glass band |k_m| < k_g (band fraction 1.0)",
        }
    )
    cfg = {
        "parameters": p,
        "frequency": f,
        "eps": eps,
        "eps_provenance": eps_prov,
        "layout": L,
        "stacks": st,
        "carrier": carrier,
        "window": window,
        "launch": launch,
        "reference": {
            "preset": p["reference"]["preset"],
            **ref,
            "x_origin_nm": L["xte_l"],
            "z_T_nm": L["z_ref"],
            "z_p_nm": 0.0,
            "medium": "water",
            "band": "full propagating water band -k_w < k_x < k_w (theta parametrization)",
        },
        "P_V2_per_m2": float(p["normalization"]["P_V2_per_m2"]),
        "four_P_V2_per_m2": 4.0 * float(p["normalization"]["P_V2_per_m2"]),
        "D_um": D_um,
        "pml": {
            "thickness_nm": float(p["pml"]["thickness_nm"]),
            "strength": float(p["pml"]["strength"]),
            "physical_x_nm": [L["xmin"], L["xmax"]],
            "physical_z_nm": [L["zmin"], L["zmax"]],
            "outer_x_nm": [L["X0"], L["X1"]],
            "outer_z_nm": [L["Z0"], L["Z1"]],
            "profile": "s = 1 + i strength (d/thickness)^2 on all four sides",
        },
        "control_volume_nm": {"x": list(L["cv_x"]), "z": [L["z_cvb"], L["z_cvt"]]},
    }
    physics, post, post_tokens = output_parts(cfg)
    cfg["output"] = {
        "physics_parts": physics,
        "parts": physics + post,
        "products": product_folders(post_tokens),
    }
    return cfg


def mesh_inputs(cfg):
    """Geometry and spatial discretization parameters for the saved mesh."""
    p = cfg["parameters"]
    return storage.to_jsonable(
        {
            "geometry": {k: p[k] for k in ("layers_nm", "lateral_nm", "domain_nm", "lines_nm")},
            "pml_thickness_nm": p["pml"]["thickness_nm"],
            "mesh": {k: v for k, v in p["mesh"].items() if k != "asset"},
        }
    )


FOLDER_TOKENS = (
    (("physics", "eps_water"), "epsW"),
    (("physics", "n_glass"), "nGlass"),
    (("physics", "eps_gap"), "epsGap"),
    (("physics", "eps_slit"), "epsSlit"),
    (("physics", "ag_table"), "agTable"),
    (("physics", "eps_ag"), "epsAg"),
    (("physics", "eps_cr"), "epsCr"),
    (("layers_nm", "plateau_excess"), "plateauExcess"),
    (("layers_nm", "cr"), "cr"),
    (("layers_nm", "use_cr"), "useCr"),
    (("lateral_nm", "n_slits"), "nSlits"),
    (("lateral_nm", "period"), "period"),
    (("lateral_nm", "slit_width"), "slitWidth"),
    (("lateral_nm", "slit_rim_radius"), "slitRimR"),
    (("lateral_nm", "margin"), "margin"),
    (("lateral_nm", "taper"), "taper"),
    (("lateral_nm", "taper_end_separation"), "taperSep"),
    (("lateral_nm", "outer_margin"), "outerMargin"),
    (("domain_nm", "glass_depth"), "glassDepth"),
    (("domain_nm", "water_height"), "waterHeight"),
    (("lines_nm", "launch_sheet_below_stack"), "sheetBelow"),
    (("lines_nm", "cv_side_inset"), "cvInset"),
    (("lines_nm", "cv_top"), "cvTop"),
    (("lines_nm", "reference_plane"), "zT"),
    (("lines_nm", "observation_height"), "z0"),
    (("lines_nm", "forward_cut_offsets"), "cutsF"),
    (("lines_nm", "backward_cut_offset"), "cutB"),
    (("lines_nm", "skin"), "lineSkin"),
    (("mesh", "h_free"), "hFree"),
    (("mesh", "h_metal"), "hMetal"),
    (("mesh", "h_gap"), "hGap"),
    (("mesh", "h_slit"), "hSlit"),
    (("mesh", "h_skin"), "hSkin"),
    (("mesh", "skin"), "meshSkin"),
    (("mesh", "grading"), "grading"),
    (("mesh", "order"), "order"),
    (("mesh", "asset"), "meshAsset"),
    (("launch", "H0_A_per_m"), "launchH0"),
    (("launch", "intensity_fwhm_nm"), "fwhm"),
    (("launch", "dx_nm"), "launchDx"),
    (("launch", "N"), "launchN"),
    (("launch", "check_dx_nm"), "checkDx"),
    (("launch", "check_N"), "checkN"),
    (("launch", "coefficients", "left"), "coefL"),
    (("launch", "coefficients", "right"), "coefR"),
    (("reference", "<preset>", "H_in_A_per_m"), "Hin"),
    (("reference", "<preset>", "gamma_in_per_um"), "gamma"),
    (("reference", "<preset>", "quadrature"), "quad"),
    (("reference", "<preset>", "H0_A_per_m"), "refH0"),
    (("reference", "<preset>", "amplitude"), "amplitude"),
    (("normalization", "P_V2_per_m2"), "P"),
    (("phases", "phi_T_rad"), "phiT"),
    (("phases", "phi_map_rad"), "phiMap"),
    (("sampling", "line_points"), "linePoints"),
    (("sampling", "water_map"), "waterMap"),
    (("sampling", "excitation_map"), "excMap"),
    (("sampling", "stack_map"), "stackMap"),
    (("display", "interference_windows_um"), "windows"),
    (("display", "interference_map_limits"), "mapLimits"),
    (("display", "model_family", "<preset>"), "family"),
)
# Tokens that change only the postprocessing (source coefficients of the combinations, phases, sampling, display): they
# name a level below the physics folder, which keeps the mesh, sources and FEM solutions of the physics.
POST_TOKENS = (
    "coefL",
    "coefR",
    "phiT",
    "phiMap",
    "linePoints",
    "waterMap",
    "excMap",
    "stackMap",
    "windows",
    "mapLimits",
    "family",
)
LINE, MAPS = ("linePoints",), ("waterMap", "excMap", "stackMap")
PRODUCTS = {
    "line_sampling": ("sampling", LINE),
    "map_sampling": ("sampling", MAPS),
    "line_fields": ("fields", LINE),
    "map_fields": ("fields", MAPS),
    "interference": ("interference", LINE + ("coefL", "coefR", "phiT")),
    "reference_line": ("plot_data", LINE),
    "plot_data": ("plot_data", LINE + MAPS + ("coefL", "coefR", "phiT", "phiMap", "family")),
    "figure_data": ("plot_data/figure_data", POST_TOKENS),
    "view_data": ("plot_data/figure_data", ()),
}


def product_folders(post_tokens):
    """The physics/ subfolder of every product category (PRODUCTS) for the differing postprocessing tokens [(short,
    token text)]: its folder plus one level of the differing tokens it depends on, in folder-token order."""
    out = {}
    for cat, (folder, deps) in PRODUCTS.items():
        level = "_".join(text for short, text in post_tokens if short in deps)
        out[cat] = folder + ("/" + level if level else "")
    return out


def name_value(v):
    """Folder-name text of a value (the TDSE format_sig/safe_str substitutions, up to 10 significant digits): '.' ->
    'p', '-' -> 'm', '+' -> 'P'; complex <re>P<im>i or <re>m<im>i; booleans 1/0; None 'none'; strings their letters
    and digits; lists and dicts their values joined by 'x'."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, str):
        return "".join(ch for ch in v if ch.isalnum())
    if isinstance(v, dict):
        return "x".join(name_value(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return "x".join(name_value(x) for x in v)
    if isinstance(v, complex):
        return name_value(v.real) + ("m" if v.imag < 0 else "P") + name_value(abs(v.imag)) + "i"
    return ("%.10g" % v).replace("-", "m").replace("+", "P").replace(".", "p")


_MISSING = object()


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def output_parts(cfg):
    """The folder levels of the configuration. Physics folder: physics and stack (E, td, tm), numerics (hScale, PML
    thickness and strength), illumination (reference preset and angle) and, when any other physics parameter differs
    from the package defaults (data/default_parameters.json), one level of <token><value> pieces joined by '_'. Below
    it, one more level of the same form when a postprocessing parameter (POST_TOKENS) differs. Returns (physics, post, the differing
    postprocessing tokens [(short, token text)])."""
    p = cfg["parameters"]
    preset = p["reference"]["preset"]
    physics = [
        "E%s_td%s_tm%s"
        % (
            name_value(p["physics"]["energy_eV"]),
            name_value(p["layers_nm"]["td"]),
            name_value(p["layers_nm"]["tm"]),
        ),
        "hScale%s_PML%ss%s"
        % (
            name_value(p["mesh"]["h_scale"]),
            name_value(p["pml"]["thickness_nm"]),
            name_value(p["pml"]["strength"]),
        ),
        "%s_a%s" % (preset, name_value(p["reference"][preset]["alpha_deg"])),
    ]
    default = storage.read_json(storage.DATA / "default_parameters.json")["parameters"]
    tokens = {"physics": [], "post": []}
    for path, short in FOLDER_TOKENS:
        path = tuple(preset if k == "<preset>" else k for k in path)
        cur, ref = _get(p, path), _get(default, path)
        if cur is not _MISSING and (ref is _MISSING or storage.canonical(cur) != storage.canonical(ref)):
            tokens["post" if short in POST_TOKENS else "physics"].append((short, short + name_value(cur)))
    if tokens["physics"]:
        physics.append("_".join(t for _, t in tokens["physics"]))
    post = ["_".join(t for _, t in tokens["post"])] if tokens["post"] else []
    return physics, post, tokens["post"]


def folder_record(cfg):
    """The content of a configuration folder's configuration.json: its folder levels and the parameters that determine
    them (the active reference preset and its comparison family only)."""
    p = copy.deepcopy(cfg["parameters"])
    preset = p["reference"]["preset"]
    p["reference"] = {"preset": preset, preset: p["reference"][preset]}
    p["display"]["model_family"] = {preset: p["display"]["model_family"][preset]}
    return storage.to_jsonable({"output_parts": cfg["output"]["parts"], "parameters": p})
