"""Sample the native P4 launch and scattered reference; add the stratified background and form the phase frames."""

import math
from pathlib import Path
import numpy as np
from . import fem_p4, geometry, layered, modes

MAP_SETS = ("stack", "excitation", "water")


def e_fields(vals, eps, om):
    """E_x, E_z, d_x E_x, d_x E_z and |E|^2 from H and its derivatives (SI) with the actual permittivity eps."""
    den = om * eps
    Ex, Ez = -1j * vals["Hz"] / den, 1j * vals["Hx"] / den
    return {
        "Ex": Ex,
        "Ez": Ez,
        "dxEx": -1j * vals["Hxz"] / den,
        "dxEz": 1j * vals["Hxx"] / den,
        "F": np.abs(Ex) ** 2 + np.abs(Ez) ** 2,
        "q": vals["Hz"] / eps,
    }


def detector_line(cfg):
    """The detector line: line_points samples at z0 from the left to the right taper end (s = 0..D)."""
    L = cfg["layout"]
    n = int(cfg["parameters"]["sampling"]["line_points"])
    x = L["xte_l"] + (L["xte_r"] - L["xte_l"]) * np.arange(n, dtype=np.float64) / (n - 1)
    return x, np.full(n, float(L["z_em"]))


def case_parts(cfg, case):
    """The launch combination G of the case: (label, coefficient) pairs, the configured source coefficients applied
    exactly once (left: a_L L; right: a_R R; counter/all: a_L L + a_R R)."""
    co = cfg["launch"]["coefficients"]
    if case == "left":
        return [("L", complex(co["left"]))]
    if case == "right":
        return [("R", complex(co["right"]))]
    return [("L", complex(co["left"])), ("R", complex(co["right"]))]


def map_grids(cfg):
    """Sample coordinates (C1 nm) of the selected maps, columns x and rows z. stack: the cell centres of the C stack
    window (the 'zoom' rule), x = x_taper_start + dx/2 + k dx < x_right_taper_end, z = -h +
    dz/2 + k dz < h. excitation: the control-volume map rule ('maps'), x = x_cv0 + 1 + k dx < x_cv1, z = z_cvb + 1 +
    k dz < z_cvt, restricted to the columns whose cells overlap the taper-end interval. water: the same rule and column
    crop with the water-map spacing, rows strictly inside the water band 0..height above the central Ag surface.
    """
    L, smp = cfg["layout"], cfg["parameters"]["sampling"]
    st = smp["stack_map"]
    grids = {
        "stack": (
            np.arange(L["xts_l"] + 0.5 * st["dx_nm"], L["xte_r"], st["dx_nm"]),
            np.arange(-st["half_height_nm"] + 0.5 * st["dz_nm"], st["half_height_nm"], st["dz_nm"]),
        )
    }
    for name in ("excitation", "water"):
        rule = smp[name + "_map"]
        dx, dz = float(rule["dx_nm"]), float(rule["dz_nm"])
        x = np.arange(L["cv_x"][0] + 1.0, L["cv_x"][1], dx)
        x = x[(x + 0.5 * dx >= L["xte_l"]) & (x - 0.5 * dx <= L["xte_r"])]
        z = np.arange(L["z_cvb"] + 1.0, L["z_cvt"], dz)
        if name == "water":
            z = z[(z > L["z_top_c"]) & (z < L["z_top_c"] + rule["height_nm"])]
        grids[name] = (x, z)
    return grids


def model_family(cfg):
    """The analytical comparison family of the configured reference preset (display.model_family): 'ideal' or
    'planar'."""
    return cfg["parameters"]["display"]["model_family"][cfg["reference"]["preset"]]


def planar_reference_fields(cfg, beam, x, z, water_axes):
    """Evaluate the planar central stack at the detector line and, when requested, over the water map."""
    stack = layered.Stack(
        beam,
        cfg["eps"],
        cfg["stacks"]["central"],
        cfg["frequency"]["k0_per_m"],
        cfg["frequency"]["omega_eps0_S_per_m"],
    )
    try:
        water = stack.n_regions - 1
        line = stack.fields(x, z, water, keys=("H", "Hx", "Hz"))
        map_field = None
        if water_axes is not None:
            X, Z = np.meshgrid(*water_axes)
            field = stack.fields(X.ravel(), Z.ravel(), water, keys=("H", "Hx", "Hz"))
            map_field = np.stack([field["Ex"], field["Ez"]], axis=-1).reshape(*X.shape, 2)
        return line, map_field
    finally:
        stack.free_device()


def phase_intensities(EG, ET, phis):
    """Raw intensity of the coherent electric-field superposition, with the phase on the reference."""
    return np.stack([np.sum(np.abs(EG + np.exp(1j * phi) * ET) ** 2, axis=-1) for phi in phis])


def intensity_frames(EG, ET, phis, four_P):
    """Normalize all phase frames by the common physical 4P."""
    return phase_intensities(EG, ET, phis) / four_P


def save_planar_comparison(physics, plot, line, planar_line, planar_map):
    """Write the full and central-planar reference fields and their interference with the same launch."""
    phis = plot["phi_T_rad"]
    four_P = plot["four_P_V2_per_m2"]
    arrays = {
        "phi_rad": phis,
        "P_V2_per_m2": plot["P_V2_per_m2"],
        "origin__E": planar_line[0],
        "line__x_nm": plot["line__x_nm"],
        "line__z_nm": plot["line__z_nm"],
        "line__s_um": plot["line__s_um"],
        "line__H": line["H"],
        "line__Hx_per_m": line["Hx"],
        "line__Hz_per_m": line["Hz"],
        "line__J_ideal": plot["model__J"],
        "map__x_nm": plot["water__x_nm"],
        "map__z_nm": plot["water__z_nm"],
    }
    for name, prefix, planar in (("line", "line", planar_line), ("map", "water", planar_map)):
        launch, full = plot[prefix + "__E_G"], plot[prefix + "__E_T"]
        raw_full = phase_intensities(launch, full, phis)
        raw_planar = phase_intensities(launch, planar, phis)
        components = {
            "E_L": launch,
            "E_full": full,
            "E_planar": planar,
            "DeltaE": full - planar,
            "I_full": raw_full,
            "I_planar": raw_planar,
            "J_full": raw_full / four_P,
            "J_planar": raw_planar / four_P,
        }
        arrays.update({name + "__" + key: value for key, value in components.items()})
    save_arrays(physics / "plot_data/central_planar_comparison.npz", **arrays)


def model_curves(cfg, case, s_um, inputs, T_pl=None):
    """The selected analytical comparison on the observation line, J_m = |a_L L_m + a_R R_m + exp(i phi_T) T_m|^2/(4P)
    for every frame: L_m = u_+ exp(i beta s), R_m = u_- exp(i beta (D - s)), u_- = (u_+,x, -u_+,z) (reversed
    propagation reverses E_z). ideal: u_+ = sqrt(P) u/|u|, T_m = u_+ exp(-i chi) exp((-Im beta + i k_a) s), referenced
    to the left origin with the configured signed carrier in every case; planar: u_+ = u, T_m = T_pl. a_L, a_R: the
    configured source coefficients of the case's launches (zero for a launch that is off), applied once."""
    fam = model_family(cfg)
    P, four_P, D = cfg["P_V2_per_m2"], cfg["four_P_V2_per_m2"], cfg["D_um"]
    beta, ka, chi = inputs["beta_per_um"], inputs["k_a_per_um"], inputs["chi_rad"]
    phis = np.array([float(v) for v in cfg["parameters"]["phases"]["phi_T_rad"]])
    co = cfg["launch"]["coefficients"]
    aL = complex(co["left"]) if case in ("left", "counter") else 0j
    aR = complex(co["right"]) if case in ("right", "counter") else 0j
    u = inputs["u_V_per_m"]
    up = math.sqrt(P) * u / np.linalg.norm(u) if fam == "ideal" else u
    um = np.array([up[0], -up[1]])
    if fam == "ideal":
        T = up[None, :] * (np.exp(-1j * chi) * np.exp((-beta.imag + 1j * ka) * s_um))[:, None]
    else:
        T = T_pl
    G = aL * (up[None, :] * np.exp(1j * beta * s_um)[:, None]) + aR * (
        um[None, :] * np.exp(1j * beta * (D - s_um))[:, None]
    )
    J = intensity_frames(G, T, phis, four_P)
    ref0 = inputs["E_ref_planar_origin_V_per_m"]
    arrays = {
        "model__J": J,
        "model__G": G,
        "model__T": T,
        "model__u_V_per_m": u,
        "model__u_plus_V_per_m": up,
        "model__u_minus_V_per_m": um,
        "model__a": np.array([aL, aR]),
        "model__beta_per_um": np.array(beta),
        "model__k_a_per_um": np.array(ka),
        "model__chi_rad": np.array(chi),
        "model__E_ref_planar_origin_V_per_m": ref0,
        "model__family": np.array(fam),
    }
    return arrays


def plot_case(cfg, case, line, msets, mevs, inputs, T_pl):
    """The plotted states of one case from the saved samples: the observation line (E_G, E_T and J of every frame), the
    stack window (H_G, H_T = H_b + H_s and H_G + exp(i phi_map) H_T), the central excitation window (E_G, E_T and |E|^2
    of the three states) and the water window (E_G, E_T and J of every frame), E with the actual permittivity of each
    point, J = |E_G + exp(i phi_T) E_T|^2/(4P) with the fixed 4P; and the selected analytical comparison."""
    om = cfg["frequency"]["omega_eps0_S_per_m"]
    four_P = cfg["four_P_V2_per_m2"]
    ph = cfg["parameters"]["phases"]
    phis = np.array([float(v) for v in ph["phi_T_rad"]])
    phi_map = float(ph["phi_map_rad"])
    parts = case_parts(cfg, case)
    A = msets["arrays"]
    EG = np.stack([line["G__Ex"], line["G__Ez"]], axis=-1)
    ET = np.stack([line["T__Ex"], line["T__Ez"]], axis=-1)
    J = intensity_frames(EG, ET, phis, four_P)
    out = {
        "line__s_um": line["s_um"],
        "line__x_nm": line["x_nm"],
        "line__z_nm": line["z_nm"],
        "line__E_G": EG,
        "line__E_T": ET,
        "line__J": J,
        "phi_T_rad": phis,
        "phi_map_rad": np.array(phi_map),
        "P_V2_per_m2": np.array(cfg["P_V2_per_m2"]),
        "four_P_V2_per_m2": np.array(four_P),
        "D_um": np.array(cfg["D_um"]),
        "coefficients": np.array([complex(c) for _, c in parts]),
    }
    for name in MAP_SETS:
        pre = name + "__"
        nz, nx = (int(v) for v in A[pre + "shape"])
        out[pre + "x_nm"] = A[pre + "x"].reshape(nz, nx)[0]
        out[pre + "z_nm"] = A[pre + "z"].reshape(nz, nx)[:, 0]
        vG = {k: sum(c * mevs[lab]["values"][pre + k] for lab, c in parts) for k in ("H", "Hx", "Hz")}
        vT = {k: mevs["T"]["values"][pre + k] + mevs["T"]["background"][pre + k] for k in ("H", "Hx", "Hz")}
        if name == "stack":
            HG, HT = vG["H"].reshape(nz, nx), vT["H"].reshape(nz, nx)
            out.update({pre + "H_G": HG, pre + "H_T": HT, pre + "H_comb": HG + np.exp(1j * phi_map) * HT})
            continue
        den = om * cfg["eps"][A[pre + "mat"]]
        EGm = np.stack([-1j * vG["Hz"] / den, 1j * vG["Hx"] / den], axis=-1).reshape(nz, nx, 2)
        ETm = np.stack([-1j * vT["Hz"] / den, 1j * vT["Hx"] / den], axis=-1).reshape(nz, nx, 2)
        out.update({pre + "E_G": EGm, pre + "E_T": ETm, pre + "mat": A[pre + "mat"].reshape(nz, nx)})
        if name == "excitation":
            out[pre + "F"] = np.stack(
                [
                    np.sum(np.abs(EGm) ** 2, axis=-1),
                    np.sum(np.abs(ETm) ** 2, axis=-1),
                    np.sum(np.abs(EGm + np.exp(1j * phi_map) * ETm) ** 2, axis=-1),
                ]
            )
        else:
            Jw = intensity_frames(EGm, ETm, phis, four_P)
            out[pre + "J"] = Jw
    out.update(model_curves(cfg, case, line["s_um"], inputs, T_pl))
    return out


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def record_plots(cfg, case, data, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    directory.mkdir(parents=True, exist_ok=True)
    s = data["line__s_um"]
    for window in cfg["parameters"]["display"]["interference_windows_um"]:
        fig, ax = plt.subplots(figsize=(8, 4.1))
        for i, (phi, values) in enumerate(zip(data["phi_T_rad"], data["line__J"])):
            label = rf"$\phi={phi:.3g}$ rad"
            ax.plot(s, values, label=label)
            ax.plot(s, data["model__J"][i], color=ax.lines[-1].get_color(), ls="--", lw=0.9)
        limits = (0, cfg["D_um"]) if window is None else window
        ax.set(xlim=limits, xlabel="Distance from the taper end (µm)", ylabel="Intensity / (4P)")
        ax.legend(frameon=False)
        fig.tight_layout()
        suffix = f"{limits[0]:g}to{limits[1]:g}um"
        fig.savefig(directory / f"interference_{case}_{suffix}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(len(data["phi_T_rad"]), 1, sharex=True, figsize=(9, 6), squeeze=False)
    lo, hi = cfg["parameters"]["display"]["interference_map_limits"]
    for ax, phi, J in zip(axes[:, 0], data["phi_T_rad"], data["water__J"]):
        artist = ax.pcolormesh(
            (data["water__x_nm"] - cfg["layout"]["xte_l"]) * 1e-3,
            data["water__z_nm"],
            J,
            shading="nearest",
            cmap="viridis",
            norm=LogNorm(lo, hi),
        )
        ax.set_ylabel("Height (nm)")
        ax.set_title(f"Phase {phi:.4g} rad", loc="left", fontsize=10)
        fig.colorbar(artist, ax=ax, label="Intensity / (4P)")
    axes[-1, 0].set_xlabel("Distance from the taper end (µm)")
    fig.tight_layout()
    fig.savefig(directory / f"water_interference_{case}.png", dpi=180)
    plt.close(fig)


def evaluate(
    cfg, tab, locator, solutions, beam, gap, launch_record, cases, physics, directory, *, planar_map=False
):
    labels = list(solutions)
    columns = np.column_stack([solutions[label] for label in labels])
    gpu = fem_p4.GPUTri4(tab, cfg["eps"][tab.material], "output fields", assembly=False, columns=len(labels))
    freq = cfg["frequency"]
    stack = layered.Stack(
        beam, cfg["eps"], cfg["stacks"]["plateau"], freq["k0_per_m"], freq["omega_eps0_S_per_m"]
    )
    background_regions = stack.region_of_z

    def sample(x, z, mat, keys):
        points, elements = locator.locate(x, z, mat)
        location = {"el": elements, "l0": np.asarray(points["x"]), "l1": np.asarray(points["y"])}
        vals = gpu.points(elements, location["l0"], location["l1"], keys=keys)
        parts = {label: fem_p4.p4_vals(vals, i) for i, label in enumerate(labels)}
        # Use the background region of the chosen element at interfaces, as in the original field readout.
        region = background_regions(tab.P[tab.T[elements], 1].mean(axis=1))
        bg = stack.fields(x, z, region, keys=keys)
        return parts, {key: bg[key] for key in keys}, region, location

    try:
        gpu.set_solution(columns)
        x, z = detector_line(cfg)
        mat = geometry.classify(cfg["layout"], x, z)
        om, eps = freq["omega_eps0_S_per_m"], cfg["eps"][mat]
        primary, bg, _, _ = sample(x, z, mat, fem_p4.POINT_KEYS)
        total = {k: bg[k] + primary["T"][k] for k in fem_p4.POINT_KEYS}
        ref_fields = dict(total, **e_fields(total, eps, om))
        t_vec = np.stack([ref_fields[k] for k in ("Ex", "Ez")], axis=-1)
        grids = map_grids(cfg)
        plane_line, plane_map = planar_reference_fields(
            cfg, beam, x, z, grids["water"] if planar_map else None
        )
        planar = np.stack([plane_line["Ex"], plane_line["Ez"]], axis=-1)
        s = (x - cfg["layout"]["xte_l"]) * 1e-3
        phases = cfg["parameters"]["phases"]["phi_T_rad"]
        u = np.array([modes.cval(launch_record["u_SPP_origin_V_per_m"][k]) for k in ("x", "z")])
        inputs = {
            "u_V_per_m": u,
            "beta_per_um": gap.beta,
            "k_a_per_um": cfg["carrier"]["k_a_per_um"],
            "chi_rad": launch_record["chi_rad"],
            "E_ref_planar_origin_V_per_m": planar[0],
        }
        arrays = {}
        map_fields = {label: {"values": {}, "background": {}} for label in labels}
        map_keys = ("H", "Hx", "Hz")
        for name, (axis_x, axis_z) in grids.items():
            xx, zz = (v.ravel() for v in np.meshgrid(axis_x, axis_z))
            mat_map = geometry.classify(cfg["layout"], xx, zz)
            values, back, regions, location = sample(xx, zz, mat_map, map_keys)
            pre = name + "__"
            arrays.update({pre + key: value for key, value in location.items()})
            arrays.update(
                {
                    pre + "x": xx,
                    pre + "z": zz,
                    pre + "mat": mat_map,
                    pre + "shape": [len(axis_z), len(axis_x)],
                }
            )
            for label in labels:
                map_fields[label]["values"].update({pre + k: v for k, v in values[label].items()})
            map_fields["T"]["background"].update({pre + k: v for k, v in back.items()})
            map_fields["T"]["background"][pre + "region"] = regions
        basename = {"L": "left_launch", "R": "right_launch", "T": "top_reference"}
        for label, item in map_fields.items():
            save_arrays(
                physics / "fields" / (basename[label] + "_map_fields.npz"),
                **item["values"],
                **{"bg__" + k: v for k, v in item["background"].items()},
            )
        save_arrays(physics / "sampling/map_points.npz", **arrays)
        refdata = {"s_um": s, "x_nm": x, "z_nm": z, "mat": mat}
        for name, physical in (("total", total), ("background", bg), ("scattered", primary["T"])):
            physical = dict(physical, **e_fields(physical, eps, om))
            refdata.update({name + "__" + k: physical[k] for k in ("H", "Ex", "Ez", "dxEx", "dxEz")})
        save_arrays(physics / "plot_data/reference_components_and_derivatives.npz", **refdata)
        for case in cases:
            factors = case_parts(cfg, case)
            G = {k: sum(weight * primary[label][k] for label, weight in factors) for k in fem_p4.POINT_KEYS}
            G.update(e_fields(G, eps, om))
            G_vec = np.stack([G["Ex"], G["Ez"]], axis=-1)
            J = intensity_frames(G_vec, t_vec, phases, cfg["four_P_V2_per_m2"])
            line = {
                "s_um": s,
                "x_nm": x,
                "z_nm": z,
                "phi_T_rad": phases,
                "four_P_V2_per_m2": cfg["four_P_V2_per_m2"],
                "J_frames": J,
                **{"G__" + k: v for k, v in G.items()},
                **{"T__" + k: v for k, v in ref_fields.items()},
            }
            for label, _ in factors:
                line.update({label + "__" + key: value for key, value in primary[label].items()})
            for i, phi in enumerate(phases):
                frame = G_vec + np.exp(1j * phi) * t_vec
                line[f"frame{i}__Ex"] = frame[:, 0]
                line[f"frame{i}__Ez"] = frame[:, 1]
                line[f"frame{i}__F"] = np.sum(np.abs(frame) ** 2, axis=-1)
            for label, field in (("T_scattered", primary["T"]), ("T_background", bg)):
                line.update({label + "__" + k: v for k, v in field.items()})
            save_arrays(physics / "interference" / f"detector_line_{case}.npz", **line)
            d = plot_case(cfg, case, line, {"arrays": arrays}, map_fields, inputs, planar)
            save_arrays(physics / "plot_data" / f"plot_{case}.npz", **d)
            record_plots(cfg, case, d, directory / "figures")
            if case == "left" and plane_map is not None:
                save_planar_comparison(physics, d, plane_line, planar, plane_map)
            save_arrays(
                physics / "plot_data" / f"central_planar_{case}.npz",
                s_um=s,
                x_nm=x,
                z_nm=z,
                E_launch=G_vec,
                E_reference=t_vec,
                E_planar=planar,
                phi_T_rad=phases,
                J_full=J,
                J_planar=intensity_frames(G_vec, planar, phases, cfg["four_P_V2_per_m2"]),
            )
    finally:
        stack.free_device()
        gpu.close()
