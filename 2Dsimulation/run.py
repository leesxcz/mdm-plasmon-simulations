"""Run a 2D MDM cross-section with the paper geometry or the sharp-rim comparison."""

import argparse
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "data/configurations/td6_p99_s35_glass_rimR5.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PAPER,
        help="Input or saved configuration JSON. Relative paths are checked in 2Dsimulation/ first, then in the working directory.",
    )
    parser.add_argument("--geometry", choices=("rounded", "sharp"), help="Select a paper job geometry.")
    parser.add_argument("--case", choices=("left", "right", "counter", "all"), default="left")
    parser.add_argument("--stage", choices=("prepare", "device", "all"), default="device")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    package_path = ROOT / args.config
    config_path = (package_path if package_path.is_file() else args.config).resolve()
    root = args.output_root.resolve()
    from sim2d.runtime import set_runtime

    set_runtime(root / ".cache")
    from sim2d import config, fem_p4, layered, mesh, modes, postprocess, projection, solver, sources, storage

    parameters = config.read_parameters(config_path)
    if args.geometry == "sharp":
        parameters["lateral_nm"].pop("slit_rim_radius", None)
        parameters["display"]["interference_windows_um"] = [None, [0.0, 2.0]]
    elif args.geometry == "rounded":
        parameters["lateral_nm"]["slit_rim_radius"] = 5.0
    cfg = config.resolve(parameters)
    prepared = root.joinpath(*cfg["output"]["physics_parts"]) / "physics"
    print("Preparing the cross-section and P4 basis", flush=True)
    tab, locator = mesh.prepare(cfg, prepared / "mesh")
    mesh.write_tables(tab, prepared / "mesh" / "p4_integration_tables")
    gap_modes = modes.device_modes(cfg)
    mode_data = {
        "modes": [mode.record for mode in gap_modes.values()],
        "constants": {k: cfg["frequency"][k] for k in ("k0_per_um", "omega_eps0_S_per_m")},
        "applies_to": {
            "energy_eV": cfg["frequency"]["energy_eV"],
            "k0_per_um": cfg["frequency"]["k0_per_um"],
            "eps": {key: modes.cj(cfg["eps"][i]) for i, key in modes.EPS_KEYS.items()},
        },
        "modal_origin": {"z_C1_nm": cfg["layout"]["zg"], "rule": "s_mode_um = (z_C1_nm - z_origin_nm)/1000"},
    }
    storage.write_json(prepared / "modes/gap_modes.json", mode_data)
    if args.stage == "prepare":
        print(prepared)
        return
    gap = gap_modes["central gap"]
    cases = ("left", "right", "counter") if args.case == "all" else (args.case,)
    launches = sources.launch_tables(cfg)
    labels = {label for case in cases for label, _ in postprocess.case_parts(cfg, case)}
    labels.add("L")  # the physical amplitude of the applied reference comes from the left launch
    print("Assembling the TM/PML system", flush=True)
    K = fem_p4.assemble_operator(tab, cfg["eps"][tab.material], cfg["frequency"]["k0_per_nm"], cfg["pml"])
    rhs, solutions = {}, {}
    # The launched magnetic field has the natural boundary condition on the outer PML.
    with solver.Factorization(K) as launch_factors:
        gpu = fem_p4.GPUTri4(tab, cfg["eps"][tab.material], "source sheets", assembly=False)
        try:
            for label, name in (("L", "left"), ("R", "right")):
                if label in labels:
                    sheet = launches[name]
                    rhs[label] = gpu.sheet_load_table(
                        tab.source_edges,
                        sheet["x_nm"],
                        sheet["sigma"],
                        (cfg["layout"]["xmin"], cfg["layout"]["xmax"]),
                        1e-9,
                    )
        finally:
            gpu.close()
        for label, load in rhs.items():
            print("Solving the " + label + " launch", flush=True)
            solutions[label] = launch_factors.solve(load)
    if args.geometry == "sharp" and args.stage == "all" and config_path == PAPER.resolve():
        doc = storage.read_json(ROOT / "data/modes/td6_E2.01_slit35glass_modes.json")["launch_projection"]
        launch_record = dict(
            doc["calibration"],
            A0_A_per_m=doc["A0_A_per_m"],
            u_SPP_origin_V_per_m=doc["u_SPP_origin_V_per_m"],
        )
    else:
        launch_record = projection.from_launch(cfg, tab, solutions["L"], gap)
    parameters["normalization"]["P_V2_per_m2"] = launch_record["P_V2_per_m2"]
    parameters["reference"]["oblique"]["H_in_A_per_m"] = launch_record["H_in_A_per_m"]
    cfg = config.resolve(parameters)
    physics = root.joinpath(*cfg["output"]["physics_parts"]) / "physics"
    directory = root.joinpath(*cfg["output"]["parts"])
    if physics != prepared:
        for sub in ("mesh", "geometry", "modes"):
            if (prepared / sub).is_dir():
                shutil.copytree(prepared / sub, physics / sub, dirs_exist_ok=True)
    storage.write_json(directory / "configuration.json", config.folder_record(cfg))
    storage.write_json(physics / "parameters.json", {"parameters": cfg["parameters"]})
    storage.write_json(physics / "projection/left_launch_projection.json", {"results": launch_record})
    postprocess.save_arrays(
        physics / "sources/left_right_launch_sources.npz",
        **{name + "_" + k: d[k] for name, d in launches.items() for k in ("x_nm", "sigma")}
    )
    beam = sources.oblique_beam(cfg)
    postprocess.save_arrays(physics / "sources/top_reference_spectrum.npz", k=beam.k, q=beam.q, c=beam.c)
    freq = cfg["frequency"]
    stack = layered.Stack(
        beam, cfg["eps"], cfg["stacks"]["plateau"], freq["k0_per_m"], freq["omega_eps0_S_per_m"]
    )
    try:
        region = stack.region_of_z(tab.P[tab.T, 1].mean(axis=1))
        rhs["T"] = fem_p4.contrast_load_p4(tab, tab.material, region, cfg["eps"], stack, 6)[0]
    finally:
        stack.free_device()
    print("Solving the scattered top-reference field", flush=True)
    # Only the scattered reference is zero on the outer PML boundary.
    with solver.Factorization(K, tab.boundary) as reference_factors:
        solutions["T"] = reference_factors.solve(rhs["T"])
    for label, name in (("L", "left_launch"), ("R", "right_launch"), ("T", "top_reference_scattered")):
        if label in solutions:
            postprocess.save_arrays(physics / "solutions" / (name + ".npz"), u=solutions[label], b=rhs[label])
    print("Evaluating electric fields and interference", flush=True)
    postprocess.evaluate(
        cfg,
        tab,
        locator,
        solutions,
        beam,
        gap,
        launch_record,
        cases,
        physics,
        directory,
        planar_map=args.stage == "device" and "left" in cases,
    )
    print(directory)


if __name__ == "__main__":
    main()
