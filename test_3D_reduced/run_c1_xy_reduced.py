"""Run the local-corner gap-SPP model for the rounded or sharp 35 nm slit job."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
DEFAULT_H_NM = 25.0
DEFAULT_H_FINE_NM = 10.0
sys.path[:0] = [str(ROOT.parent / "2Dsimulation"), str(ROOT.parent)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("rounded", "sharp"), default="rounded")
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--h-nm", type=float, default=DEFAULT_H_NM)
    parser.add_argument("--h-fine-nm", type=float, default=DEFAULT_H_FINE_NM)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    root = args.output_root.resolve()
    from sim2d.runtime import set_runtime

    set_runtime(root / ".cache")
    import numpy as np
    from sim2d import mesh, solver, storage
    from test_3D_reduced.sim2d import c1_xy_calib as feed, c1_xy_coeff as coeff, c1_xy_common as corner
    from test_3D_reduced.sim2d import c1_xy_fields as fields, c1_xy_mesh as xy_mesh, c1_xy_solve as xy_solve

    case_tag = corner.DEVICES[args.geometry]["tag"]
    if args.h_nm != DEFAULT_H_NM or args.h_fine_nm != DEFAULT_H_FINE_NM:
        case_tag += f"_h{args.h_nm}nm_hfine{args.h_fine_nm}nm"
    destination = root / "c1_xy_reduced" / case_tag
    launch = corner.launch_record(args.geometry)
    print("Constructing the local mode family and coefficient tables", flush=True)
    family = corner.mode_family(args.geometry)
    tables = coeff.build_tables(family)
    cdir = destination / "coefficients"
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez(cdir / "tables.npz", **{key: value for key, value in tables.items() if not key.startswith("_")})
    tab, volume, elements = xy_mesh.prepare(destination / "mesh", args.h_nm, args.h_fine_nm)
    mesh.write_tables(tab, destination / "mesh/p4_tables")
    calibration = feed.calibrate(tables, launch, destination / "calibration")
    storage.write_json(destination / "calibration/feed_calibration.json", calibration)
    print("Solving the X-launched corner field", flush=True)
    K = xy_solve.assemble_operator(tab, lambda x, y: coeff.pml_coefficients(x, y, tables))
    b = xy_solve.line_load(tab, calibration["a_bar_p"])
    with solver.Factorization(K, tab.boundary) as factors:
        psi = factors.solve(b)
    sdir = destination / "solve"
    sdir.mkdir(parents=True, exist_ok=True)
    np.savez(sdir / "psi.npz", psi=psi, b=b, boundary=tab.boundary)
    print("Reconstructing the vector fields and interference maps", flush=True)
    directory = destination / "fields"
    directory.mkdir(parents=True, exist_ok=True)
    result = fields.sample(family, tables, tab, volume, elements, psi)
    maps = fields.maps(family, result, launch)
    mirrored = fields.mirrored_maps(family, result, launch)
    np.savez(directory / "fields_x_z10.npz", **result)
    np.savez(directory / "phase_maps_x.npz", **maps)
    np.savez(directory / "phase_maps_mirrored.npz", **mirrored)
    fields.draw(maps, directory / "figures")
    storage.write_json(
        destination / "configuration.json",
        {
            "geometry": args.geometry,
            "h_nm": args.h_nm,
            "h_fine_nm": args.h_fine_nm,
            "z0_nm": corner.Z0_NM,
            "launch": launch,
            "phase_rad": list(fields.PHASES_RAD),
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
