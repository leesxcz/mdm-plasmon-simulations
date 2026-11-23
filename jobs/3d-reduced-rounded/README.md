# Reduced 3D corner with the rounded launch

This job uses the 35 nm rounded glass-slit launch to drive the local reduced corner. Both the amplitude and phase of the input come from the matching 2D case. The saved normalization is P = 166.8898681494581 V²/m² and the applied reference amplitude is H_in = 0.03300619364834117 A/m.

## Run

Activate the [repository environment](../../README.md#run-the-code) and run the following from the repository root.

```powershell
python -B test_3D_reduced/run_c1_xy_reduced.py --geometry rounded --stage all --h-nm 25 --h-fine-nm 10 --output-root test_3D_reduced/runs
```

New results go into `test_3D_reduced/runs/`. The output case folder is `E2p01_td6_tm20_slit35glass_rim5_c1corner_xlaunch`.

## Saved data

Download `3d-reduced-rounded-data.zip` from Releases. Place the ZIP in a sibling folder named `release-assets` and extract from the repository root.

```powershell
python -B -m zipfile -e ../release-assets/3d-reduced-rounded-data.zip .
```

The saved job opens at the following path.

```text
test_3D_reduced/outputs/c1_xy_reduced/E2p01_td6_tm20_slit35glass_rim5_c1corner_xlaunch/
```

Open `fields/phase_maps_x.npz` for the three normalized phase maps and `fields/fields_x_z10.npz` for the reconstructed complex electric and magnetic fields. Open `fields/figures/` for the original figures.
