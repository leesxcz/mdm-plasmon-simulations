# Rounded 2D paper case

A 5 nm radius rounds each rim of the 35 nm glass-filled grating slits. The input projection uses the computed left launch to obtain the reference amplitude of this rounded device. The saved normalization is P = 166.8898681494581 V²/m² and the applied reference amplitude is H_in = 0.03300619364834117 A/m.

## Run

Activate the [repository environment](../../README.md#run-the-code) and run the following from the repository root.

```powershell
python -B 2Dsimulation/run.py --geometry rounded --case left --stage prepare --output-root 2Dsimulation/runs
python -B 2Dsimulation/run.py --geometry rounded --case left --stage device --output-root 2Dsimulation/runs
```

New results go into `2Dsimulation/runs/`. A recomputed launch can give a slightly different calibrated amplitude and corresponding folder name.

## Saved data

Download `2d-rounded-paper-run.zip` and `2d-rounded-paper-fields.zip` from Releases. Place the two ZIPs in a sibling folder named `release-assets` and extract from the repository root.

```powershell
python -B -m zipfile -e ../release-assets/2d-rounded-paper-run.zip .
python -B -m zipfile -e ../release-assets/2d-rounded-paper-fields.zip .
```

The saved job opens at the following path.

```text
2Dsimulation/outputs/E2p01_td6_tm20/hScale1_PML600s12/oblique_a29p173006/epsSlit2p295225_slitWidth35_slitRimR5_Hin0p03300619365_P166p8898681/
```

Open `physics/plot_data/plot_left.npz` for the launch and total reference on the detector line, `physics/interference/detector_line_left.npz` for the three phase frames, or `windowsnonex0x3/figures/` for the original figures. The `physics/plot_data/central_planar_comparison.npz` file holds the full and central-planar reference fields and their interference intensities on both the detector line and the water map.
