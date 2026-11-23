# Sharp 2D glass-slit case

The slit walls meet the film surfaces at sharp corners. This job uses the paper input with the rim radius omitted and the historical 0–2 µm plot window. It uses the sharp launch coefficient from the packaged 2D mode record. The saved normalization is P = 150.7736484241917 V²/m² and the applied reference amplitude is H_in = 0.03137207030525111 A/m.

## Run

Activate the [repository environment](../../README.md#run-the-code) and run the following from the repository root.

```powershell
python -B 2Dsimulation/run.py --geometry sharp --case left --stage all --output-root 2Dsimulation/runs
```

New results go into `2Dsimulation/runs/`. A recomputed launch can give a slightly different calibrated amplitude and corresponding folder name.

## Saved data

Download `2d-sharp-run.zip` and `2d-sharp-fields.zip` from Releases. Place the two ZIPs in a sibling folder named `release-assets` and extract from the repository root.

```powershell
python -B -m zipfile -e ../release-assets/2d-sharp-run.zip .
python -B -m zipfile -e ../release-assets/2d-sharp-fields.zip .
```

The saved job opens at the following path.

```text
2Dsimulation/outputs/E2p01_td6_tm20/hScale1_PML600s12/oblique_a29p173006/epsSlit2p295225_slitWidth35_Hin0p03137207031_P150p7736484/
```

Open `physics/plot_data/plot_left.npz` for the launch and total reference on the detector line, `physics/interference/detector_line_left.npz` for the three phase frames, or `figures/` for the original figures.
