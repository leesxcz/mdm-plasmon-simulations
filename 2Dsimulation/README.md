# 2D full-wave cross-section

The unknown is the complex transverse-magnetic field `H_y(x,z)`. The model uses the actual layer permittivities, the configured tapers and slits, and absorbing layers around the cross-section. A focused sheet in glass drives the grating. An oblique beam with its full propagating angular spectrum supplies the top reference.

The grating launch uses the natural finite-element boundary condition on the outer PML. For the top beam, the solver fixes the scattered magnetic field to zero on that boundary. The field readout adds the layered background before combining it with the launch. For each phase φ the plotted intensity is `|E_launch + exp(iφ) E_reference|² / (4P)`. Fields use SI units and the exp(−iωt) convention. Geometry coordinates are in nm.

## Inputs

| File or directory | Use |
| --- | --- |
| [Paper configuration](data/configurations/td6_p99_s35_glass_rimR5.json) | Complete starting input for the 5 nm rounded rim. The reference amplitude and normalization here are provisional until the launch projection. |
| `data/default_parameters.json` | Reference values for building readable output folder names. |
| `data/modes/` | Bound-mode roots for the layer stacks and the recorded sharp-device launch amplitude. |
| `data/Ag_JC.yml` | Wavelength-dependent silver optical data from Johnson and Christy. |
| `sim2d/` | Geometry, P4 finite elements, stratified optical fields, launch projection, cuDSS solve and field readout. |

The solver constructs its Netgen mesh from the polygons. The rounded job projects the solved launch onto the forward gap mode and computes the applied reference amplitude. Both jobs use the same paper input. With that input, `--geometry sharp --stage all` selects the sharp geometry and its recorded launch calibration. A saved run’s `configuration.json` can be passed to `--config` to reuse its parameters. When a separate config file is supplied, the runner recalculates the launch calibration.

## Jobs

- [Rounded paper calculation and downloads](../jobs/2d-rounded-paper/README.md)
- [Sharp calculation and downloads](../jobs/2d-sharp/README.md)

The saved datasets include native solution vectors, detector-line fields, phase frames, figure data, and original figures. Fresh runs write the native solutions, the launch normalization, the detector line, the electric and magnetic field maps, and interference plots. In the line plots, solid curves are full-wave results and dashed curves are the analytical comparison.
