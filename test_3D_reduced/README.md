# Reduced 3D corner model

A single vertical gap-plasmon mode varies with local upper-silver thickness. Its in-plane amplitude is obtained by solving a complex scalar P4 finite-element problem in x and y. The code reconstructs the vector electric and magnetic fields from this amplitude, its derivatives and the local mode profile.

The computational domain contains the local meeting tapers with a PML around the lateral domain. Only the x arm launches a wave. The passive y slits are represented by silver in this reduction. The two 35 nm jobs have the same in-plane geometry. Their 2D launch amplitudes, reference amplitudes and phases capture the difference between sharp and rounded launch-device rims.

## Inputs and outputs

The selected job reads its launch record from `data/modes/` and its reference record from `data/sources/`. A straight-arm unit-feed calculation sets the amplitude for the corner solve. The default mesh has 25 nm elements with 10 nm refinement near the feed and taper lines. Changing either mesh size adds both mesh sizes to the run folder name. The field maps use a 5 nm observation grid and z = 10 nm.

The `fields/phase_maps_x.npz` product contains the computed corner launch field `E_spp_rom`, the analytical top-reference field `E_ref`, three phases `phi_rad`, the raw intensities `I_x`, normalized intensities `J_x`, and the water and observation masks. The last axis of each electric field is ordered x, y, z. The mirrored display is a separate derived plot.

- [Rounded corner calculation and downloads](../jobs/3d-reduced-rounded/README.md)
- [Sharp corner calculation and downloads](../jobs/3d-reduced-sharp/README.md)
