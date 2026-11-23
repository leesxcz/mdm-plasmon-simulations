# MDM plasmon simulations

This repository contains the 2D full-wave and reduced 3D simulation code associated with the paper *Plasmon-Enabled High-Precision Single Molecule Localization Microscopy over an Extended Field of View*. [How to cite](#how-to-cite)

The simulations calculate electromagnetic fields and interference patterns in a metal–dielectric–metal structure for single-molecule localization microscopy. The 2D solver models the transverse-magnetic cross-section. The reduced 3D solver follows the gap surface plasmon polariton through a local corner and reconstructs its three electric-field components.

## Choose a job

| Job | Calculation | Slit rim |
| --- | --- | --- |
| [2D rounded paper case](jobs/2d-rounded-paper/README.md) | Full-wave launch, top reference and interference in the device cross-section | 5 nm radius |
| [2D sharp case](jobs/2d-sharp/README.md) | The same cross-section with sharp glass-filled slits | Sharp |
| [Reduced 3D rounded corner](jobs/3d-reduced-rounded/README.md) | Local corner field from the rounded device’s 2D launch | 5 nm launch-device rim |
| [Reduced 3D sharp corner](jobs/3d-reduced-sharp/README.md) | Local corner field from the sharp device’s 2D launch | Sharp launch-device rim |

The four cases use a photon energy of 2.01 eV, a 6 nm dielectric gap, a 99 nm grating period and 35 nm glass-filled slits. The observation height is 10 nm above the central upper silver surface. The reduced solver represents the active x grating with a line feed. The feed amplitude and top reference are specific to the selected 2D geometry.

## Run the code

With the simulation environment active, run one of these commands from the repository root. Each command writes results under the corresponding solver’s `runs/` directory.

**2D rounded paper case**

```powershell
python 2Dsimulation/run.py --output-root 2Dsimulation/runs
```

**2D sharp**

```powershell
python 2Dsimulation/run.py --geometry sharp --stage all --output-root 2Dsimulation/runs
```

**Reduced 3D rounded corner**

```powershell
python test_3D_reduced/run_c1_xy_reduced.py --geometry rounded --output-root test_3D_reduced/runs
```

**Reduced 3D sharp corner**

```powershell
python test_3D_reduced/run_c1_xy_reduced.py --geometry sharp --output-root test_3D_reduced/runs
```

The 2D runner builds the mesh automatically. Each job page lists its physical inputs and the location of the saved dataset.

### Environment setup

The saved calculations used Windows, Python 3.12, NGSolve/Netgen, and an NVIDIA RTX 5090 with 32 GB of VRAM. Both solvers use custom CUDA kernels to evaluate fourth-order finite-element fields and their derivatives at observation points and to sum element contributions during sparse matrix assembly. NVIDIA cuDSS 0.8 for CUDA 13 solves the resulting complex sparse systems. Use a 64-bit Windows environment with a compatible NVIDIA driver. A short checkout path accommodates the parameter-based output names.

From an Anaconda or Miniforge prompt in the repository root, create and activate the environment.

```powershell
conda env create -f environment.yml
conda activate mdm-plasmon
```

## Read the saved results

Each job page names its ZIP downloads in GitHub Releases. Extract them at the repository root. The archives contain the full solver-relative paths. The 2D results appear under `2Dsimulation/outputs/` and the reduced corner results under `test_3D_reduced/outputs/c1_xy_reduced/`. Existing PNG figures and captions can be opened directly. NumPy reads the numerical arrays from the NPZ files.

[2D solver and conventions](2Dsimulation/README.md) · [Reduced corner solver and conventions](test_3D_reduced/README.md)

## How to cite

If you use these simulations or their saved fields in a publication, please cite our paper, *Plasmon-Enabled High-Precision Single Molecule Localization Microscopy over an Extended Field of View*. [arXiv:2606.31758](https://arxiv.org/abs/2606.31758) · [BibTeX from arXiv](https://arxiv.org/bibtex/2606.31758)

## Technical references

### Optical methods

- **Gap-plasmon dispersion.** Both models use the transverse-magnetic mode confined between two silver layers. J. A. Dionne, L. A. Sweatlock, H. A. Atwater, and A. Polman (2006) analyze the propagation constant and field confinement of planar metal–dielectric–metal slots in “Plasmon slot waveguides: Towards chip-scale propagation with subwavelength-scale localization,” *Physical Review B* **73**, 035407. [DOI](https://doi.org/10.1103/PhysRevB.73.035407)
- **Multilayer reflection and transmission.** For each plane wave, the 2D solver uses transfer matrices to compute the reflected field and the field inside each layer. S. J. Byrnes (2016) derives the oblique-incidence formulas for absorbing layers in “Multilayer optical calculations,” arXiv:1603.02720. [Derivation](https://arxiv.org/abs/1603.02720)
- **Scattered-field finite elements.** The 2D top-illumination calculation first solves the planar stack. The material contrast with that stack supplies the source for the scattered field. G. Demésy, F. Zolla, A. Nicolet, and B. Vial describe this formulation in “Finite Element Method,” chapter 5 of E. Popov (ed.), *Gratings: Theory and Numeric Applications* (2012). [Authors’ chapter](https://arxiv.org/abs/1302.1032)
- **Forward-mode amplitude.** The rounded 2D job extracts the forward gap-plasmon amplitude by taking an unconjugated overlap of the computed fields with the backward mode. For the Lorentz-reciprocity relation behind this projection in lossy waveguides, see G. K. Svendsen, M. W. Haakestad, and J. Skaar (2013), “Reciprocity and the scattering matrix of waveguide modes,” *Physical Review A* **87**, 013838. [DOI](https://doi.org/10.1103/PhysRevA.87.013838)
- **Single-mode reduction of the 3D corner.** A. Ivanova (2010), *Dimensionality reduction in computational photonics*, University of Twente, gives the general variational method of expanding the field in vertical waveguide modes and solving for their in-plane amplitudes. In this repository the retained mode follows the local upper-silver thickness. The corner coefficients and calibrated line feed are specific to this model. [Thesis DOI](https://doi.org/10.3990/1.9789036529976)

### Mesh and absorbing boundaries

- **Netgen.** Netgen generates the meshes for the 2D cross-section and the reduced corner model. J. Schöberl (1997). “NETGEN – An advancing front 2D/3D-mesh generator based on abstract rules.” *Computing and Visualization in Science* **1**, 41–52. [DOI](https://doi.org/10.1007/s007910050004)
- **Perfectly matched layers.** Both models use coordinate stretching at the outer boundaries. W. C. Chew and W. H. Weedon (1994). “A 3D perfectly matched medium from modified Maxwell’s equations with stretched coordinates.” *Microwave and Optical Technology Letters* **7**, 599–604. [DOI](https://doi.org/10.1002/mop.4650071304)

### Optical constants

- **Silver.** The code interpolates the measured refractive index and extinction coefficient in `Ag_JC.yml`. P. B. Johnson and R. W. Christy (1972). “Optical Constants of the Noble Metals.” *Physical Review B* **6**, 4370–4379. [DOI](https://doi.org/10.1103/PhysRevB.6.4370)
- **Chromium.** The paper configuration takes its chromium optical constants from a separate measurement by P. B. Johnson and R. W. Christy (1974). “Optical constants of transition metals: Ti, V, Cr, Mn, Fe, Co, Ni, and Pd.” *Physical Review B* **9**, 5056–5070. [DOI](https://doi.org/10.1103/PhysRevB.9.5056)
- **Optical-constants database.** The bundled silver table comes from the database described by M. N. Polyanskiy (2024). “Refractiveindex.info database of optical constants.” *Scientific Data* **11**, article 94. [DOI](https://doi.org/10.1038/s41597-023-02898-2)

### Numerical software

- **NGSolve.** Both solvers use NGSolve’s fourth-order finite-element basis and quadrature rules. J. Schöberl (2014). “C++11 Implementation of Finite Elements in NGSolve.” ASC Report 30/2014, Institute for Analysis and Scientific Computing, Vienna University of Technology. [Technical report](https://www.ngsolve.org/_static/ngs-cpp11.pdf)
- **CuPy** provides the GPU arrays and compiles and launches the solvers’ custom CUDA kernels. R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Loomis (2017). “CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations.” Workshop on Machine Learning Systems at NIPS 2017. [Authors’ citation](https://github.com/cupy/cupy/blob/main/CITATION.bib)
- **NVIDIA cuDSS** solves the assembled complex sparse systems on the GPU. The saved runs used NVIDIA Corporation’s cuDSS 0.8.0.10 for CUDA 13. [cuDSS 0.8.0 documentation](https://docs.nvidia.com/cuda/cudss/0.8.0/index.html)
- **NumPy** handles numerical arrays and field data. C. R. Harris et al. (2020). “Array programming with NumPy.” *Nature* **585**, 357–362. [DOI](https://doi.org/10.1038/s41586-020-2649-2)
- **SciPy** supplies physical constants, interpolation, and the one-dimensional feed calibration’s sparse solve. P. Virtanen et al. (2020). “SciPy 1.0: fundamental algorithms for scientific computing in Python.” *Nature Methods* **17**, 261–272. [DOI](https://doi.org/10.1038/s41592-019-0686-2)
- **Matplotlib** produces the field and intensity plots. J. D. Hunter (2007). “Matplotlib: A 2D Graphics Environment.” *Computing in Science & Engineering* **9**(3), 90–95. [DOI](https://doi.org/10.1109/MCSE.2007.55)
