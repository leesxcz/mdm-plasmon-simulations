"""Frequency-dependent material inputs and physical constants.

The photon energy is an explicit argument everywhere (no import-time frequency). Constants follow the accepted
mdm_uniform/mdm_beam definitions: lambda0 = 1.23984193/E um, k0 = 2 pi/lambda0, omega = c k0 (scipy.constants c,
epsilon_0, mu_0), Z0 = sqrt(mu0/eps0). Silver: data/Ag_JC.yml (Johnson & Christy 1972, refractiveindex.info),
linear interpolation of n and k in wavelength, eps = (n + i k)^2 (mdm_uniform.load_ag_jc, copied). Chromium:
Johnson & Christy 1974 n, k at the three tabulated energies of C1 (JC_CR); other energies need an explicit complex
permittivity. Material codes: water 0, Ag 1, gap dielectric 2, glass 3, Cr 4, slit dielectric 5 (C1 face names
water/ag/diel/glass/cr/slit)."""

import math

import numpy as np
from scipy.constants import c as C_LIGHT
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

HC_UM_EV = 1.23984193
Z0 = math.sqrt(MU0 / EPS0)
WATER, AG, GAP, GLASS, CR, SLIT = range(6)
NAMES = ("water", "silver", "gap dielectric", "glass", "chromium", "slit dielectric")
FACE_NAMES = {"water": WATER, "ag": AG, "diel": GAP, "glass": GLASS, "cr": CR, "slit": SLIT}
CR_JC = {1.02: (3.67, 3.60), 2.01: (3.17, 3.30), 2.50: (2.75, 3.30)}  # PIFLUXwave_C1.py JC_CR (J&C 1974)


def frequency(energy_eV):
    """lambda0 (um), k0 (1/um), omega (rad/s) and omega eps0 (S/m) for a photon energy (eV)."""
    lam0 = HC_UM_EV / float(energy_eV)
    k0 = 2.0 * np.pi / lam0
    omega = C_LIGHT * k0 * 1e6
    return {
        "energy_eV": float(energy_eV),
        "lambda0_um": lam0,
        "k0_per_um": k0,
        "k0_per_m": k0 * 1e6,
        "k0_per_nm": k0 * 1e-3,
        "omega_rad_per_s": omega,
        "omega_eps0_S_per_m": omega * EPS0,
    }


def load_ag_jc(path, lam):
    """Johnson & Christy n, k (refractiveindex.info YAML), interpolated linearly in wavelength; returns eps."""
    rows, on = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("data:"):
                on = True
                continue
            if on:
                parts = s.split()
                if len(parts) == 3:
                    try:
                        rows.append([float(p) for p in parts])
                        continue
                    except ValueError:
                        pass
                if rows:
                    break
    d = np.array(rows)
    if not d[0, 0] <= lam <= d[-1, 0]:
        raise ValueError(
            "wavelength %.6g um outside the tabulated silver data [%g, %g] um" % (lam, d[0, 0], d[-1, 0])
        )
    n = np.interp(lam, d[:, 0], d[:, 1])
    k = np.interp(lam, d[:, 0], d[:, 2])
    return (n + 1j * k) ** 2


def chromium(energy_eV, explicit=None):
    """Cr permittivity: an explicit complex value, else the tabulated J&C n, k at a supported energy."""
    if explicit is not None:
        return complex(explicit), "explicit complex permittivity input"
    for e, (n, k) in CR_JC.items():
        if abs(float(energy_eV) - e) <= 1e-12:
            return complex(n, k) ** 2, "Johnson & Christy 1974 n = %g, k = %g at %g eV (C1 JC_CR)" % (n, k, e)
    raise ValueError(
        "no chromium data at %g eV: supply physics.eps_cr explicitly (supported table energies %s)"
        % (energy_eV, sorted(CR_JC))
    )


def material_eps(physics, data_dir):
    """Relative permittivities by material code (complex128) and their provenance for the configured energy."""
    f = frequency(physics["energy_eV"])
    if physics.get("eps_ag") is not None:
        ag, ag_src = complex(physics["eps_ag"]), "explicit complex permittivity input"
    else:
        path = data_dir / physics["ag_table"]
        ag = complex(load_ag_jc(path, f["lambda0_um"]))
        ag_src = "%s: linear interpolation of n, k at lambda0 = %.12g um, eps = (n + i k)^2" % (
            physics["ag_table"],
            f["lambda0_um"],
        )
    cr, cr_src = chromium(physics["energy_eV"], physics.get("eps_cr"))
    eps = np.array(
        [
            complex(physics["eps_water"]),
            ag,
            complex(physics["eps_gap"]),
            complex(float(physics["n_glass"]) ** 2),
            cr,
            complex(physics["eps_slit"]),
        ],
        dtype=np.complex128,
    )
    prov = {
        "water": "input eps_water",
        "silver": ag_src,
        "gap dielectric": "input eps_gap",
        "glass": "input n_glass, eps = n^2",
        "chromium": cr_src,
        "slit dielectric": "input eps_slit",
    }
    if np.any(eps.imag < 0):
        raise ValueError("passive media need Im(eps) >= 0 for exp(-i omega t): %s" % eps)
    return eps, prov
