"""Local taper geometry, input data from the 2D launch, and the analytical top reference."""

import math
from pathlib import Path
import numpy as np
from sim2d import materials as mt, modes, storage
from sim2d.materials import GAP, WATER
from . import profile

DATA = Path(__file__).resolve().parents[1] / "data"
DEVICES = {
    "sharp": {
        "tag": "E2p01_td6_tm20_slit35glass_c1corner_xlaunch",
        "modes_record": "td6_E2.01_slit35glass_modes.json",
        "reference_record": "td6_E2.01_slit35glass_reference.json",
    },
    "rounded": {
        "tag": "E2p01_td6_tm20_slit35glass_rim5_c1corner_xlaunch",
        "modes_record": "td6_E2.01_slit35glass_rim5_modes.json",
        "reference_record": "td6_E2.01_slit35glass_rim5_reference.json",
    },
}
# Both 35 nm slit cases have the same xy reduction. The launch and reference carry the rim dependence.
A_NM, B_NM, L_NM, LAST_SLIT_END_NM = 1026.0, 1526.0, 2500.0, 926.0
PROBE_SHIFT_NM = 0.0
ALPHA_DEG = 29.173006
PHYSICS = {
    "energy_eV": 2.01,
    "eps_water": 1.77,
    "n_glass": 1.515,
    "eps_gap": 2.25,
    "eps_slit": 2.25,
    "ag_table": "Ag_JC.yml",
    "eps_ag": None,
    "eps_cr": None,
}
XS_NM = 0.5 * (
    LAST_SLIT_END_NM + A_NM
)  # equivalent feed line midway between the last X slit and the taper start
TAPER_NM, PLATEAU_NM = 500.0, 20.0  # taper length and plateau excess height
SLOPE = PLATEAU_NM / TAPER_NM  # |ds/du| on a taper, dimensionless (0.04)
TM_NM, TD_NM, CR_NM = 20.0, 6.0, 10.0
# glass/Ag interface, gap bottom, gap top, central upper-Ag top (-46, -26, -20, 0 for td 6; -50, -30, -20, 0 for td 10)
ZG_NM, Z_GB_NM, Z_GT_NM, Z_TOP_NM = -(2.0 * TM_NM + TD_NM), -(TM_NM + TD_NM), -TM_NM, 0.0
Z0_NM = 10.0
REAL_X, REAL_Y = (-200.0, L_NM), (-200.0, L_NM)  # physical (real) domain
PML_NM, PML_STRENGTH = 600.0, 12.0
OUTER_X = (REAL_X[0] - PML_NM, REAL_X[1] + PML_NM)
OUTER_Y = (REAL_Y[0] - PML_NM, REAL_Y[1] + PML_NM)

INTERFACES_NM = (ZG_NM, Z_GB_NM, Z_GT_NM, Z_TOP_NM)


def frequency():
    return mt.frequency(PHYSICS["energy_eV"])


def permittivities():
    """eps by material code (complex128; water 0, Ag 1, gap 2, glass 3, Cr 4, slit 5) and the provenance."""
    return mt.material_eps(PHYSICS, DATA)


def roof_s(u_nm):
    """s(u) (nm): plateau 20 for u <= a, linear taper to 0 at b, 0 beyond."""
    u = np.asarray(u_nm, dtype=np.float64)
    return np.where(u <= A_NM, PLATEAU_NM, np.where(u < B_NM, PLATEAU_NM * (B_NM - u) / TAPER_NM, 0.0))


def geometry_xy(x_nm, y_nm):
    """t_nm = 20 + max(s(x), s(y)), its gradient (tx, ty) (dimensionless; the x branch on the ridge s(x) = s(y),
    zero on the plateaus and the central film) and the Cr mask (x < a or y < a)."""
    x, y = np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64)
    sx, sy = roof_s(x), roof_s(y)
    t = TM_NM + np.maximum(sx, sy)
    on_x = (x > A_NM) & (x < B_NM) & (sx >= sy)
    on_y = (y > A_NM) & (y < B_NM) & (sy > sx)
    tx = np.where(on_x, -SLOPE, 0.0)
    ty = np.where(on_y, -SLOPE, 0.0)
    cr = (x < A_NM) | (y < A_NM)
    return t, tx, ty, cr


def branch_xy(x_nm, y_nm):
    """Coefficient branch of a lateral point (the mesh faces carry the same labels): 3 plateau (x <= a or y <= a),
    0 central (x >= b and y >= b), 1 x taper (a < x < b with s(x) >= s(y), i.e. above or on the ridge), 2 y taper.
    """
    x, y = np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64)
    sx, sy = roof_s(x), roof_s(y)
    plateau = (x <= A_NM) | (y <= A_NM)
    central = (x >= B_NM) & (y >= B_NM)
    xt = (~plateau) & (~central) & (x < B_NM) & (sx >= sy)
    return np.where(plateau, 3, np.where(central, 0, np.where(xt, 1, 2))).astype(np.int64)


def surface_top(x_nm, y_nm):
    """Top of the solid (metal or Cr) above (x, y), nm: s_C + 10 on the Cr plateaus."""
    t, _, _, cr = geometry_xy(x_nm, y_nm)
    return (t - TM_NM) + np.where(cr, CR_NM, 0.0)


def water_mask(x_nm, y_nm, z_nm):
    """True where (x, y, z) is water (strictly above the solid top)."""
    return np.asarray(z_nm, dtype=np.float64) > surface_top(x_nm, y_nm)


def pml_stretch(x_nm, y_nm):
    """s_x, s_y = 1 + 12 i (d/600)^2 with d the outward distance into the lateral PML (0 inside the real domain)."""
    x, y = np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64)
    dx = np.maximum(np.maximum(REAL_X[0] - x, x - REAL_X[1]), 0.0)
    dy = np.maximum(np.maximum(REAL_Y[0] - y, y - REAL_Y[1]), 0.0)
    return 1.0 + 1j * PML_STRENGTH * (dx / PML_NM) ** 2, 1.0 + 1j * PML_STRENGTH * (dy / PML_NM) ** 2


def mode_family(device):
    doc = storage.read_json(DATA / "modes" / DEVICES[device]["modes_record"])
    gap = next(mode for mode in doc["modes"] if mode["label"] == "central gap")
    eps, _ = permittivities()
    return profile.ModeFamily(
        eps,
        TM_NM * 1e-3,
        TD_NM * 1e-3,
        frequency()["k0_per_um"],
        TM_NM * 1e-3,
        modes.cval(gap["beta_per_um"]),
        1e-3,
        1.0 / (eps[GAP].real * modes.A_GAP_UM),
    )


def launch_record(device):
    """Physical launch and reference from the 2D run for the selected slit geometry."""
    spec = DEVICES[device]
    md = storage.read_json(DATA / "modes" / spec["modes_record"])
    lp = md["launch_projection"]
    src = storage.read_json(DATA / "sources" / spec["reference_record"])
    eps, _ = permittivities()
    f = frequency()
    ew = float(eps[WATER].real)
    k_w = f["k0_per_um"] * math.sqrt(ew)
    al = math.radians(ALPHA_DEG)
    u = np.array([modes.cval(lp["u_SPP_origin_V_per_m"]["x"]), modes.cval(lp["u_SPP_origin_V_per_m"]["z"])])
    return {
        "A0_A_per_m": modes.cval(lp["A0_A_per_m"]),
        "P_V2_per_m2": float(src["values"]["P_V2_per_m2"]),
        "u_V_per_m": u,
        "x_origin_nm": float(lp["applies_to"]["observation"]["x_origin_nm"]),
        "z0_nm": float(lp["applies_to"]["observation"]["z0_nm"]),
        "chi_rad": float(src["chi"]["chi_rad"]),
        "k_a_per_um": k_w * math.sin(al),
        "q_a_per_um": k_w * math.cos(al),
        "alpha_deg": ALPHA_DEG,
        "H_in_A_per_m": float(src["values"]["H_in_A_per_m"]),
        "C_Ohm": np.array([modes.cval(src["values"]["C_x_Ohm"]), modes.cval(src["values"]["C_z_Ohm"])]),
        "device": device,
    }


def reference_ideal_z0(x_nm, fam, rec):
    """E_ref(x, z0) of the production ideal family (V/m, shape (..., 3) = [Ex, 0, Ez]) at the fixed height z0, with
    the original phase origin x = b and the signed carrier k_a; defined at z0 only (the production family is a line
    model). Y-invariant."""
    beta = fam.root(fam.tm)["beta"]
    u = rec["u_V_per_m"]
    up = math.sqrt(rec["P_V2_per_m2"]) * u / np.linalg.norm(u)
    s_um = (np.asarray(x_nm, dtype=np.float64) - B_NM) * 1e-3
    fac = np.exp(-1j * rec["chi_rad"]) * np.exp((-beta.imag + 1j * rec["k_a_per_um"]) * s_um)
    out = np.zeros(np.shape(s_um) + (3,), dtype=np.complex128)
    out[..., 0] = up[0] * fac
    out[..., 2] = up[1] * fac
    return out


def ideal_spp_x(x_nm, z_nm, fam, A0):
    """The central ideal X SPP of the calibrated coefficient A0 in water, E = E_c(z; A0) exp(i beta_c (x - b)),
    V/m, shape (..., 3) = [Ex, 0, Ez]: E_x = -i A0 f_c'(z) 1e6/(omega eps0 eps_w), E_z = -beta_c 1e6 A0 f_c(z)/(omega
    eps0 eps_w) (modes.origin_vector form); valid for z in water (z > 0 of the flat central stack)."""
    eps, _ = permittivities()
    f = frequency()
    om = f["omega_eps0_S_per_m"]
    beta = fam.root(fam.tm)["beta"]
    z = np.asarray(z_nm, dtype=np.float64)
    x = np.asarray(x_nm, dtype=np.float64)
    fc, fcs = fam.profile(fam.tm, (z - ZG_NM) * 1e-3)
    den = om * complex(eps[WATER])
    ph = np.exp(1j * beta * (x - B_NM) * 1e-3)
    out = np.zeros(np.broadcast(x, z).shape + (3,), dtype=np.complex128)
    out[..., 0] = -1j * A0 * fcs * 1e6 / den * ph
    out[..., 2] = -beta * 1e6 * A0 * fc / den * ph
    return out
