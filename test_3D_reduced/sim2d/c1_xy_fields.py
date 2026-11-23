"""Electromagnetic field reconstruction and local corner phase maps."""

import math
import numpy as np
from sim2d import mesh as meshmod, fem_p4
from sim2d.materials import AG, CR, GAP, GLASS, WATER
from . import c1_xy_common as cc

PHASES_RAD = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
GRID_DX_NM = 5.0  # observation spacing (a sampling choice, independent of the element size)
GRID_OFFSET_NM = 2.5  # x = b + 2.5 + 5 k: no sample on the branch line x = b
CUT_OFFSETS_NM = (
    59.5,
    259.5,
    459.5,
    859.5,
)  # the Y cuts measured from the Y-taper end y = b (the report's cuts)
Y_CUTS_NM = tuple(
    cc.B_NM + d for d in CUT_OFFSETS_NM
)  # absolute rows (1600, 1800, 2000, 2400 nm for b = 1540.5)
Y_GRID_START_NM = (
    cc.B_NM + CUT_OFFSETS_NM[0]
) % GRID_DX_NM  # y rows y0 + 5 k hold the cut rows (y0 = 0 for 1540.5)
EXTRA_HEIGHTS_NM = (30.0, 60.0)
LOCATE_BATCH = 32768  # points per Locator call
CHUNK_POINTS = 2048  # GPU evaluation chunk (plan section 9: the 2048-point chunk bound)


class BranchLocator(meshmod.Locator):
    """mesh.Locator built from in-memory arrays with the coefficient branch of every element as the material label
    (mesh.Locator loads 'material' from an npz; the reduced mesh stores the branch codes of c1_xy_common.branch_xy).
    """

    def __init__(self, mesh, points_nm, triangles, branch):
        self.P = np.asarray(points_nm, dtype=np.float64)
        self.T = np.asarray(triangles, dtype=np.int64)
        self.mat = np.asarray(branch, dtype=np.int64)
        self.mesh = mesh
        nv = self.P.shape[0]
        flat = self.T.ravel()
        order = np.argsort(flat, kind="stable")
        self.v2t_ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=nv))])
        self.v2t = (order // 3).astype(np.int64)
        self.cen = self.P[self.T].mean(axis=1)


def mesh_arrays(mesh_npz):
    """points_nm, triangles and the element branch of the reduced mesh npz ('branch', else 'material')."""
    with np.load(mesh_npz, allow_pickle=False) as d:
        branch = d["branch"] if "branch" in d.files else d["material"]
        return d["points_nm"], d["triangles"], np.asarray(branch, dtype=np.int64)


def evaluate_psi(tab, volume, mesh_file, psi, x_nm, y_nm):
    """Evaluate the scalar P4 solution and its two in-plane derivatives."""
    import ngsolve as ng

    x = np.asarray(x_nm, dtype=np.float64)
    y = np.asarray(y_nm, dtype=np.float64)
    shape = np.broadcast(x, y).shape
    xf = np.ascontiguousarray(np.broadcast_to(x, shape)).ravel()
    yf = np.ascontiguousarray(np.broadcast_to(y, shape)).ravel()
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    P, T, branch = mesh_arrays(mesh_file)
    mesh = ng.Mesh(str(volume))
    loc = BranchLocator(mesh, P, T, branch)
    want = cc.branch_xy(xf, yf)
    n = xf.size
    el = np.empty(n, dtype=np.int64)
    l0, l1 = np.empty(n), np.empty(n)
    for s0 in range(0, n, LOCATE_BATCH):
        sl = slice(s0, min(n, s0 + LOCATE_BATCH))
        mp, e = loc.locate(xf[sl], yf[sl], want[sl])
        el[sl], l0[sl], l1[sl] = (
            e,
            np.asarray(mp["x"], dtype=np.float64),
            np.asarray(mp["y"], dtype=np.float64),
        )
    gpu = fem_p4.GPUTri4(
        tab, np.ones(int(tab.ne)), "corner field", assembly=False, columns=1, chunk_points=CHUNK_POINTS
    )
    try:
        gpu.set_solution(psi)
        v = gpu.points(el, l0, l1, keys=("H", "Hx", "Hz"))
    finally:
        gpu.close()
    Psi = v["H"][:, 0].reshape(shape)
    Px = v["Hx"][:, 0].reshape(shape)
    Py = v["Hz"][:, 0].reshape(shape)
    return Psi, Px, Py


def permittivity_at(x_nm, y_nm, z_nm):
    """eps_r (complex128) of every point and the water and Cr masks: water strictly above the solid top
    (c1_xy_common.water_mask), Cr on the plateaus for metal top < z <= metal top + 10 nm, silver in the upper Ag
    (Z_GT_NM <= z <= metal top) and the lower Ag, the gap dielectric for Z_GB_NM <= z < Z_GT_NM, glass below ZG_NM
    (c1_xy_common: -26, -20, -46 nm for td 6; -30, -20, -50 nm for td 10)."""
    eps, _ = cc.permittivities()
    x, y, z = np.broadcast_arrays(
        np.asarray(x_nm, dtype=np.float64),
        np.asarray(y_nm, dtype=np.float64),
        np.asarray(z_nm, dtype=np.float64),
    )
    t, _, _, cr = cc.geometry_xy(x, y)
    metal_top = t - cc.TM_NM
    water = cc.water_mask(x, y, z)
    chromium = cr & (z > metal_top) & ~water
    out = np.full(z.shape, eps[AG], dtype=np.complex128)
    out[(z >= cc.Z_GB_NM) & (z < cc.Z_GT_NM)] = eps[GAP]
    out[z < cc.ZG_NM] = eps[GLASS]
    out[chromium] = eps[CR]
    out[water] = eps[WATER]
    return out, water, chromium


def reconstruct(fam, tables, Psi, Px, Py, x_nm, y_nm, z_nm, parts=None):
    """E (..., 3) = [Ex, Ey, Ez] (V/m) and H (..., 3) (A/m) of (11)-(12) at the points (x, y, z) (nm) from Psi (A/m),
    Px = dPsi/dx and Py = dPsi/dy (per nm): hbar_x = -C_y Py, hbar_y = C_x Px with c1_xy_coeff.coefficients_xy (the
    real-domain coefficients, not the PML ones), (fbar, fbar_z, fbar_t) = c1_xy_coeff.profile_normalized at the local
    thickness and the physical height, eta = 1/eps_r of the actual material at the point (permittivity_at),
    factor = i eta 1e9/(omega eps0): Ex = -factor fbar_z hbar_y, Ey = factor fbar_z hbar_x,
    Ez = factor [fbar M Psi + fbar_t (-t_y hbar_x + t_x hbar_y)], H = (fbar hbar_x, fbar hbar_y, 0). parts (dict,
    optional) receives the two terms of Ez and the ingredients."""
    from . import c1_xy_coeff

    Psi, Px, Py, x, y, z = np.broadcast_arrays(
        np.asarray(Psi, dtype=np.complex128),
        np.asarray(Px, dtype=np.complex128),
        np.asarray(Py, dtype=np.complex128),
        np.asarray(x_nm, dtype=np.float64),
        np.asarray(y_nm, dtype=np.float64),
        np.asarray(z_nm, dtype=np.float64),
    )
    x, y, z = (np.ascontiguousarray(v) for v in (x, y, z))
    t, tx, ty, _ = cc.geometry_xy(x, y)
    Cx, Cy, M = c1_xy_coeff.coefficients_xy(x, y, tables)
    hx = -Cy * Py
    hy = Cx * Px
    fbar, fbar_z, fbar_t = c1_xy_coeff.profile_normalized(fam, t, z, tables)
    fbar, fbar_z, fbar_t = (
        np.broadcast_to(np.asarray(v, dtype=np.complex128), x.shape) for v in (fbar, fbar_z, fbar_t)
    )
    eps_pts, _, _ = permittivity_at(x, y, z)
    om = cc.frequency()["omega_eps0_S_per_m"]
    eta = 1.0 / eps_pts
    factor = 1j * eta * 1e9 / om
    tangent = fbar_t * (-ty * hx + tx * hy)
    Ex = -factor * fbar_z * hy
    Ey = factor * fbar_z * hx
    Ez_profile = factor * fbar * M * Psi
    Ez_tangent = factor * tangent
    E = np.stack([Ex, Ey, Ez_profile + Ez_tangent], axis=-1)
    H = np.stack([fbar * hx, fbar * hy, np.zeros(x.shape, dtype=np.complex128)], axis=-1)
    if parts is not None:
        parts.update(
            {
                "Ez_profile_term": Ez_profile,
                "Ez_tangent_term": Ez_tangent,
                "hbar_x": hx,
                "hbar_y": hy,
                "fbar": fbar,
                "fbar_z_per_nm": fbar_z,
                "fbar_t_per_nm": fbar_t,
                "t_nm": t,
                "tx": tx,
                "ty": ty,
                "Cx_nm": Cx,
                "Cy_nm": Cy,
                "M_per_nm": M,
                "eps": eps_pts,
            }
        )
    return E, H


def phase_maps(E_spp, E_ref, phis, four_P):
    """I_phi = sum_j |E_spp,j + exp(i phi) E_ref,j|^2 for every phase (the vector sum before the squared norm; shape
    (nphi, ...)) and J = I/(4P) with the fixed 4P divided once (never per frame)."""
    E_spp = np.asarray(E_spp, dtype=np.complex128)
    E_ref = np.asarray(E_ref, dtype=np.complex128)
    phis = np.asarray(phis, dtype=np.float64).ravel()
    inten = np.stack([np.sum(np.abs(E_spp + np.exp(1j * ph) * E_ref) ** 2, axis=-1) for ph in phis])
    return inten, inten / float(four_P)


def observation_grid():
    """The X-strip grid: x = b + 2.5 + 5 k < L and y = y0 + 5 k <= L with y0 = Y_GRID_START_NM, so that the cut rows
    b + CUT_OFFSETS_NM are grid rows (b = 1540.5: x 1543 to 2498 nm, 192 columns, y = 0, 5, ..., 2500 nm, 501 rows).
    """
    x = np.arange(cc.B_NM + GRID_OFFSET_NM, cc.L_NM, GRID_DX_NM)
    y = np.arange(Y_GRID_START_NM, cc.L_NM + 1e-9, GRID_DX_NM)
    return x, y


def square_axis():
    """The square (b, L)^2 of the mirrored display at the same spacing: both axes b + 2.5 + 5 k (192 values)."""
    return observation_grid()[0]


def sample(fam, tables, tab, mesh_vol, mesh_npz, psi):
    """Sample vector fields on the physical strip and the separate square display grid."""
    x, y = observation_grid()
    X, Y = np.meshgrid(x, y)
    z0 = cc.Z0_NM
    Psi, Px, Py = evaluate_psi(tab, mesh_vol, mesh_npz, psi, X, Y)
    E, H = reconstruct(fam, tables, Psi, Px, Py, X, Y, z0)
    roi = (X > cc.B_NM) & (X < cc.L_NM)
    water = cc.water_mask(X, Y, z0)
    arrays = {
        "x_axis_nm": x,
        "y_axis_nm": y,
        "X_nm": X,
        "Y_nm": Y,
        "z0_nm": np.array(z0),
        "Psi": Psi,
        "Px_per_nm": Px,
        "Py_per_nm": Py,
        "E_spp_rom": E,
        "H_rom": H,
        "roi_mask": roi,
        "water_mask": water,
        "extra_heights_nm": np.array(EXTRA_HEIGHTS_NM),
    }
    for zh in EXTRA_HEIGHTS_NM:
        Eh, _ = reconstruct(fam, tables, Psi, Px, Py, X, Y, zh)
        wh = cc.water_mask(X, Y, zh)
        Eh[~wh] = 0.0
        key = "z%d" % int(round(zh))
        arrays["E_spp_rom_" + key] = Eh
        arrays["water_mask_" + key] = wh
    u = square_axis()
    XS, YS = np.meshgrid(u, u)
    Psi_s, Px_s, Py_s = evaluate_psi(tab, mesh_vol, mesh_npz, psi, XS, YS)
    E_s, H_s = reconstruct(fam, tables, Psi_s, Px_s, Py_s, XS, YS, z0)
    arrays.update(
        {
            "square_axis_nm": u,
            "XS_nm": XS,
            "YS_nm": YS,
            "Psi_square": Psi_s,
            "Px_square_per_nm": Px_s,
            "Py_square_per_nm": Py_s,
            "E_spp_rom_square": E_s,
            "H_rom_square": H_s,
            "roi_mask_square": (XS > cc.B_NM) & (XS < cc.L_NM),
            "water_mask_square": cc.water_mask(XS, YS, z0),
        }
    )
    return arrays


def maps(fam, fields, rec):
    """Combine the corner launch with the matched top reference at three phases."""
    X, Y, z0 = fields["X_nm"], fields["Y_nm"], float(fields["z0_nm"])
    E = fields["E_spp_rom"]
    P = float(rec["P_V2_per_m2"])
    four_P = 4.0 * P
    E_ref = cc.reference_ideal_z0(X, fam, rec)
    I_x, J_x = phase_maps(E, E_ref, PHASES_RAD, four_P)
    E_ideal = cc.ideal_spp_x(X, z0, fam, rec["A0_A_per_m"])
    I_id, J_id = phase_maps(E_ideal, E_ref, PHASES_RAD, four_P)
    delta_J = (I_x - I_id) / four_P
    D_E = np.linalg.norm(E - E_ideal, axis=-1) / math.sqrt(P)
    roi, water = fields["roi_mask"].astype(bool), fields["water_mask"].astype(bool)
    valid = roi & water
    y_axis = fields["y_axis_nm"]
    rows = []
    for c in Y_CUTS_NM:
        k = np.flatnonzero(y_axis == c)
        rows.append(int(k[0]))
    rows = np.array(rows, dtype=np.int64)
    arrays = {
        "phi_rad": np.array(PHASES_RAD),
        "P_V2_per_m2": np.array(P),
        "four_P_V2_per_m2": np.array(four_P),
        "x_axis_nm": fields["x_axis_nm"],
        "y_axis_nm": y_axis,
        "X_nm": X,
        "Y_nm": Y,
        "z0_nm": np.array(z0),
        "E_ref": E_ref,
        "E_spp_rom": E,
        "E_spp_ideal": E_ideal,
        "I_x": I_x,
        "J_x": J_x,
        "I_ideal": I_id,
        "J_ideal": J_id,
        "Delta_J": delta_J,
        "D_E": D_E,
        "roi_mask": roi,
        "water_mask": water,
        "valid_mask": valid,
        "cuts_y_nm": np.array(Y_CUTS_NM),
        "cuts_row": rows,
        "cuts_E_rom": E[rows],
        "cuts_E_ideal": E_ideal[rows],
        "cuts_E_ref": E_ref[rows],
        "cuts_water_mask": water[rows],
    }
    return arrays


def mirrored_maps(fam, fields, rec):
    u = fields["square_axis_nm"]
    X, Y = fields["XS_nm"], fields["YS_nm"]
    four_P = 4.0 * rec["P_V2_per_m2"]
    ref = cc.reference_ideal_z0(X, fam, rec)
    I, J = phase_maps(fields["E_spp_rom_square"], ref, PHASES_RAD, four_P)
    valid = fields["roi_mask_square"] & fields["water_mask_square"]
    lower = Y <= X
    valid_display = np.where(lower, valid, valid.T)
    display = np.where(lower[None], I, I.transpose(0, 2, 1))
    display = np.where(valid_display[None], display, 0.0)
    return {
        "square_axis_nm": u,
        "XS_nm": X,
        "YS_nm": Y,
        "I_x_square": I,
        "J_x_square": J,
        "valid_x_square": valid,
        "I_display": display,
        "J_display": display / four_P,
        "valid_display": valid_display,
        "lower_mask": lower,
        "z0_nm": fields["z0_nm"],
        "phi_rad": np.array(PHASES_RAD),
        "four_P_V2_per_m2": four_P,
    }


def draw(maps, directory):
    """Plot the X-launched reduced field in the water region at the saved three phases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    directory.mkdir(parents=True, exist_ok=True)
    x = (maps["x_axis_nm"] - cc.B_NM) * 1e-3
    y = (maps["y_axis_nm"] - cc.B_NM) * 1e-3
    fig, axes = plt.subplots(1, len(maps["phi_rad"]), figsize=(10, 4), sharey=True, layout="constrained")
    for ax, phi, image in zip(axes, maps["phi_rad"], maps["J_x"]):
        J = np.ma.array(image, mask=~maps["valid_mask"])
        artist = ax.pcolormesh(x, y, J, shading="nearest", norm=LogNorm(1e-5, 2.0), cmap="viridis")
        ax.set(xlabel="x from taper end (µm)", aspect="equal", title=f"Phase {phi:.4g} rad")
    axes[0].set_ylabel("y from taper end (µm)")
    fig.colorbar(artist, ax=list(axes), label="Intensity / (4P)")
    fig.savefig(directory / "phase_maps_x.png", dpi=180)
    plt.close(fig)
