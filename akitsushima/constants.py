"""Physical constants (eV / A / amu / fs unit system) and atomic masses."""
import math

EV_PER_JOULE = 6.241509074e18          # J  -> eV
ANGSTROM = 1.0e-10                     # m
K_E_SI = 8.9875517923e9                # N m^2 C^-2
ELECTRON_CHARGE = 1.60217663e-19       # C
K_B_SI = 1.380649e-23                  # J K^-1

# k_e . e^2 in eV.A
K_E = K_E_SI * ELECTRON_CHARGE**2 / ANGSTROM * EV_PER_JOULE   # ~= 14.3996 eV.A
# Boltzmann constant in eV K^-1
K_B = K_B_SI * EV_PER_JOULE                                    # ~= 8.6173e-5 eV K^-1

# Unit conversion: eV/A -> amu.A/fs^2
# F [eV/A] = m [amu] . a [A/fs^2]
# 1 eV = 1.60218e-19 J ; 1 amu = 1.66054e-27 kg ; 1 fs = 1e-15 s
# a [m/s^2] = F[N] / m[kg] = F[eV/A] * 1.60218e-19/(1e-10) / (m[amu]*1.66054e-27)
# a [A/fs^2] = a[m/s^2] * 1e-10 / (1e-15)^2 = a[m/s^2] * 1e20
EV_ANG_TO_AMU_ANG_FS2 = (1.60217663e-19 / 1.0e-10) / (1.66053906660e-27) * 1.0e-20
# ~= 9.6485e-3  (i.e. 1 eV/A ~= 0.009649 amu.A/fs^2)

# Speed of light in vacuum, exact by the SI definition of the metre:
# c = 299792458 m/s = 2.99792458e10 cm/s, and 1 fs = 1e-15 s.
SPEED_OF_LIGHT_CM_S  = 2.99792458e10                  # cm s^-1
SPEED_OF_LIGHT_CM_FS = SPEED_OF_LIGHT_CM_S * 1.0e-15  # cm fs^-1 (= 2.99792458e-5)

# Angular frequency [rad/fs] -> wavenumber [cm^-1]:  nu = omega / (2*pi*c),
# with c = 2.99792458e-5 cm/fs. Shared by
# TransitionStateAnalyser.analyse_modes() and
# PhononCalculator._eig_to_freq(), both of which convert a Hessian/IFC
# eigenvalue lambda [eV/A^2] (or its mass-weighted, per-amu form) to a
# vibrational frequency via omega^2 = lambda * EV_ANG_TO_AMU_ANG_FS2,
# and by DiffusionAnalyzer.compute_vacf_dos() for the vibrational DOS axis.
#
# Bug-fix note: this constant previously used the rounded c = 3e-5 cm/fs,
# evaluating to 5305.1648 instead of the correct 5308.8375 -- so EVERY
# wavenumber this package reported was systematically 0.069% too low.  A
# rounding slip rather than a deliberate choice: thermal_conductivity.py
# and tsa.py's Wigner correction already used the exact 2.99792458e10 cm/s,
# the latter applying it to the very frequencies_cm produced with the
# rounded value here.
FREQ_RAD_FS_TO_WAVENUMBER_CM = 1.0 / (2.0 * math.pi * SPEED_OF_LIGHT_CM_FS)
# ~= 5308.837458876 cm^-1 per (rad/fs)

# Stress/pressure unit conversion.  1 eV/A^3 = 160.2176621 GPa (from
# 1 eV = 1.602176634e-19 J and 1 A^3 = 1e-30 m^3: 1 eV/A^3 =
# 1.602176634e11 Pa = 160.2176634 GPa, rounded to the precision used
# throughout this codebase).  1 GPa = 10 kbar exactly (1 bar = 1e5 Pa),
# so 1 eV/A^3 = 1602.176621 kbar.
EV_ANG3_TO_GPA  = 160.2176621           # eV/A^3 -> GPa
EV_ANG3_TO_KBAR = EV_ANG3_TO_GPA * 10.0  # eV/A^3 -> kbar (VASP OUTCAR "kB")

# VASP OUTCAR "in kB" -> this codebase's stress, in eV/A^3.
#
# NO SIGN FLIP.  This codebase's stress is sigma_ab = -(1/V) dE/dT_ab,
# which is POSITIVE for a compressed cell (docs/adr/0001, CONTEXT.md
# "Compressive-positive").  VASP's raw "in kB" line has that same sign:
# it is positive under compression.  ASE's convention is the opposite
# one -- ase.calculators.vasp.vasp::read_stress negates the raw line
# (``stress = -np.array([...]) * 1e-1 * ase.units.GPa``) to get it --
# so the chain is
#
#     ASE stress           = -raw_kB / 1602.176621        (read_stress)
#     akitsushima stress   = -ASE stress                  (measured)
#     =>  akitsushima stress = +raw_kB / 1602.176621
#
# where "measured" is ASE's own LennardJones calculator against this
# package's LennardJonesPotential on the same compressed and expanded
# FCC cells: ratio -1.000000 in both, magnitudes equal to 7 digits
# (tests/test_io.py pins the whole chain against ASE's production code).
#
# Bug-fix note: this constant was -1/1602.176621.  It was derived from
# the premise that this codebase is "tensile-positive" in ASE's sense,
# which is a misnomer for the convention it actually uses (see
# CONTEXT.md) -- the same indirect-from-VASP reasoning that ADR-0001
# records as having wrongly predicted a sign flip in the 'vasp' cell
# update scheme.  Training data built with it carried stresses of the
# opposite sign to what the engine predicts for the same structure, so
# a stress-weighted fit was pulled toward the mirror-image answer.  The
# regression test that accompanied it could not see this: it built its
# "raw VASP" input as target / CONSTANT and checked that multiplying by
# CONSTANT gave target back, which holds for either sign.
#
# Component ORDER is a separate trap: VASP writes xx yy zz xy yz zx,
# whereas akitsushima's training CSV expects Voigt xx yy zz yz zx xy.
# Reorder raw VASP values with [0, 1, 2, 4, 5, 3] (ASE's own
# read_stress does exactly this).
VASP_KB_TO_EV_ANG3 = +1.0 / EV_ANG3_TO_KBAR

# Atomic masses [amu] for common elements
ATOMIC_MASS = {
    "H": 1.008, "He": 4.003,
    "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.96, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91,
    "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93,
    "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23,
    "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0, "Rn": 222.0,
}
