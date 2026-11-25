"""Physical constants (SI, from scipy.constants / CODATA) and atomic masses."""

from scipy import constants as _c

E_CHARGE = _c.e  # C
EPS0 = _c.epsilon_0  # F/m
KB = _c.k  # J/K
AMU = _c.atomic_mass  # kg
ANGSTROM = 1.0e-10  # m
FS = 1.0e-15  # s
THZ = 1.0e12  # Hz
CM1_TO_HZ = _c.c * 100.0  # Hz per cm^-1
MEV_TO_HZ = 1.0e-3 * _c.e / _c.h  # Hz per meV
EV = _c.e  # J per eV
C_CM = _c.c * 100.0  # speed of light [cm/s]
# 1 Debye = 1e-21 / c C m (exact definition, 3.33564e-30 C m) in e * Angstrom (0.2081943)
DEBYE_TO_E_ANG = 1.0e-21 / _c.c / (_c.e * ANGSTROM)

# Standard atomic weights [amu]; same values as akitsushima's ATOMIC_MASS
# table so that f-sum normalisation of akitsushima runs uses the engine's
# masses (ADR-0002 item 5).  VASP runs take POMASS from the OUTCAR instead.
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
}  # fmt: skip
