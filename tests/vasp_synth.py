"""Minimal synthetic ``vasprun.xml`` / ``OUTCAR`` writers for the VASP reader
and ``dataset`` tests.  They reproduce only the blocks the readers parse,
in VASP's own layout (checked against a real VASP 6 GeSn run, whose
excerpt lives in tests/data/vasp_gesn).  Stress is given in kB as VASP
writes it."""

import numpy as np


def make_frames(species_counts, n_steps=3, seed=0, a=10.0, interleave=False):
    """Random but well-formed frames: energies [eV], forces [eV/A],
    stress [kB] with distinct off-diagonals so a component mix-up shows."""
    rng = np.random.default_rng(seed)
    types = [sp for sp, n in species_counts for _ in range(n)]
    if interleave:
        types = types[::2] + types[1::2]
    n = len(types)
    frames = []
    for k in range(n_steps):
        lat = np.eye(3) * a + rng.uniform(-0.3, 0.3, (3, 3))
        frac = rng.uniform(0, 1, (n, 3))
        s = rng.uniform(-50, 50, (3, 3))
        frames.append(
            {
                "lattice": lat,
                "positions": frac @ lat,
                "atom_types": types,
                "energy_free": -4.0 * n + 0.1 * k + 1e-3,
                "energy_wo_entropy": -4.0 * n + 0.1 * k,
                "energy_sigma0": -4.0 * n + 0.1 * k + 5e-4,
                "forces": rng.normal(scale=0.5, size=(n, 3)),
                "stress_kB": 0.5 * (s + s.T),
            }
        )
    return frames


def write_vasprun(path, frames, truncate=False):
    types = frames[0]["atom_types"]
    species = list(dict.fromkeys(types))
    out = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        "<modeling>",
        " <atominfo>",
        f"  <atoms>{len(types):8d} </atoms>",
        f"  <types>{len(species):8d} </types>",
        '  <array name="atoms" >',
        '   <dimension dim="1">ion</dimension>',
        '   <field type="string">element</field>',
        '   <field type="int">atomtype</field>',
        "   <set>",
    ]
    for t in types:
        out.append(f"    <rc><c>{t:<2}</c><c>{species.index(t) + 1:4d}</c></rc>")
    out += ["   </set>", "  </array>", " </atominfo>"]

    def v(rows):
        return [f"   <v>{r[0]:17.8f}{r[1]:17.8f}{r[2]:17.8f} </v>" for r in rows]

    for f in frames:
        frac = f["positions"] @ np.linalg.inv(f["lattice"])
        out += (
            [
                " <calculation>",
                "  <scstep>",
                "   <energy>",
                '    <i name="e_fr_energy">   999.0 </i>',
                '    <i name="e_wo_entrp">   999.0 </i>',
                '    <i name="e_0_energy">   999.0 </i>',
                "   </energy>",
                "  </scstep>",
                "  <structure>",
                "   <crystal>",
                '    <varray name="basis" >',
            ]
            + v(f["lattice"])
            + ["    </varray>", "   </crystal>", '   <varray name="positions" >']
            + v(frac)
            + ["   </varray>", "  </structure>", '  <varray name="forces" >']
            + v(f["forces"])
            + ["  </varray>", '  <varray name="stress" >']
            + v(f["stress_kB"])
            + [
                "  </varray>",
                "  <energy>",
                f'   <i name="e_fr_energy">{f["energy_free"]:18.8f} </i>',
                f'   <i name="e_wo_entrp">{f["energy_wo_entropy"]:18.8f} </i>',
                f'   <i name="e_0_energy">{f["energy_sigma0"]:18.8f} </i>',
                "  </energy>",
                " </calculation>",
            ]
        )
    out.append("</modeling>")
    text = "\n".join(out) + "\n"
    if truncate:  # cut inside the last <calculation>
        text = text[: text.rfind('<varray name="forces"')]
    with open(path, "w") as fh:
        fh.write(text)


def write_outcar(path, frames):
    types = frames[0]["atom_types"]
    # VASP's species blocks are runs of equal labels (POSCAR order)
    blocks = []
    for t in types:
        if blocks and blocks[-1][0] == t:
            blocks[-1][1] += 1
        else:
            blocks.append([t, 1])
    out = []
    for sp, _ in blocks:
        out += [f"   VRHFIN ={sp}: s2p2", f"   TITEL  = PAW_PBE {sp} 05Jan2001"]
    out.append("   ions per type =  " + "".join(f"{n:5d}" for _, n in blocks))
    for f in frames:
        s = f["stress_kB"]
        out += [
            "  FORCE on cell =-STRESS in cart. coord.  units (eV):",
            "  Total   0.0 0.0 0.0 0.0 0.0 0.0",
            "  in kB  "
            + "".join(
                f"{x:12.5f}"
                for x in (s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[2, 0])
            ),
            "  external pressure =  0.0 kB  Pullay stress =  0.0 kB",
            "",
            " VOLUME and BASIS-vectors are now :",
            " " + "-" * 60,
            "  energy-cutoff  :      500.00",
            f"  volume of cell : {abs(np.linalg.det(f['lattice'])):11.2f}",
            "      direct lattice vectors                 reciprocal lattice vectors",
        ]
        rec = np.linalg.inv(f["lattice"]).T
        for r, q in zip(f["lattice"], rec):
            out.append(
                "   "
                + "".join(f"{x:13.9f}" for x in r)
                + "   "
                + "".join(f"{x:13.9f}" for x in q)
            )
        out += [
            "",
            " POSITION                                       TOTAL-FORCE (eV/Angst)",
            " " + "-" * 80,
        ]
        for p, fr in zip(f["positions"], f["forces"]):
            out.append(
                "".join(f"{x:13.5f}" for x in p)
                + "    "
                + "".join(f"{x:14.6f}" for x in fr)
            )
        out += [
            " " + "-" * 80,
            "",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            "  ---------------------------------------------------",
            f"  free  energy   TOTEN  = {f['energy_free']:18.8f} eV",
            "",
            f"  energy  without entropy= {f['energy_wo_entropy']:16.8f}"
            f"  energy(sigma->0) = {f['energy_sigma0']:16.8f}",
            "",
        ]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def write_run(run_dir, frames):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_vasprun(run_dir / "vasprun.xml", frames)
    write_outcar(run_dir / "OUTCAR", frames)
