"""
====================================================================
GaAs Electronic Structure Calculation Workflow using ASE + QE
====================================================================

Author  : Student Agent
Date    : July 20, 2025
Version : 1.0
Language: Python 3.x

Description:
------------
This script provides a modular workflow for performing first-principles
calculations of the band structure, total density of states (DOS), and
projected density of states (PDOS) of gallium arsenide (GaAs) using 
Quantum ESPRESSO via the ASE (Atomic Simulation Environment) interface.

Key features:
- SCF, NSCF, and band structure calculation
- DOS and PDOS post-processing
- Pseudopotential-based DFT input using PBE functionals
- Bandpath definition based on high-symmetry points

Modules:
--------
- create_structure(): Generate bulk GaAs structure
- run_scf(): Perform SCF calculation
- run_band_structure(): Compute band energies along high-symmetry path
- run_nscf(): Perform NSCF calculation for DOS
- run_dos(): Run DOS post-processing via `dos.x`
- run_pdos(): Run PDOS post-processing via `projwfc.x`

Requirements:
-------------
- ASE
- Quantum ESPRESSO (pw.x, dos.x, projwfc.x)
- NumPy
- Pseudopotentials from PSLibrary

"""

# file: gaas_workflow.py

import os
import numpy as np
from ase.build import bulk
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.dft.kpoints import bandpath
import xml.etree.ElementTree as ET

def create_structure():
    return bulk('GaAs', crystalstructure='zincblende', a=5.65)

def run_scf(atoms, profile):
    scf_calc = Espresso(
        profile=profile,
        input_data={
            'control': {'calculation': 'scf', 'prefix': 'gaas', 'outdir': './out'},
            'system': {
                'ecutwfc': 40, 'ecutrho': 320,
                'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.01
            },
            'electrons': {'conv_thr': 1e-6}
        },
        pseudopotentials={
            'Ga': 'Ga.pbe-dn-rrkjus_psl.1.0.0.UPF',
            'As': 'As.pbe-n-rrkjus_psl.1.0.0.UPF'
        },
        kpts=(8, 8, 8)
    )
    atoms.calc = scf_calc
    energy = atoms.get_potential_energy()
    print(f"[SCF] Total energy: {energy:.6f} eV")

def run_band_structure(atoms, profile):
    path = bandpath('GXWKGLUWLK', atoms.cell, npoints=100)
    kpts_bands = np.zeros((len(path.kpts), 4))
    kpts_bands[:, :3] = path.kpts

    bands_calc = Espresso(
        profile=profile,
        input_data={
            'control': {'calculation': 'bands', 'prefix': 'gaas', 'outdir': './out'},
            'system': {
                'ecutwfc': 40, 'ecutrho': 320,
                'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.01
            },
            'electrons': {'conv_thr': 1e-6}
        },
        pseudopotentials={
            'Ga': 'Ga.pbe-dn-rrkjus_psl.1.0.0.UPF',
            'As': 'As.pbe-n-rrkjus_psl.1.0.0.UPF'
        },
        kpts=kpts_bands,
        koffset=False
    )
    atoms.calc = bands_calc
    atoms.calc.calculate(atoms, properties=['bands'], system_changes=['cell'])
    print("[Bands] Band structure calculation completed.")

def run_nscf(atoms, profile):
    nscf_calc = Espresso(
        profile=profile,
        input_data={
            'control': {'calculation': 'nscf', 'prefix': 'gaas', 'outdir': './out'},
            'system': {
                'ecutwfc': 40, 'ecutrho': 320,
                'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.01
            },
            'electrons': {'conv_thr': 1e-6}
        },
        pseudopotentials={
            'Ga': 'Ga.pbe-dn-rrkjus_psl.1.0.0.UPF',
            'As': 'As.pbe-n-rrkjus_psl.1.0.0.UPF'
        },
        kpts=(12, 12, 12)
    )
    atoms.calc = nscf_calc
    atoms.calc.calculate(atoms, properties=['energy'], system_changes=['cell'])
    print("[NSCF] NSCF calculation completed.")

def run_dos():
    with open('dos.in', 'w') as f:
        f.write("""
&dos
  outdir = './out',
  prefix = 'gaas',
  fildos = 'gaas.dos',
  emin = -15.0,
  emax = 15.0,
  deltae = 0.01,
/
""")
    os.system('dos.x < dos.in > dos.out')
    print("[DOS] DOS calculation completed.")

def run_pdos():
    with open('projwfc.in', 'w') as f:
        f.write("""
&projwfc
  outdir = './out',
  prefix = 'gaas',
  filpdos = 'gaas.pdos',
  emin = -15.0,
  emax = 15.0,
  deltae = 0.01,
  ngauss = 0,
  degauss = 0.01,
/
""")
    os.system('projwfc.x < projwfc.in > projwfc.out')
    print("[PDOS] PDOS calculation completed.")

def main():
    pseudo_dir = '/home/xqhan/InvDesAgents/QE/qe-7.4.1/pseudo'
    profile = EspressoProfile(command='pw.x', pseudo_dir=pseudo_dir)
    atoms = create_structure()

    run_scf(atoms, profile)
    run_band_structure(atoms, profile)
    run_nscf(atoms, profile)
    run_dos()
    run_pdos()

if __name__ == "__main__":
    main()

def run_all(profile, output_dir='./out'):
    atoms = bulk('GaAs', crystalstructure='zincblende', a=5.65)
    run_scf(atoms, profile, output_dir)
    run_band_structure(atoms, profile, output_dir)
    run_nscf(atoms, profile, output_dir)
    run_dos(output_dir)
    run_pdos(output_dir)

# Example usage from another script:
# from gaas_workflow import run_all
# profile = EspressoProfile(command='pw.x', pseudo_dir='...')
# run_all(profile)

