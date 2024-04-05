"""Generate a cube file from a chkfile
"""

import numpy as np
import os
from pyscf.tools.cubegen import orbital
from pyscf.lib.chkfile import load_mol, load_chkfile_key

def gen_cube_cas_wrapper(chkfile : str, cube_dir="./cas_cube"):
	# load the chkfile
	mol = load_mol(chkfile=chkfile)
	mo_coeff = load_chkfile_key(chkfile, 'mcscf/mo_coeff')
	ncore = load_chkfile_key(chkfile, 'mcscf/ncore')
	ncas = load_chkfile_key(chkfile, 'mcscf/ncas')
	print(f"Loading CASSCF file with {ncas} active orbitals")
	print(f"Generating CAS orbs from idx {ncore} to {ncas}.)")
	if os.path.exists(cube_dir) == False:
		os.mkdir(cube_dir)
	for orb_idx in np.arange(ncore, ncore + ncas):
		#avas_init_file = os.path.join(base, 'avas_init_orbs', f'avas_mo_{orb_idx+1}.cube')
		#orbital(mol, avas_init_file, avas_init_mos[:, orb_idx])
		cas_orbs_file = os.path.join(cube_dir, f'cas_mo_{orb_idx}.cube')
		orbital(mol, cas_orbs_file, mo_coeff[:, orb_idx])
	return None


def gen_cube_rhf_wrapper(chkfile : str, cube_dir="./rhf_cube", start=0, stop=100):
	# load the chkfile
	assert(stop > 0)
	assert(start > 0)
	print(f"Generating {start - stop + 1} mo cubes, idx {start} to {stop} (0-based).")
	mol = load_mol(chkfile=chkfile)
	mo_coeff = load_chkfile_key(chkfile, 'scf/mo_coeff')
	if os.path.exists(cube_dir) == False:
		os.mkdir(cube_dir)
	for orb_idx in np.arange(start, stop):
		#avas_init_file = os.path.join(base, 'avas_init_orbs', f'avas_mo_{orb_idx+1}.cube')
		#orbital(mol, avas_init_file, avas_init_mos[:, orb_idx])
		orbs_file = os.path.join(cube_dir, f'rhf_mo_{orb_idx}.cube')
		orbital(mol, orbs_file, mo_coeff[:, orb_idx])
	return None

if __name__ == "__main__":
	# cas
	cas_chk = "./casscf.chk"
	gen_cube_cas_wrapper(cas_chk)
	# rhf
	rhf_chk = "./rhf.chk"
	gen_cube_rhf_wrapper(rhf_chk, start=0, stop=1)
	