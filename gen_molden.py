'''Generate the molden file from rhf chk files
'''

from pyscf.lib.chkfile import load_mol, load_chkfile_key
from pyscf.tools import molden
import os 

def gen_molden_from_rhfchk(chk_path : str, molden_path : str):
	'''
	Generate molden file for orbitals visualization from rhf chk file.

	Args:
	chk_path : str
		path to the check file
	molden_path : str
		path to the molden file to be saved
	
	Output: 
		None
	'''

	mol = load_mol(chk_path)

	print(mol.nelectron)
	mo_coeff = load_chkfile_key(chk_path, 'scf/mo_coeff')
	mo_energy = load_chkfile_key(chk_path, 'scf/mo_energy')
	mo_occ = load_chkfile_key(chk_path, 'scf/mo_cc')

	with open(molden_path, 'w') as f:
		molden.header(mol=mol, fout=f)
		molden.orbital_coeff(mol=mol, fout=f, mo_coeff=mo_coeff, ene=mo_energy, occ=mo_occ)

def gen_molden_from_caschk(chk_path : str, molden_path : str):
	pass

if __name__ == '__main__':
	base_dir = '/Users/aarodynamic95/projects/Research/metallocene_molden/DZ/nihmferrocene_0_3/'
	chk_path = os.path.join(base_dir, 'rhf.chk')
	molden_path = os.path.join(base_dir, 'RHF_orbs.molden')
	gen_molden_from_rhfchk(chk_path, molden_path)