import numpy as np
import os 

def parse_ci_dat(ci_file_path):
	with open(ci_file_path) as f:
		config_list = []
		for l in f.readlines():
			l_list = l.strip().split(' ')
			l_list = [i for i in l_list if i != '']
			assert(len(l_list) == 3)
			config_list.append(l_list)
	print(f"Total number of configs : {len(config_list)}")
	return config_list

def compute_occupation(config_list, nact_orbs):
	occupations = np.zeros(nact_orbs)
	for idx in range(len(config_list)):
		config = config_list[idx]
		a_config = list(config[0][2:])
		a_config.reverse()
		b_config = list(config[1][2:])
		b_config.reverse()
		ci_coeff = float(config[2])
		a_config_occ = np.zeros(nact_orbs)
		b_config_occ = np.zeros(nact_orbs)
		for orb_idx in range(nact_orbs):
			# A
			if orb_idx < len(a_config):
				a_config_occ[orb_idx] += ci_coeff**2*int(a_config[orb_idx])
			else:
				a_config_occ[orb_idx] += 0
			
			# B
			if orb_idx < len(b_config):
				b_config_occ[orb_idx] += ci_coeff**2*int(b_config[orb_idx])
			else:
				b_config_occ[orb_idx] += 0
			
			occupations[orb_idx] += a_config_occ[orb_idx] + b_config_occ[orb_idx]
			return occupations

def check_new_as_orbs(config_list, ncore_old, ncore_new, nact_old, ndet):
	for idx in range(ndet):
		config = config_list[idx]
		a_config = list(config[0][2:])
		a_config.reverse()
		b_config = list(config[1][2:])
		b_config.reverse()
		ci_coeff = float(config[2])
		for i in range(len(a_config)):
			if int(a_config[i]) == 1:
				if ncore_new + i >= ncore_old + nact_old:
					print(f"Config {idx} promotes an alpha electron to orb_idx {ncore_new + i} outside the old AS!")
		
		for i in range(len(b_config)):
			if int(b_config[i]) == 1:
				if ncore_new + i >= ncore_old + nact_old:
					print(f"Config {idx} promotes an beta electron to orb_idx {ncore_new + i} outside the old AS!")




if __name__ == "__main__":
	CI_file = "/Users/aarodynamic95/projects/Research/3dTMV/MR_CASSCF/28/init/large/1e-3/CI_coeff.dat"
	config = parse_ci_dat(CI_file)
	check_new_as_orbs(config_list=config, ncore_old=64, nact_old=45, ncore_new=55, ndet=24)