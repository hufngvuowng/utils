"""Functions dealing with analyzing and extracting information from a multi-
determinant wavefuncton (orthgonal).
"""
import numpy as np
import os

def parse_ci_coeff(ci_coeff_file, ncas_a, ncas_b, ncore=0, ndet=0):
	'''Parse a CI_coeff.dat file, for example:
	```
	0b111111111111       0b11111111111          0.6208214919329170
	0b1000000000011111111111 0b11111111111          0.3399556625020033
	```
	, where the columns are alpha, beta config and the CI coefficients. Ignoring '0b',
	1 indicates occupied orbital, 0 indicates unoccupied orbitals. Configurations 
	are read from right to left (aka the first orbital to the right is index ncore
	in 0-based convention.) 

	Parameters
	----------
	ci_coeff_file : str
		Path to the CI_coeff.dat file.
	ncas_a : int
		Number of active occupied alpha orbitals
	ncas_b : int
		Number of active occupied beta orbitals.
	ncore : int, optional
		Number of core orbitals (the doubly occupied orbitals that are not included
		in the active space). Default 0.
	
	Returns
	--------
	ci_coeffs: numpy.ndarray
	 	Array of size [n_det], where n_det is the number of configs, storing the 
		CI expansion coefficients.
	alpha_config: numpy.ndarray
		Array of size [max_excit_rank, n_det], where max_excit_rank is the maxium
		number of occupied orbitals excited in the CI expansion, and n_det is
		the total number of configurations.
	beta_config : numpy.ndarray
		Same as alpha but for beta.
	'''

	# Helper
	def process_raw_config(config, ncas_occ, ncore=0):
		'''Convert raw config of the form '111101' to an array of MO indices occupied
		in the config.

		Parameters
		----------
		config : str
			The configuration being processed, of the form '110011' for example.
			Note that the MO is ordered from right to left.
		ncas_occ : int
			Number of occupied, active orbitals.
		ncore : int, optional.
			Number of core orbitals. Default 0.

		Returns
		-------
		config_array : list
			Array storing the indices of the MOs being occupied in the current 
			configuration.
		excit_rank : int
			The current excitation rank 
		'''
		# flip the config and split into a list
		config_list = list(config[::-1])
		# Get the excitation rank
		# Count the number of zeros appearing in the first `ncas_occ` inx in the 
		# config_list
		excit_rank = config_list[:ncas_occ].count('0')
		#print(f'Current config: {config[::-1]}, with rank {excit_rank}')
		config_array = []
		for idx in np.arange(len(config_list)):
			occ = config_list[idx]
			if occ == '1':
				config_array.append(ncore+idx)
		return config_array, excit_rank
	 
	ci_coeffs = []
	alpha_raw_config = []
	beta_raw_config = []
	with open(ci_coeff_file, 'r') as cif:
		counter = 0
		for lines in cif.readlines():
			word_list = lines.strip().split(' ')
			word_list = [i for i in word_list if i != '']
			assert(len(word_list) == 3)
			# CI coeffs are in the last column
			ci_coeffs.append(float(word_list[2]))
			# alpha config, dropping the '0b'
			alpha_raw_config.append(word_list[0][2:])
			beta_raw_config.append(word_list[1][2:])
			counter += 1
			if ndet > 0 :
				if counter >= ndet:
					break
	
	num_config = len(ci_coeffs) # total number of dets
	excit_rank = np.zeros(num_config, dtype=np.int32)
	for idx in range(num_config):
		_, rank_a = process_raw_config(alpha_raw_config[idx], ncas_a, ncore)
		_, rank_b = process_raw_config(beta_raw_config[idx], ncas_b, ncore)
		excit_rank[idx] = rank_a + rank_b
	# Get the "unique" configurations
	alpha_unique_configs = np.unique(alpha_raw_config)
	print(f"In {num_config} configs, there are {len(alpha_unique_configs)} unique excitation for alpha.")
	beta_unique_configs = np.unique(beta_raw_config)
	print(f"In {num_config} configs, there are {len(beta_unique_configs)} unique excitation for beta.")
	num_unique_configs = [len(alpha_unique_configs), len(beta_unique_configs)]

	excit_rank_a = np.zeros(num_unique_configs[0], dtype=np.int32)
	excit_rank_b = np.zeros(num_unique_configs[0], dtype=np.int32)
	alpha_config = np.zeros((num_unique_configs[0], ncas_a), dtype=np.int32)
	beta_config = np.zeros((num_unique_configs[0], ncas_b), dtype=np.int32)
	# mapping from the actual to the unique config
	alpha_map = np.zeros(num_config, dtype=np.int32)
	beta_map = np.zeros(num_config, dtype=np.int32)
	# Now parse the unique configs to get the mo indices
	# Alpha
	for config_idx in np.arange(num_unique_configs[0]):
		a_array, excit_a = process_raw_config(alpha_unique_configs[config_idx], ncas_a, 
										ncore)
		excit_rank_a[config_idx] = excit_a
		alpha_config[config_idx, :] = a_array
	# Beta_section
	for config_idx in np.arange(num_unique_configs[1]):
		b_array, excit_b = process_raw_config(beta_unique_configs[config_idx], ncas_b, 
										ncore)
		excit_rank_b[config_idx] = excit_b
		beta_config[config_idx, :] = b_array
	max_excit_a = np.max(excit_rank_a)
	max_excit_b = np.max(excit_rank_b)

	# Get the mapping from the actual determinants to the unique values arrays
	for idx in np.arange(num_config):
		# Alpha
		config = alpha_raw_config[idx]
		unq_idx = np.where(alpha_unique_configs == config)[0][0]
		alpha_map[idx] = unq_idx
		#print(f'det {idx} with alpha config = {config}, the idx in the unique array is {unq_idx}')
		# Beta 
		config = beta_raw_config[idx]
		unq_idx = np.where(beta_unique_configs == config)[0][0]
		beta_map[idx] = unq_idx
		if ( idx > 0 and int(idx) % (int(num_config/10)) == 0 ):
			print(f"Done parsing {idx/num_config*100:.2f} %")
		#print(f'det {idx} with beta config = {config}, the idx in the unique array is {unq_idx}')
		#print(f"The config in unique config is {alpha_unique_configs[unq_idx]}")
	
	# Get the maximum excitation rank among ALL the configs
	print(f"The max excitation rank is {max_excit_a} for alpha and {max_excit_b} for beta section.")
	return np.array(ci_coeffs), alpha_config, beta_config, alpha_map, beta_map, alpha_unique_configs, beta_unique_configs, excit_rank

def get_sig_dets(ci_coeff_file, thres=0.995):
	'''Compute the number of determinants whose sum of CI percentage adds up to
	a given threshold in a (sorted) CI expansion, i.e. return smallest N such that 
	\sum_{i}^{N} |c_i|^2 >= thres. If all the dets in the expansion do not add up,
	return the total number of determinants in the given expansion instead.

	Parameters
	----------
	ci_coeff_file : str
		Path to the CI_coeff.dat file.
	thres : float, optional. Default 0.995
		Threshold for the sum of squares of CI coeff. Must be larger than 0.0

	Return
	------
	n_det : int 
		Number of determinants that add up to the threshold.
	'''
	assert(thres >= 0.0)
	n_det = 0
	sum_coeff = 0.0
	with open(ci_coeff_file, 'r') as f:
		for line in f.readlines():
			n_det += 1
			coeff = line.strip().split(' ')[-1]
			sum_coeff += float(coeff)**2
			if sum_coeff >= thres:
				print(f'Number of dets {n_det} with total ovlp square {sum_coeff}')
				return n_det
	print("The dets file does not contain enough dets to sum to threshold. Returning the total number of dets in file")
	return n_det

def get_sum_ovlp(ci_coeff_file, n_det=1):
	'''Return the sum of squares of coefficients of a given number of determinants
	in a sorted (descending) CI expansion. If the input number of dets is larger
	than the number of dets in the CI file, return the sum of squares of all coeffs
	in the CI expansion instead.

	Parameters
	----------
	ci_coeff_file : str
		Path to the CI_coeff.dat file
	n_det : int, optional. Default 1.
		Number of determinants to compute the sum of CI squares of. Must be 
		greater than 0

	Return
	------
	sum_coeff : float
		Sum of squares of the first `n_det` CI coeffs.
	'''
	assert(n_det>0) 
	sum_coeff = 0.0
	cur_ndets = 0
	with open(ci_coeff_file, 'r') as f:
		for line in f.readlines():
			cur_ndets += 1
			if cur_ndets > n_det:
				print(f'n_det = {n_det}, sum ovlp squared = {sum_coeff}')
				return sum_coeff
			else: 
				coeff = line.strip().split(' ')[-1]
				sum_coeff += float(coeff)**2
	
	print('Input number of determinants is larger than the number of dets contained in file\n')
	print('Returning the total sum of coeff squares in the files\n')
	return sum_coeff

if __name__ == '__main__':
	CI_PATH_SPECIES = '../scratch/inputs/cu2o2/f0/hciscf-32e33o/tight_hci/CI_coeff.dat'
	ndet = 100000
	sum_coeff = get_sum_ovlp(CI_PATH_SPECIES, n_det=ndet)
	#n_det = get_sig_dets(CI_PATH_SPECIES, sum_coeff)
	ci_coeffs, alpha_config, beta_config, alpha_map, beta_map, alpha_unique, beta_unique, excit_rank = parse_ci_coeff(ci_coeff_file=CI_PATH_SPECIES, 
													   ncas_a=16, ncas_b=16, ncore=10,
                                                       ndet=ndet)
	np.savetxt("../scratch/inputs/cu2o2/f0/hciscf-32e33o/tight_hci/excit_rank.csv", excit_rank)
	# Write to file
	#with open('../scratch/outputs/o3-hciscf/unique_alpha_config.dat', 'w') as f:
	#	for i in np.arange(len(alpha_unique)):
	#		f.write(f"{alpha_unique[i]}\n")
	#with open('../scratch/outputs/o3-hciscf/unique_beta_config.dat', 'w') as f:
	#	for i in np.arange(len(beta_unique)):
	#		f.write(f"{beta_unique[i]}\n")
	#with open('../scratch/outputs/o3-hciscf/ci_coeff.dat', 'w') as f:
	#	for i in np.arange(ndet):
	#		f.write(f"{ci_coeffs[i]:24.16f}\t{alpha_map[i]:9d}\t{beta_map[i]:9d}\n")
	

	#np.savetxt('../scratch/outputs/ci_coeff.csv', ci_coeffs, delimiter=',', fmt='%24.16f')
	#np.savetxt('../scratch/outputs/alpha.csv', alpha_config, delimiter=',', fmt='%d')
	#np.savetxt('../scratch/outputs/beta.csv', beta_config, delimiter=',', fmt='%d')
