"""Compute the number of "significant" determinants in a CI expansion up to a
given threshold alpha, i.e. compute smallest N such that 
\sum_{i}^{N} |c_i|^2 >= alpha
"""
import numpy as np
import os

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
	CI_PATH_SPECIES_1 = ''
	CI_PATH_SPECIES_2 = ''
	sum_coeff = get_sum_ovlp(CI_PATH_SPECIES_1, n_det=300)
	n_det = get_sig_dets(CI_PATH_SPECIES_2, sum_coeff)