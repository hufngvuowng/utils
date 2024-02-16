import numpy as np
from scipy.linalg import eigh
from pyscf.mp import pt
from pyscf.lib.chkfile import load_mol #, load_chkfile_key
from pyscf import scf

RHF_CHK = 'path/to/rhf/chk'
UHF_CHK = 'path/to/uhf/chk'
FROZEN = 8

# Load the `Mole` class from chkfile
mol = load_mol(RHF_CHK)
# Read in the RHF chkpoint file and runs for 0 cycles
m_rhf = scf.sfx2c(scf.ROHF(mol))
m_rhf.verbose = 6
m_rhf.max_cycle  = 0                                                          
m_rhf.conv_tol   = 1e-8                                                                    
m_rhf.max_memory = 64000
## project the 1RDM from the chkpoint file as initial guess:
dm = m_rhf.init_guess_by_chkfile(RHF_CHK, project=True)
m_rhf.kernel(dm)
rhf_mo_coeff = 

# Read the UHF chkpoint file as initial guess and run for 0 cycles
m_uhf = scf.sfx2c(scf.UHF(mol))
m_uhf.verbose = 6
m_uhf.max_cycle = 0
m_uhf.conv_tol = 1e-8
m_uhf.max_memory = 64000
dm = m_uhf.init_guess_by_chkfile(UHF_CHK, project=True)
m_uhf.kernel(dm)

# Run UMP2
ump2 = pt.UMP2(m_uhf, frozen=FROZEN)
ump2.verbose = 6
ump2.kernel()
dma, dmb = ump2.make_1rdm()
dm = np.asarray(dma) + np.asarray(dmb)
occ_nums, nat_orbs = eigh((-1.0)*dm)                                  
mp2_natorbs = np.matmul(m_rhf.mo_coeff, nat_orbs)
print("Natural orbital occupation numbers of MP2")                                 
print(-occ_nums)                                                                   
total = np.sum(-occ_nums)                                                       
print("total NOONs = {}".format(total))                                            
np.savetxt("MP2_orbs.csv", mp2_natorbs, fmt='%24.16f', delimiter=',')           
np.savetxt('noons.csv', -occ_nums,  fmt='%24.16f', delimiter=',')   