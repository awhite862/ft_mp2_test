import numpy
from pyscf import gto, scf, fci
import fci_simple
import ft_mp2_simple as ft
from mpmath import mp

def computeG(Z,T):
    return float(-T*mp.log(Z))

mol = gto.M(
    verbose = 0,
    atom = 'Be 0 0 0',
    basis = 'sto-3g',
    spin = 0)

T = 3.1668554445E+03
mu = 0.0
hartree_to_ev = 27.211
kb = 8.617330350e-5 # in eV
Tkelvin = T*hartree_to_ev / (kb)
print('Running FT-MP2 at an electronic temperature of %f' % Tkelvin)

m = scf.RHF(mol)
print(m.scf())

#cisolver = fci.FCI(mol, m.mo_coeff)
#print('E(FCI) = %.12f' % cisolver.kernel()[0])
#
#fci_simple.fci_simple(mol, m)

Z = fci_simple.fci_partition_all(mol, m, T, mu)
Eref = computeG(Z,T)

# Compute free energies
#delta = 5.12e-5
delta = 1.e-1
Z2f = fci_simple.fci_partition_all(mol, m, T, mu,0.0 + 2*delta)
Zf = fci_simple.fci_partition_all(mol, m, T, mu,0.0 + delta)
Zc = fci_simple.fci_partition_all(mol, m, T, mu,0.0)
Zb = fci_simple.fci_partition_all(mol, m, T, mu,0.0 - delta)
Z2b = fci_simple.fci_partition_all(mol, m, T, mu,0.0 - 2*delta)
E2f = computeG(Z2f,T)
Ef = computeG(Zf,T)
Ec = computeG(Zc,T)
Eb = computeG(Zb,T)
E2b = computeG(Z2b,T)

# Compute 0th order free energy
E0r = Ec

# Compute 1st order free energy by finite differences
E1r = (Ef - Eb) / (2.0*delta)

# Compute 2nd order free energy by finite differences
E2r = (Ef - 2.0*Ec + Eb) / (2.0*delta*delta)

En = mol.energy_nuc()
E0,E1 = ft.ftmp1(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T)
E2 = ft.ftmp2(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T)
E3 = 0.0
print("{}  {}  {}".format(E0r,E1r,E2r))
print("{}  {}  {}".format(E0,E1,E2))
