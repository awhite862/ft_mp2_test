import numpy
from pyscf import gto, scf, fci
import fci_simple
import ft_mp2_simple as ft
import ft_mp3_simple as ft3
import ccs
from mpmath import mp

def computeG(Z,T):
    return -T*(mp.log(Z))

mp.dps = 100
mol = gto.M(
    verbose = 0,
    atom = 'H 0 0 0; F 0 0 0.9168',
    basis = 'sto-3g',
    spin = 0)

Ts = [
8.6173303500E-02,
8.6173303500E-01,
8.6173303500E+00,
8.6173303500E+01,
8.6173303500E+02,
8.6173303500E+03,
8.6173303500E+04]

mus = [
1.1745322971E-01,
8.6905852849E-01,
1.1627894307E+01,
1.3432362884E+02,
1.3822313736E+03,
1.3864352388E+04,
1.3868587277E+05]

#deltas = [
#1E-04,
#2E-04,
#4E-04,
#8E-04,
#2.0E-03,
#2.0E-03,
#2.0E-03]

for i in range(6,7):
    T = Ts[i]
    mu = mus[i]
    #hartree_to_ev = 27.211
    #kb = 8.617330350e-5
    #Tkelvin = T*hartree_to_ev / (kb)
    #print('Running FT-MP2 at an electronic temperature of %f' % Tkelvin)
    
    m = scf.RHF(mol)
    Escf = m.scf()
    
    #cisolver = fci.FCI(mol, m.mo_coeff)
    #print('E(FCI) = %.12f' % cisolver.kernel()[0])
    #
    #fci_simple.fci_simple(mol, m)
    
    Z = fci_simple.fci_partition_all(mol, m, T, mu)
    Eref = computeG(Z,T)
    
    # Compute free energies
    delta = 1e1
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
    
    # Compute 3rd order free energy by finite differences
    E3r = (E2f - 2.0*Ef + 2.0*Eb - E2b) / (12*delta*delta*delta)
    
    En = mol.energy_nuc()
    E0,E1 = ft.ftmp1(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T)
    E2 = ft.ftmp2(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T,2*T)
    #E3 = ft3.ftmp3(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T,2*T)
    E3 = 0.0
    #E33 = ft3.lccs1(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T)
    Etot = E0 + E1 + E2 + E3 + En
    Etotr = E0r + E1r + E2r + E3r + En
    print(float(E0r),float(E1r),float(E2r))
    print(E0,E1,E2)
    print("\n")
