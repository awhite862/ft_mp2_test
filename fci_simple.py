import numpy
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring
from pyscf.fci import fci_slow
import time
from mpmath import mp

def fci_partition_all(mol, m, T, mu, lam = 1.0):
    beta = 1.0 / T
    norb = m.mo_coeff.shape[1]
    Z = mp.mpf(0.0)
    f = m.get_fock()
    hprime = f + lam*(m.get_hcore() - f)
    for nalpha in range(0,norb + 1):
        for nbeta in range(0,norb + 1):
            #print(nalpha,nbeta)
            nel = nalpha + nbeta
            nelec = (nalpha,nbeta)
            h1e = reduce(numpy.dot, (m.mo_coeff.T, hprime, m.mo_coeff))
            eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
            eri = eri.reshape(norb,norb,norb,norb)
            eri = lam*eri
            h2e = fci_slow.absorb_h1e(h1e, eri, norb, nelec, .5)
            na = cistring.num_strings(norb, nalpha)
            nb = cistring.num_strings(norb, nbeta)
            N = na*nb
            assert(N < 2000)
            H = numpy.zeros((N,N))
            I = numpy.identity(N)
            for i in range(N):
                hc = fci_slow.contract_2e(h2e,I[:,i],norb,nelec)
                #hc = fci_slow.contract_1e(h1e,I[:,i],norb,nelec)
                hc.reshape(-1)
                H[:,i] = hc

            e,v = numpy.linalg.eigh(H)
            #print(e[0] + mol.energy_nuc())
            for j in range(N):
                ex = mp.mpf(beta*(nel*mu - e[j]))
                Z += mp.exp(ex)

    return Z

# only for RHF
def fci_simple(mol, m, lam = 1.0):
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    f = m.get_fock()
    hprime = f + lam*(m.get_hcore() - f)
    h1e = reduce(numpy.dot, (m.mo_coeff.T, hprime, m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = lam*eri.reshape(norb,norb,norb,norb)
    h2e = fci_slow.absorb_h1e(h1e, eri, norb, nelec, .5)
    na = cistring.num_strings(norb, nelec//2)
    N = na*na
    assert(N < 2000)
    H = numpy.zeros((N,N))
    I = numpy.identity(N)
    for i in range(N):
        hc = fci_slow.contract_2e(h2e,I[:,i],norb,nelec)
        hc.reshape(-1)
        H[:,i] = hc

    e,v = numpy.linalg.eigh(H)
    #print(e[0] + mol.energy_nuc())
    return e[0] + mol.energy_nuc()
