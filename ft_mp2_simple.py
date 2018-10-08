import numpy
from pyscf import gto, scf, ao2mo

def fermi_function(beta, epsilon, mu):
    """Return the Fermi-Dirac distribution function."""
    #return 1.0 / (numpy.exp(beta*(epsilon - mu)) + 1.0)
    emm = epsilon - mu
    x = beta*emm
    if x < -30.0:
        return 1.0 - numpy.exp(x)
    elif x > 30.0:
        return numpy.exp(-x)
    else:
        return 1.0 / (numpy.exp(x) + 1.0)

# TODO: more accurate computation of 1 - n

def ff(beta, epsilon, mu):
    focc = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        focc[i] = fermi_function(beta, epsilon[i], mu)

    return focc

def grand_potential0(beta, epsilon, mu):
    emm = epsilon - mu
    x = beta*emm
    if x < -30.0:
        return emm
    elif x > 30.0:
        return 0.0
    else:
        return numpy.log(fermi_function(beta, epsilon, mu))/beta + emm 

def GP0(beta, epsilon, mu):
    argA = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        argA[i] = grand_potential0(beta, epsilon[i], mu)

    return argA


def ftmp1(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T):

    # form occupations 
    beta = 1.0 / (T + 1e-12)
    foccA = ff(beta, mo_energyA, mu)
    foccB = ff(beta, mo_energyB, mu)

    # form zero temperature density matrices
    na0 = (mo_energyA <  0).sum()
    nb0 = (mo_energyB <  0).sum()
    aocc = mo_coeffA[:,:na0]
    bocc = mo_coeffB[:,:nb0]
    pa0 = numpy.dot(aocc, aocc.T)
    pb0 = numpy.dot(bocc, bocc.T)
    dm0 = numpy.array((pa0,pb0))
    mf = scf.UHF(mol)
    h1 = mf.get_hcore(mol)
    veff0 = mf.get_veff(mol, dm0)
    f0 = h1 + veff0
    E0 = mo_energyA[:na0].sum() + mo_energyB[:nb0].sum()

    # form finite temperature density matrices
    pa = numpy.dot(numpy.dot(mo_coeffA,numpy.diag(foccA)),mo_coeffA.T)
    pb = numpy.dot(numpy.dot(mo_coeffB,numpy.diag(foccB)),mo_coeffB.T)
    dm = numpy.array((pa,pb))
    na = numpy.sum(foccA)
    nb = numpy.sum(foccB)

    # form Fock matrices
    veff = mf.get_veff(mol, dm)
    f = h1 + veff
    fa = f[0] - 2*f0[0]
    fb = f[1] - 2*f0[1]

    # compute contributions to 0th and 1st order energy
    ec1 = numpy.tensordot(h1,pa + pb,axes=([0,1],[0,1])) 
    ec2 = numpy.tensordot(fa,pa,axes=([0,1],[0,1]))
    ec3 = numpy.tensordot(fb,pb,axes=([0,1],[0,1]))
    argA = GP0(beta, mo_energyA, mu)
    argB = GP0(beta, mo_energyB, mu)

    # compute free energy at 0th and 1st order
    E0 = argA.sum() + argB.sum()
    E1 = 0.5*(ec1 + ec2 + ec3)
    #E1 = numpy.asscalar(numpy.tensordot(h1 - f[0],pa + pb,axes=([0,1],[0,1])) )

    return E0,E1

def ftmp2_singles(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T):
    rpre = 1e-10
    beta = 1.0 / (T + 1e-12)

    # compute occupation numbers
    foccA = ff(beta, mo_energyA, mu)
    foccB = ff(beta, mo_energyB, mu)

    # form zero temperature density matrices
    na0 = (mo_energyA < 0.0).sum()
    nb0 = (mo_energyB < 0.0).sum()
    aocc = mo_coeffA[:,:na0]
    bocc = mo_coeffB[:,:nb0]
    pa0 = numpy.dot(aocc, aocc.T)
    pb0 = numpy.dot(bocc, bocc.T)
    dm0 = numpy.array((pa0,pb0))
    mf = scf.UHF(mol)
    h1 = mf.get_hcore(mol)
    veff0 = mf.get_veff(mol, dm0)
    f0 = (h1 + veff0[0],h1 + veff0[1])
    #E0 = mo_energyA[:na0].sum() + mo_energyB[:nb0].sum()

    # form finite temperature density matrices
    pa = numpy.dot(numpy.dot(mo_coeffA,numpy.diag(foccA)),mo_coeffA.T)
    pb = numpy.dot(numpy.dot(mo_coeffB,numpy.diag(foccB)),mo_coeffB.T)
    dm = numpy.array((pa,pb))
    na = numpy.sum(foccA)
    nb = numpy.sum(foccB)

    # form Fock matrices
    veff = mf.get_veff(mol, dm)
    f = (h1 + veff[0] - f0[0], h1 + veff[1] - f0[1])

    # transform to MO basis
    fa = numpy.dot(mo_coeffA.T,numpy.dot(f[0],mo_coeffA))
    fb = numpy.dot(mo_coeffB.T,numpy.dot(f[1],mo_coeffB))
    Da = mo_energyA[:,None] - mo_energyA[None,:]
    Db = mo_energyB[:,None] - mo_energyB[None,:]
    Na = foccA[None,:]*(1 - foccA[:,None])
    Nb = foccB[None,:]*(1 - foccB[:,None])
    rfac = 2*T
    for x in numpy.nditer(Da,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac
    for x in numpy.nditer(Db,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac

    Da = Na/Da
    Db = Nb/Db
    fa = numpy.abs(fa)**2
    fb = numpy.abs(fb)**2
    Ea = -numpy.einsum('pq,qp->',fa,Da)
    Eb = -numpy.einsum('pq,qp->',fb,Db)
    return Ea + Eb

def ftmp2_doubles(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T):
    rpre = 1e-10
    beta = 1.0 / (T + 1e-12)

    # compute occupation numbers
    foccA = ff(beta, mo_energyA, mu)
    foccB = ff(beta, mo_energyB, mu)

    n = mo_coeffA.shape[1]
    assert(n == mo_coeffB.shape[1])
    eriaa = ao2mo.general(mol, (mo_coeffA,mo_coeffA,mo_coeffA,mo_coeffA), 
        compact=False).reshape(n,n,n,n)
    eriab = ao2mo.general(mol, (mo_coeffA,mo_coeffA,mo_coeffB,mo_coeffB), 
        compact=False).reshape(n,n,n,n)
    eribb = ao2mo.general(mol, (mo_coeffB,mo_coeffB,mo_coeffB,mo_coeffB), 
            compact=False).reshape(n,n,n,n)
    eriaa = numpy.abs(eriaa - eriaa.transpose(0,3,2,1))**2
    eribb = numpy.abs(eribb - eribb.transpose(0,3,2,1))**2
    Daa = mo_energyA[:,None,None,None] - mo_energyA[None,:,None,None] \
            + mo_energyA[None,None,:,None] - mo_energyA[None,None,None,:]
    Dab = mo_energyA[:,None,None,None] - mo_energyA[None,:,None,None] \
            + mo_energyB[None,None,:,None] - mo_energyB[None,None,None,:]
    Dbb = mo_energyB[:,None,None,None] - mo_energyB[None,:,None,None] \
            + mo_energyB[None,None,:,None] - mo_energyB[None,None,None,:]
    Faa = foccA[None,:,None,None]*foccA[None,None,None,:]*\
            (1.0 - foccA[:,None,None,None])*(1 - foccA[None,None,:,None])
    Fab = foccA[None,:,None,None]*foccB[None,None,None,:]*\
            (1.0 - foccA[:,None,None,None])*(1 - foccB[None,None,:,None])
    Fbb = foccB[None,:,None,None]*foccB[None,None,None,:]*\
            (1.0 - foccB[:,None,None,None])*(1 - foccB[None,None,:,None])
    
    rfac = 2*T
    for x in numpy.nditer(Daa,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac
    for x in numpy.nditer(Dab,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac
    for x in numpy.nditer(Dbb,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac

    Daa = Faa/Daa
    Dab = Fab/Dab
    Dbb = Fbb/Dbb

    Eaa = -0.25*numpy.einsum('pqrs,pqrs->',eriaa,Daa)
    Ebb = -0.25*numpy.einsum('pqrs,pqrs->',eribb,Dbb)
    Eab = -1.0*numpy.einsum('pqrs,pqrs,pqrs->',eriab,Dab,eriab)

    E2d = Eaa + Eab + Ebb
    return E2d

def ftmp2(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T):
    E2d = ftmp2_doubles(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T)
    E2s = ftmp2_singles(mol, mo_energyA, mo_energyB, mo_coeffA, mo_coeffB, mu, T)
    return E2s + E2d

def ftpttot(mol, m, mu, T):
    En = mol.energy_nuc()
    E2d = ftmp2(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T,0.1)
    E0,E1 = ftmp1(mol, m.mo_energy, m.mo_energy, m.mo_coeff, m.mo_coeff, mu, T,0.1)
    Etot = E0 + E1 + E2d + En
    return Etot
