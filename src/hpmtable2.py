import os
#os.environ['JAX_ENABLE_X64']='True'

from scipy.interpolate import CubicSpline
import numpy as np
import pdb 
import jax
import jax.numpy as jnp
import jax_cosmo as jc

def c200c_nfw(cosmo, M200c):
    # Local parameters
    kappa,alpha,beta,phi0,phi1,eta0,eta1 = 1.00,1.08,1.77,6.58,1.27,7.28,1.56


    #! Lagrangian radius in comoving Mpc/h
    #rho0 = cosmo%om*rhoc_ast ---> rhoc_ast = 2.77550D+11
    #M    = M200c/Msun_cgs*cosmo%h
    #R    = (3*M/(4*pi*rho0))**(1./3)

    # Lagrangian radius in comoving Mpc/h
    rho0       = cosmo.Omega_m * jc.constants.rhocrit * 1 # to be changed to proper conversion factor self.cosmo['rhoc_ast']
    M          = M200c * cosmo.h /Msun_cgs #* cosmo.h
    R          = (3*M/(4*jnp.pi*rho0))**(1./3)
    k          = kappa*2*jnp.pi/R
    neff_pk_fn = jax.grad(lambda k: jnp.log(jc.power.linear_matter_power(cosmo, k)))
    n          = neff_pk_fn(k)

    nu         = 1.686/jc.power.sigmasqr(cosmo, R, jc.transfer.Eisenstein_Hu) 
    cmin       = phi0 + phi1*n
    numin      = eta0 + eta1*n
    c200c_nfw  = cmin/2*((numin/nu)**alpha + (nu/numin)**beta)
    #pdb.set_trace()
    return c200c_nfw

def M500c_from_M200c(cosmo, M200c, R200c, c200c):
    rs = R200c/c200c
    Am = M200c/(jnp.log(1+c200c) - c200c/(1+c200c))
    # Calc M500c iteratively
    x1  = 0
    x2  = c200c
    rho = 0
    while (abs(rho/(500*jc.constants.rhocrit) - 1) > 1E-4):
        # Calc average density
        x   = (x1 + x2)/2
        R   = x*rs
        M   = Am*(jnp.log(1+x) - x/(1+x))
        rho = M/(4*jnp.pi/3*R**3)
        # Check average density
        if (rho > 500*jc.constants.rhocrit ):
            x1 = x
        else:
            x2 = x

    M500c_from_M200c = M
    return M500c_from_M200c
 
def P500c_gnfw(cosmo,a,icm,M):
    # Arnaud et al
    h70    = cosmo.h/0.7

    hsq    = cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    #hsq      = cosmo%or/a**4 + cosmo%om/a**3 + cosmo%ol

    Hz     = (cosmo.h*100)*jnp.sqrt(hsq)

    hz     = Hz/(H0_cgs*cosmo.h)
    P500c_gnfw = 1.65*eV2erg*(icm['mue']/icm['mu'])*jnp.power(hz, 8./3) * (M/(3E14*Msun_cgs/h70))**(2./3)*jnp.power(h70, 2)
    return P500c_gnfw

def rho_nfw(x, M, R, c):
    #print(M / (4 * jnp.pi * R ** 3) * c ** 3 / (jnp.log(1 + c) - c / (1 + c)) / (x * (1 + x) ** 2))
    #pdb.set_trace()
    return M / (4 * jnp.pi * R ** 3) * c ** 3 / (jnp.log(1 + c) - c / (1 + c)) / (x * (1 + x) ** 2)

def acc_nfw(x, M, R, c):
    return -G_cgs*M/(R*x/c)**2*(jnp.log(1+x) - x/(1+x))/(jnp.log(1+c) - c/(1+c))

def P_gnfw(icm,x):
    # Dimensionless pressure profile
    P_gnfw = icm['p0']/(icm['c500']*x)**icm['gamma']/(1+(icm['c500']*x)**icm['alpha'])**((icm['beta']-icm['gamma'])/icm['alpha'])
    return P_gnfw

def fnth_gnfw(x):
    L,b,k,d   = 0.00,0.913,0.244, 1.244
    fnth_gnfw = 1-L+(L-b)*jnp.exp(-(k*x)**d)
    return fnth_gnfw

def Pnth_gnfw(icm,x):
    fnth      = fnth_gnfw(x)
    Pnth_gnfw = fnth/(1 - fnth)*P_gnfw(icm,x)
    return Pnth_gnfw

def dPdx_gnfw(icm,x):
    return -P_gnfw(icm,x) * (icm['gamma']/x + icm['c500']*(icm['beta']-icm['gamma'])*(icm['c500']*x)**(icm['alpha']-1)/(1+(icm['c500']*x)**icm['alpha']))

def dPdx_tot(icm,x):
    L, b, k, d = 0.00, 0.913, 0.244, 1.244
    fth        = L-(L-b)*jnp.exp(-(k*x)**d)
    Pth        = P_gnfw(icm,x)
    dfthdx     = (L-b)*jnp.exp(-(k*x)**d)*(k*x)**(d-1.0)*d*k     
    dPthdx     = dPdx_gnfw(icm,x)
    dPdx_tot   = (dPthdx*fth-dfthdx*Pth)/fth**2
    return dPdx_tot

def psi_nfw(x,M,R,c):
    G_cgs = 6.6743e-8
    rhos  = M/(4*np.pi*R**3)*c**3/(jnp.log(1+c) - c/(1+c))
    rs    = R/c
    Apsi  = 4*jnp.pi*G_cgs*rhos*rs
    if jnp.abs(x-1) > 1E-6:
        psi_nfw = Apsi*jnp.log(x)/(x**2 - 1)
    else:
        psi_nfw = Apsi*(1-x/2)
    return psi_nfw


def rhocrit_z_cgs(cosmo,a):
    hsq      = cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    Hz       = H0_cgs*cosmo.h*jnp.sqrt(hsq)
    return jnp.float64(3*Hz**2/(8*jnp.pi*G_cgs))


Msun_cgs = 1.98900e+33
H0_cgs   = 3.24086e-18
rhoc_ast = 2.77550e+11
eV2erg   = eV_cgs  = 1.60218e-12
G_cgs    = 6.6743e-8
k_cgs    = 1.38066e-16

#M,r = func(rho,fscal)
#P   = func(M,r)
#T   = func(M,r)


def table_halo(cosmo,a,icm,M,r):
    '''
    Parameters
    ----------
    cosmo : jaxcosmo cosmology instance
      cosmology object
    icm   : dict
      dictionary containing icm model parameters
    M     : float 
      Mass at which to evaluate (M200c) *in units of Msun/h*
    r     : float
      radius at which to evaluate

    Returns
    -------
    cl : array
      the cl array
    '''

    #(M,r) -> compute rho,psi etc.

    M200c   = M/cosmo.h*Msun_cgs
    rho200c = 200*rhocrit_z_cgs(cosmo,a) # rhocrit in cgs units
    rho500c = 500*rhocrit_z_cgs(cosmo,a)
    #M200c   = 10**((im-1)*self.lgMdel + self.lgMmin) / self.cosmo['h'] * Msun_cgs
    R200c   = (M200c/1e30/(4*jnp.pi/3*(rho200c*1e30) ))**(1./3)*(1.e30)**(1./3)*(1.e30)**(1./3) # avoid float64
    #pdb.set_trace()
    c200c   = c200c_nfw(cosmo,M200c)
    rs      = R200c/c200c
    #pdb.set_trace()

    M500c   = M500c_from_M200c(cosmo,M200c,R200c,c200c)
    R500c   = (M500c/(4*jnp.pi/3*rho500c))**(1./3)
    c500c   = R500c/rs
    P500c   = P500c_gnfw(cosmo,a,icm,M500c)

    #r = 10**((ir-1)*self.lgrdel + self.lgrmin)*R200c
    s = r/R500c
    x = r/rs
            
    # NFW
    #pdb.set_trace()
    dm = rho_nfw(x,M200c,R200c,c200c);print(x,M200c,R200c,c200c,dm)
    f  = acc_nfw(x,M200c,R200c,c200c);print(f)
    f  = abs(f)

    # GNFW
    P       = P500c*P_gnfw(icm,s);print(P)
    Pnth    = P500c*Pnth_gnfw(icm,s)
    dPdr    = P500c*dPdx_gnfw(icm,s)/R500c
    dPtotdr = P500c*dPdx_tot(icm,s)/R500c
    dg      = abs(dPtotdr/f)
    #fp      = abs(dPdr/dg)
    T       = P/(k_cgs*dg/icm['mu'])
    vsq     = 3*Pnth/dg

    rho     = dm
    psi     = abs(psi_nfw(x,M200c,R200c,c200c))
            
    # Here no smoothing is required because we are just computing the exact values
    #rhosmth = rho 
    #psismth = psi
    #Tsmth   = T 
    print(M,r,s,x,rho,psi,T,vsq,P,Pnth)
    return M,r,s,x,rho,psi,T,vsq,P,Pnth#,f,fp



def table_icm(cosmo,a,icm,rho,psi):

    Mmin, Mmax, lgMdel = 1E8, 5E15, 1E-2
    Nmass  = int(1 + np.round((np.log10(Mmax) - np.log10(Mmin))/lgMdel))

    rmin, rmax, lgrdel = 1E-2, 4., 1E-2
    Nrad   = int(1 + np.round((np.log10(rmax) - np.log10(rmin))/lgrdel))

    M200c  = 10**((np.arange(Nmass))*lgMdel + np.log10(Mmin))/cosmo.h * Msun_cgs
    r      = 10**((np.arange(Nrad))*lgrdel  + np.log10(rmin))*R200c

    #(rho,psi) -> compute M,r,T,P
    M_grid, r_grid  = np.meshgrid(M200c,r) 
    M,r,s,x,rho,psi,T,vsq,P,Pnth = table_halo(cosmo,icm,M_grid,r_grid)#.reshape((M_arr.shape[0],r_arr.shape[0]))


    #drho    = abs((self.hpmtabA['rho'] - lgd)/self.lgddel)
    #drhomin = np.minimum(drhomin,drho)
    #drhomin[drhomin>0.5]=0.5

    drhomin = np.ones_like(psi)*np.finfo(np.float64).max
    drho    = abs((rho - lgd)/lgddel)
    drhomin = np.minimum(drhomin,drho)
    drhomin[drhomin>0.5] = 0.5

    dpsimin = np.ones_like(drho)*np.finfo(np.float64).max
    dpsi    = abs((psi - lgp)/lgpdel)
    dpsimin = np.minimum(dpsimin,dpsi)
    dpsimin[dpsimin>0.5] = 0.5  
    
    idx  = np.where((drho <= drhomin) & (dpsi <= dpsimin))
    
    return rho,psi,np.mean(M),np.mean(r),np.mean(s),np.mean(s),np.mean(x),np.mean(T),np.mean(vsq),np.mean(P),np.mean(Pnth)

mH_cgs   = 1.67223e-24
cosmo = jc.Planck15()
icm = {}
icm['XH']  = 0.76
icm['YHe'] = 0.24
icm['mu']   = mH_cgs/(2*icm['XH'] + 3*icm['YHe']/4)
icm['mue']  = mH_cgs/(  icm['XH'] + 2*icm['YHe']/4)
#icm['mue']  = mH_cgs/(  cosmo['XH'] + 4*cosmo['YHe']/4 + 676*icm%Zxry*cosmo%XH)
icm['p0']     = 8.403
icm['c500']   = 1.177
icm['gamma']  = 0.3081
icm['alpha']  = 1.0510
icm['beta']   = 5.4905

M=1e14
r=1
a=1
table_halo(cosmo,a,icm,M,r)