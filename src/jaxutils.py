import jax
import jax.numpy as jnp
import jax_cosmo as jc

def c200c_nfw(cosmo, M200c):
    # Local parameters
    kappa,alpha,beta,phi0,phi1,eta0,eta1 = 1.00,1.08,1.77,6.58,1.27,7.28,1.56

    # Lagrangian radius in comoving Mpc/h
    rho0       = cosmo.Omega_m * jc.constants.rhocrit * 1 # to be changed to proper conversion factor self.cosmo['rhoc_ast']
    M          = M200c * cosmo.h #/Msun_cgs * cosmo.h
    R          = (3*M/(4*jnp.pi*rho0))**(1./3)
    k          = kappa*2*jnp.pi/R
    neff_pk_fn = jax.grad(lambda k: jnp.log(jc.power.linear_matter_power(cosmo, k)))
    n          = neff_pk_fn(k)

    nu         = 1.686/jc.power.sigmasqr(cosmo, R, jc.transfer.Eisenstein_Hu) 
    cmin       = phi0 + phi1*n
    numin      = eta0 + eta1*n
    c200c_nfw  = cmin/2*((numin/nu)**alpha + (nu/numin)**beta)
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
 
def P500c_gnfw(cosmo,icm,M):
    # Arnaud et al
    h70    = cosmo['h']/0.7
    hz     = cosmo['Hz']/(H0_cgs*cosmo['h'])
    P500c_gnfw = 1.65*eV2erg*(icm['mue']/icm['mu'])*jnp.power(hz, 8./3) * (M/(3E14*Msun_cgs/h70))**(2./3)*jnp.power(h70, 2)
    return P500c_gnfw

def rho_nfw(x, M, R, c):
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
    Pnth_gnfw = fnth/(1 - fnth)*P_gnfw(x,icm['p0'],icm['c500'],icm['alpha'],icm['beta'],icm['gamma'])
    return Pnth_gnfw

def dPdx_gnfw(icm,x):
    return -P_gnfw(x) * (icm['gamma']/x + icm['c500']*(icm['beta']-icm['gamma'])*(icm['c500']*x)**(icm['alpha']-1)/(1+(icm['c500']*x)**icm['alpha']))

def dPdx_tot(icm,x):
    L, b, k, d = 0.00, 0.913, 0.244, 1.244
    fth        = L-(L-b)*jnp.exp(-(k*x)**d)
    Pth        = P_gnfw(x,icm['p0'],icm['c500'],icm['alpha'],icm['beta'],icm['gamma'])
    dfthdx     = (L-b)*jnp.exp(-(k*x)**d)*(k*x)**(d-1.0)*d*k     
    dPthdx     = dPdx_gnfw(x,icm['p0'],icm['c500'],icm['alpha'],icm['beta'],icm['gamma'])
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

