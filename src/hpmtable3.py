import os,sys
# os.environ['JAX_ENABLE_X64']='True'
#from jax import config
#config.update("jax_enable_x64", True)

from scipy.interpolate import CubicSpline
import numpy as np
import pdb 
import jax
import jax.numpy as jnp
import jax_cosmo as jc

def neff_pk_fn(cosmo,keff):
    #! Eisenstein & Hu (1998)
    gamma = cosmo.Omega_m*cosmo.h
    theta = 2.725/2.7
    dlnP  = 0
    dlnk  = 0.01

    for i in [-1, 1]:
        k    = keff*jnp.exp(i*dlnk/2)
        q    = k*theta**2/gamma
        L0   = jnp.log(5.43656 + 1.8*q)
        C0   = 14.2 + 731/(1 + 62.5*q)
        T0   = L0/(L0 + C0*q**2)
        P    = k**cosmo.n_s*T0**2
        dlnP = dlnP + i*jnp.log(P)

    neff_pk = dlnP/dlnk

    return neff_pk

def tophat_transform(x):
    return 3*(jnp.sin(x)-jnp.cos(x)*x)/x**3

def c200c_nfw(cosmo,a, M200c,verbose=False):
    kappa,alpha,beta,phi0,phi1,eta0,eta1 = 1.00,1.08,1.77,6.58,1.27,7.28,1.56

    # Lagrangian radius in comoving Mpc/h
    rho0       = cosmo.Omega_m * jc.constants.rhocrit*cosmo.h**2  # [(Msun)(1/Mpc)^3]
    M          = M200c                                 # [Msun]
    R          = (3*M/(4*jnp.pi*rho0))**(1./3)         # [Mpc]
    #pdb.set_trace()
    if verbose: print("R",R)
    #--------XCHECK----------
    # this :  6.540351575457153
    # hyper:  6.53995918296178    

    k          = kappa*2*jnp.pi/R  #[1/Mpc]
    if verbose: print('k',k)
    #--------XCHECK----------
    # this :  0.9606800543807782
    # hyper:  0.960737694441404    

    #neff_pk_fn = jax.grad(lambda k: jnp.log10(jc.power.linear_matter_power(cosmo, k) )) #####<_---- different answer
    #jc.power.linear_matter_power(cosmo, k)/2/np.pi**2*k**3 <--- need some scaling of k
    n          = neff_pk_fn(cosmo,k/cosmo.h) 
    #pdb.set_trace() 
    if verbose: print('n',n)
    #--------XCHECK----------
    # this :  -2.251191462464597
    # hyper:  -2.25120419419956 



    tmp = jnp.logspace(-5,5,1001) #[1/Mpc]
    k   = (tmp[1:]+tmp[:-1])/2
    dk  = tmp[1:]-tmp[:-1]

    pk   = jc.power.linear_matter_power(cosmo, k/cosmo.h)/2/jnp.pi**2*(k/cosmo.h)**3
    kR   = k*R

    dlnk = dk/k #jnp.log(tmp[1:]/tmp[:-1])#/dk
    #dlnk = jnp.log(k[1]/k[])/(k2-k1) #equivalent
    
    var    = jnp.sum(pk*tophat_transform(kR)**2*dlnk)
    sigmaR = jnp.sqrt(var)*jc.background.growth_factor(cosmo, jnp.array([a])).item()
    #pdb.set_trace()
    #kR = R*k
    #W  = 3.0*(jnp.sin(kR)/kR**3-jnp.cos(kR)/kR**2)
    #pk = jc.power.linear_matter_power(cosmo, k)#/2/np.pi**2*k**3
    #I  = dk*pk*W*W*k*k
    #sigmaR = np.sum(I)/(2.0*jnp.pi*jnp.pi)


    #kr   = k#cosmo%Plin(1,k)
    #pk   = jc.power.linear_matter_power(cosmo, k)/2/np.pi**2*k**3#cosmo%Plin(3,k)
    #dlnk = jnp.log(pk[1:]/pk[:-1])/dk
    #x    = kr*dlnk
    #pk*tophat_transform(kR)**2*dlnk
    #TF=0.21371743396866935

    #k = tmp#jnp.exp(np.log(tmp))
    #dk = 1.0*k
    #kR = R*k
    #W  = 3.0*(jnp.sin(kR)/kR**3-jnp.cos(kR)/kR**2)
    #pk =jc.power.linear_matter_power(cosmo, k)#/2/np.pi**2*k**3
    #sigmaR = np.sum(dk*pk*W*W*k*k)/(2.0*jnp.pi*jnp.pi)*jc.background.growth_factor(cosmo, np.array([a])).item()

    nu         = 1.686/sigmaR
    if verbose: print('nu',nu)
    #pdb.set_trace()
    #--------XCHECK----------
    # this :  8.500356
    # hyper:  8.49532892283707 

    #pdb.set_trace()
    cmin       = phi0 + phi1*n
    numin      = eta0 + eta1*n
    c200c_nfw  = cmin/2*((numin/nu)**alpha + (nu/numin)**beta)
    
    #pdb.set_trace()
    return c200c_nfw


def M500c_from_M200c(cosmo, M200c, R200c, c200c, rho500c):
    
    '''
    rs = R200c/c200c
    Am = M200c/1e24/(jnp.log(1+c200c) - c200c/(1+c200c))
    # Calc M500c iteratively
    x1  = 0
    x2  = c200c
    rho = 0
    
    while (abs(rho/(rho500c) - 1) > 1E-4):

        #print( abs(rho/(rho500c) - 1) )

        # Calc average density
        x   = (x1 + x2)/2
        R   = x*rs
        M   = Am*(jnp.log(1+x) - x/(1+x))
        rho = M/(4*jnp.pi/3*(R/1e12)**3)*1e24/1e12/1e12/1e12 # from Am
        # Check average density
        #print(rho,rho500c)
        if (rho > rho500c ):
            x1 = x
        else:
            x2 = x
        #print(  abs(rho/(rho500c) - 1) )

    M500c_from_M200c = M#*1e30
    print("OLD M500c_from_M200c:",M500c_from_M200c)
    '''
    #--------------------------------------------------------------------------

    rs = R200c/c200c
    Am = M200c/(jnp.log(1+c200c) - c200c/(1+c200c))
    x   = jnp.logspace(0,c200c,100001)
    R   = x*rs
    M   = Am*(jnp.log(1+x) - x/(1+x))
    rho = M/(4*jnp.pi/3*R**3) 
    
    Mi  = jnp.interp(rho500c,rho[::-1],M[::-1]) # somehow doesnt like decreasing function

    M500c_from_M200c = jnp.interp(rho500c,rho[::-1],M[::-1])
        
    return M500c_from_M200c

"""
def M500c_from_M200c(cosmo, M200c, R200c, c200c, rho500c):
    
    kappa   = 1.83
    a0      = 1.95
    a1      = 1.17
    b0      = 3.57
    b1      = 0.91
    c_alpha = 0.26
"""
 
def P500c_gnfw(cosmo,a,icm,M,verbose=False):
    # Arnaud et al
    h70    = cosmo.h/0.7

    hsq    = 8.5e-5/a**4+cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    
    Hz     = H0_cgs*jnp.sqrt(hsq)*cosmo.h # [1/s]
    if verbose: print('Hz',Hz)
    #--------XCHECK----------
    # this :  1.7999619e-17
    # hyper:  1.799961888504084E-017  

    hz     = jnp.sqrt(hsq) #[unitless]
    #print('Hz',Hz)
    #print('braket',(H0_cgs*cosmo.h))
    if verbose: print('hz',hz)
    #--------XCHECK----------
    # this :  8.198942
    # hyper:  8.19894261475222  


    #P500c_gnfw = 1.65*eV2erg*(icm['mue']/icm['mu'])*jnp.power(hz, 8./3) * (M/(3E14*(Msun_cgs/1e24)/h70))**(2./3)*jnp.power(h70, 2)
    P500c_gnfw = 1.65*eV2erg*1.5515e-09*(icm['mue']/icm['mu'])*jnp.power(hz, 8./3) * (M/(3E14/h70))**(2./3)*jnp.power(h70, 2)
    #1.5515e-09 converts from erg/cm^3 -> Msun/s^2/Mpc
    #pdb.set_trace()
    if verbose: print('P500c_gnfw',P500c_gnfw)
    #--------XCHECK----------
    # this :  6.770829e-10
    # hyper:  6.770158664810687E-010
    return P500c_gnfw

def rho_nfw(x, M, R, c):
    #print(M / (4 * jnp.pi * R ** 3) * c ** 3 / (jnp.log(1 + c) - c / (1 + c)) / (x * (1 + x) ** 2))
    A = c*c*c / (jnp.log(1 + c) - c / (1 + c)) / (x * (1 + x) ** 2) 
    B = (M) / (4 * jnp.pi * (R) ** 3) 
    return A*B

def acc_nfw(x, M, R, c):
    #a=-G_cgs*(M/1e30)/(R*x/c/1e20)**2*(jnp.log(1+x) - x/(1+x))/(jnp.log(1+c) - c/(1+c))*1e-10
    # For this function, the outputs dont really matter but we should just match it 
    return -G_si*(Msun_si/Mpc_m**2)*100*(M)/(R*x/c)**2*(jnp.log(1+x) - x/(1+x))/(jnp.log(1+c) - c/(1+c))

def P_gnfw(icm,x):
    # Dimensionless pressure profile

    P_gnfw = icm['p0']/(icm['c500']*x)**icm['gamma']/(1+(icm['c500']*x)**icm['alpha'])**((icm['beta']-icm['gamma'])/icm['alpha'])
    #pdb.set_trace()
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

def psi_nfw(x,M,R,c,verbose=False):
    rhos  = (M)/(4*jnp.pi*(R)**3)*(c)**3/(jnp.log(1+c) - c/(1+c))
    #print('rhos',rhos)
    if verbose: print('rhos',rhos)
    #--------XCHECK----------
    # this : 1.8115183e-23
    # hyper: 1.807925321299102E-023

    rs    = R/c
    if verbose: print('rs',rs)
    #--------XCHECK----------
    # this : 9.8051515e+22
    # hyper: 9.813421148458727E+022
    
    #Apsi  = 4*jnp.pi*G_si*rhos*rs*Msun_si/Mpc_m**3*Mpc_cm
    Apsi  = 4*jnp.pi*(G_si*1e11)*(rhos*1e-17)*rs*(Msun_si*1e-30)/(Mpc_m*1e-22)**3*(Mpc_cm*1e-24)/1e6
    #print('Apsi',Apsi)
    #pdb.set_trace()
    if verbose: print('Apsi',Apsi)
    #pdb.set_trace()
    #--------XCHECK----------
    # this : 1.4897473e-06
    # hyper:  1.487665285580003E-006
    
    #pdb.set_trace()
    #pdb.set_trace()

    #if jnp.abs(x-1) > 1E-6:
    #    psi_nfw = Apsi*jnp.log(x)/(x**2 - 1)
    #else:
    #    psi_nfw = Apsi*(1-x/2)

    psi_nfw=jnp.where(jnp.abs(x-1) < 1E-6, Apsi*(1-x/2), Apsi*jnp.log(x)/(x**2 - 1) )  

    return psi_nfw
'''
def rhocrit_z(cosmo,a):
    hsq      = 8.5e-5/a**4+cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    Hz       = H0_cgs*cosmo.h*jnp.sqrt(hsq) #[1/s] note here we are leaving in h
    #print(hsq)
    #print(Hz)
    #print(3*Hz**2/(8*jnp.pi*G)/Msun_si*Mpc_m**3)
    #pdb.set_trace()
    #print(Msun_si)
    #print(Mpc_m**3)
    #return 3*(Hz*1e17)**2/(8*jnp.pi*G_si*1e11)/(Msun_si*1e-30)*(Mpc_m*1e-22)**3*1e-15 #[Msun/(Mpc)^3]
    return 3*(Hz)**2/(8*jnp.pi*G_si)/(Msun_si)*(Mpc_m)**3 #[Msun/(Mpc)^3]
'''
def rhocrit_z(cosmo,a):
    hsq      = 8.5e-5/a**4+cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    Hz       = H0_cgs*cosmo.h*jnp.sqrt(hsq) #[1/s] note here we are leaving in h
    #print(3*Hz**2/(8*jnp.pi*G)/Msun_si*Mpc_m**3)
    #pdb.set_trace()
    return 3*(Hz*1e17)**2/(8*jnp.pi*G_si*1e11)/(Msun_si*1e-30)*(Mpc_m*1e-22)**3/1e-13 #[Msun/(Mpc)^3]
    #return 3*Hz**2/(8*jnp.pi*G_si)/Msun_si*Mpc_m**3 #[Msun/(Mpc)^3]

def rhocrit_z_cgs(cosmo,a):
    hsq      = 8.5e-5/a**4+cosmo.Omega_m/a**3 + cosmo.Omega_de #in the original code there is also Omega_r/a**4
    Hz       = H0_cgs*cosmo.h*jnp.sqrt(hsq)
    return 3*Hz**2/(8*jnp.pi*G_cgs)

G_si     = 6.67e-11 #[m^3/kg/s^2]
Mpc_m    = 3.086e22
Mpc_cm    = 3.086e24
Msun_si  = 1.988e30 
Msun_cgs = 1.98900e+33
H0_cgs   = 3.24086e-18 #1/s assuming H0=100 km/s/Mpc
rhoc_ast = 2.77550e+11
eV2erg   = eV_cgs  = 1.60218e-12
G_cgs    = 6.6743e-8
k_cgs    = 1.38066e-16
k_si     = 1.38065e-23
#M,r = func(rho,fscal)
#P   = func(M,r)
#T   = func(M,r)


def table_halo(cosmo,a,icm,M,rx,verbose=False):
    '''
    Parameters
    ----------
    cosmo : jaxcosmo cosmology instance
      cosmology object
    icm   : dict
      dictionary containing icm model parameters
    M     : float 
      Mass at which to evaluate (M200c) *in units of Msun*
    r     : float
      radius at which to evaluate

    Returns
    -------
    cl : array
      the cl array
    '''

    #(M,r) -> compute rho,psi etc.
    #M200c   = M/cosmo.h*Msun_cgs
    M200c    = M #[Mpc]
    if verbose: print("M200c",M200c)
    #--------XCHECK----------
    # this : 2.936226749335696e+47
    # hyper: 2.936226760400337E+047

    rho200c = 200*rhocrit_z(cosmo,a) #[(Msun)/(Mpc)^3]
    if verbose: print("rho200c",rho200c) 
    #--------XCHECK----------
    # this : 1.1587519e-25
    # hyper: 1.159160489937735E-025

    rho500c = 500*rhocrit_z(cosmo,a) #[(Msun/h)/(h/Mpc)^3]
    if verbose: print("rho500c",rho500c) 
    #--------XCHECK----------
    # this : 2.89688e-25
    # hyper: 2.897901224844337E-025
    
    #rho200c = 200*test_rhocrit_z_cgs(cosmo,0.166666666666667 ) # XCHECKED rhocrit in cgs units
    #rho500c = 500*test_rhocrit_z_cgs(cosmo,0.166666666666667 ) # XCHECKED rhocrit in cgs units
    
    #sys.exit()
    #M200c   = 10**((im-1)*self.lgMdel + self.lgMmin) / self.cosmo['h'] * Msun_cgs
    #R200c   = (M200c/1e30/(4*jnp.pi/3*(rho200c*1e30) ))**(1./3)*(1.e30)**(1./3)*(1.e30)**(1./3) # avoid float64
    R200c   = ( M200c/(4*np.pi/3*rho200c) ) **(1./3) # [Mpc]
    #pdb.set_trace()
    if verbose: print('R200c',R200c)
    #--------XCHECK----------
    # this : 8.457399482718055e+23
    # hyper: 8.456419471688975E+023

    c200c   = c200c_nfw(cosmo,a,M200c) #<---- still room for improvement
    #pdb.set_trace()
    if verbose: print('c200c',c200c)
    #--------XCHECK----------
    # this : 8.625189
    # hyper: 8.61719816540954  

    rs      = R200c/c200c
    if verbose: print('rs',rs)
    #--------XCHECK----------
    # this :  9.8051515e+22
    # hyper:  9.813421148458727E+022

    M500c   = M500c_from_M200c(cosmo,M200c,R200c,c200c,rho500c) 
    if verbose: print('M500c',M500c)
    #--------XCHECK----------
    # this :  2.2985634e+47
    # hyper:  2.298221906926749E+047
    
    #pdb.set_trace()
    R500c   = (M500c/(4*jnp.pi/3*rho500c))**(1./3)
    if verbose: print('R500c',R500c)
    #--------XCHECK----------
    # this :  5.74291e+23
    # hyper:  5.742142494073986E+023
    
    c500c   = R500c/rs
    if verbose: print('c500c',c500c)
    #--------XCHECK----------
    # this :  5.8570333
    # hyper:  5.85131566984245
    
    P500c   = P500c_gnfw(cosmo,a,icm,M500c)
    if verbose: print('P500c',P500c)
    #--------XCHECK----------
    # this :  1.3663051e-08
    # hyper:  6.770158664810688E-010
    
    ry = rx*R200c
    if verbose: print('r',r)
    #--------XCHECK----------
    # this :  8.457127621049304e+21
    # hyper:  8.456419282673272E+021
    
    s = ry/R500c
    if verbose: print('s',s)
    #--------XCHECK----------
    # this :  0.014726206
    # hyper:  1.472694084377126E-002
    
    x = ry/rs
    if verbose: print('x',x)
    #--------XCHECK----------
    # this :  0.08625188
    # hyper:  8.617197972800157E-002
        
    # NFW
    dm = rho_nfw(x,M200c,R200c,c200c)      # (Msun)/(Mpc)**3
    if verbose: print('dm',dm)
    #pdb.set_trace()
    #--------XCHECK----------
    # this :  1.7799716e-22
    # hyper:  1.778349949090758E-022
    
    f  = abs(acc_nfw(x,M200c,R200c,c200c)) # cm/s^2
    if verbose: print('f',f)
    #pdb.set_trace()
    #--------XCHECK----------
    # this :  6.6681764e-07
    # hyper:  6.659556321600092E-007
    
    #pdb.set_trace()

    # GNFW
    #print('ee')
    P       = P500c*P_gnfw(icm,s) # (Msun/s^2/Mpc) * [unitless]
    if verbose: print('P',P)
    #--------XCHECK----------
    # this :  1.8522899e-08
    # hyper:  1.852071281979021E-008

    #Pnth    = P500c*Pnth_gnfw(icm,s)
    #dPdr    = P500c*dPdx_gnfw(icm,s)/R500c
    dPtotdr = P500c*dPdx_tot(icm,s)/R500c # (Msun/s^2/Mpc) * (1/Mpc)
    if verbose: print('dPtotdr',dPtotdr)
    #--------XCHECK----------
    # this :  -9.100081e-31
    # hyper:  -9.099857761046103E-031

    dg      = abs(dPtotdr/f)
    if verbose: print('dg',dg)
    #--------XCHECK----------
    # this :  1.3647031e-24
    # hyper:  1.366436038919133E-024

    #####fp      = abs(dPdr/dg)
    #T       = P/(k_cgs*(dg)/icm['mu'])
    T       = P/(k_si*(dg)/icm['mu'])/100*Mpc_m/1000
    
    #pdb.set_trace()
    if verbose: print('T',T)
    #--------XCHECK----------
    # this :  96701032
    # hyper:  96566999.2430575

    #vsq     = 3*Pnth/dg

    rho     = dm
    psi     = abs(psi_nfw(x,M200c,R200c,c200c))
    if verbose: print('psi',psi)
    #--------XCHECK----------
    # this :  3.677963e-06
    # hyper:  3.674160763279385E-006
    #pdb.set_trace()

    return M,rx,s,x,rho,psi,T,P#,f,fp



def table_icm(cosmo,a,icm,rho,psi):

    Mmin, Mmax, lgMdel = 1E8, 5E15, 1E-2
    Nmass  = int(1 + jnp.round((jnp.log10(Mmax) - jnp.log10(Mmin))/lgMdel))

    rmin, rmax, lgrdel = 1E-2, 4., 1E-2
    Nrad   = int(1 + jnp.round((jnp.log10(rmax) - jnp.log10(rmin))/lgrdel))

    # Arbitrary gridding scheme to compute table
    M      = 10**((jnp.arange(Nmass))*lgMdel + jnp.log10(Mmin)) #/cosmo.h #* Msun_cgs <--------Msun/h
    r      = 10**((jnp.arange(Nrad))*lgrdel  + jnp.log10(rmin))#*1e24 # proxy for R200c

    #pdb.set_trace()
    psigrid    = jnp.zeros((Nmass,Nrad)) 
    rhogrid    = jnp.zeros((Nmass,Nrad)) 
    Tgrid      = jnp.zeros((Nmass,Nrad)) 
    Pgrid      = jnp.zeros((Nmass,Nrad)) 
    
    #(rho,psi) -> compute M,r,T,P
    #M_grid, r_grid  = np.meshgrid(M200c,r)
    c=0 
    for i in range(Nmass):
        for j in range(Nrad):
            #pdb.set_trace()
            icmM,icmr,icms,icmx,icmrho,icmpsi,icmT,icmP = table_halo(cosmo,a,icm,M200c[i],r[j])#.reshape((M_arr.shape[0],r_arr.shape[0]))
            psigrid.at[i,j].set(icmpsi)
            rhogrid.at[i,j].set(icmrho)
            Tgrid.at[i,j].set(icmT)
            Pgrid.at[i,j].set(icmP)
            
            c+=1
        print(i)
            

    #drho    = abs((self.hpmtabA['rho'] - lgd)/self.lgddel)
    #drhomin = np.minimum(drhomin,drho)
    #drhomin[drhomin>0.5]=0.5


    #! Density range in rho/<rho>
    dmin   = 0.124999994412065/2
    dmax   = 664.926049658025*2
    lgddel = 5E-2
    lgdmin = jnp.log10(dmin)
    lgdmax = jnp.log10(dmax)
    Nrho   = 1 + jnp.int((lgdmax - lgdmin)/lgddel)


    #! psi range in particle units
    pmin   = 2.09123620573559/2
    pmax   = 32.3989365191839*2
    lgpdel = 5E-2
    lgpmin = jnp.log10(pmin)
    lgpmax = jnp.log10(pmax)
    Npsi   = 1 + jnp.int((lgpmax - lgpmin)/lgpdel)

    """
    write(*,*) part%pmin
    write(*,*) part%pmax
    write(*,*) part%dmin
    write(*,*) part%dmax
    2.09123620573559
    32.3989365191839
    0.124999994412065
    664.926049658025
    """;

    lgp = (jnp.arange(Npsi)-1)*lgpdel + lgpmin
    lgd = (jnp.arange(Nrho)-1)*lgddel + lgdmin

    drhomin = jnp.ones_like(icmpsi)*jnp.finfo(jnp.float64).max
    drho    = abs((rhogrid - lgd)/lgddel)
    drhomin = jnp.minimum(drhomin,drho)
    drhomin[drhomin>0.5] = 0.5

    dpsimin = np.ones_like(icmrho)*jnp.finfo(jnp.float64).max
    dpsi    = abs((psigrid - lgp)/lgpdel)
    dpsimin = jnp.minimum(dpsimin,dpsi)
    dpsimin[dpsimin>0.5] = 0.5  

    #pdb.set_trace()
    
    idx  = np.where((drho <= drhomin) & (dpsi <= dpsimin))
    
    return rho,psi,jnp.mean(M[idx]),jnp.mean(r[idx]),jnp.mean(T[idx]),jnp.mean(P[idx])#,jnp.mean(s[idx]),jnp.mean(x[idx]),jnp.mean(vsq[idx]),jnp.mean(Pnth[idx])


if __name__ == "__main__":

    verbose = True

    cosmo   = jc.Planck15()
    mH_cgs  = 1.67223e-24
    G       = 4.301e-9#
    
    icm = {}
    icm['XH']   = 0.76
    icm['YHe']  = 0.24
    icm['mu']   = mH_cgs/(2*icm['XH'] + 3*icm['YHe']/4)
    icm['mue']  = mH_cgs/(icm['XH'] + 2*icm['YHe']/4)
    icm['p0']   = 8.403
    icm['c500'] = 1.177
    icm['gamma']= 0.3081
    icm['alpha']= 1.0510
    icm['beta'] = 5.4905


    print('------------------table_halo-------------------')
    M =1e14                     #[Msun]
    r =9.999999776482584E-003   #[unitless]
    a =0.166666667
    table_halo(cosmo,a,icm,M,r,verbose=verbose) 


    print('------------------table_icm-------------------')
    # Compute grid using jitted function
    batched_r  = jax.vmap(table_halo,in_axes=[None, None, None, None, 0])
    batched_Mr = jax.vmap(batched_r, in_axes=[None,None,None,0,None])
    m_grid     = jnp.logspace(11,16,20)# Msun
    #r_grid     = jnp.logspace(np.log10(0.1),np.log10(4),20)# unitless, to be multiplied by R200c 
    r_grid     = jnp.linspace(0.1,4,20)# unitless, to be multiplied by R200c 
    
    res        = batched_Mr(cosmo,a,icm, m_grid.flatten(), r_grid.flatten())

    tabM,tabrx,tabs,tabx,tabrho,tabpsi,tabT,tabP = res 

    _rho    = jnp.log10(tabrho).flatten()
    _psi    = jnp.log10(tabpsi).flatten()
    _T      = jnp.log10(tabT).flatten()
    _P      = jnp.log10(tabP).flatten()
    _M      = jnp.log10(tabM).flatten()
    _R      = jnp.log10(tabrx).flatten()
    #_rho    = (tabrho).flatten()
    #_psi    = (tabpsi).flatten()
    #_T      = (tabT).flatten()
    #_P      = (tabP).flatten()

    import tensorflow_probability as tfp;
    tfp = tfp.substrates.jax
    tfb = tfp.bijectors
    tfd = tfp.distributions
    psd_kernels  = tfp.math.psd_kernels

    rho,psi = table_halo(cosmo,a,icm, 1e+14, 2 )[4:6]

    print('table rho',rho )
    print('table psi',psi )# 4->rho 5->psi
    
    index_points = jnp.array([jnp.log10(rho),jnp.log10(psi)]).reshape([-1,2]) # rho, psi

    print('index_point finite?',  np.all(np.isfinite(index_points)))
    print('_rho finite?',  np.all(np.isfinite(_rho)))
    print('_psi finite?',  np.all(np.isfinite(_psi)))
    print('_M finite?',  np.all(np.isfinite(index_points)))

    model_M = tfd.GaussianProcessRegressionModel( psd_kernels.ExponentiatedQuadratic(),
                                 index_points=index_points,
                                 observation_index_points=jnp.stack([_rho, _psi], axis=-1),
                                 observations=_M.flatten(),
                                 jitter=1e-03
                                )

    model_r = tfd.GaussianProcessRegressionModel( psd_kernels.RationalQuadratic(),
                                 index_points=index_points,
                                 observation_index_points=jnp.stack([_rho, _psi], axis=-1),
                                 observations=_R.flatten(),
                                 jitter=1e-03
                                )

    print('GPmodel', 10**(model_M.mean()), 10**model_r.mean())
     
    np.save('rho.npy',rho)
    np.save('psi.npy',psi)
    np.save('_rho.npy',_rho)
    np.save('_psi.npy',_psi)
    np.save('_M.npy',_M)
    np.save('_R.npy',_R)

    #print('table', table_halo(cosmo,a,icm, 3.4320555e+14,2.62402334e-08 )[4:6])# 4->rho 5->psi
    #print('------------------table_icm-------------------')
    #rho = 1e-2
    #psi = 3.6779604e-06
    #table_icm(cosmo,a,icm,rho,psi)
