import os
#os.environ['JAX_ENABLE_X64']='True'

from scipy.interpolate import CubicSpline
import numpy as np
import pdb 
#from jax_cosmo.power import linear_matter_power
#from jax_cosmo import Cosmology as jaxCosmology
#import pyccl as ccl
#import jax

Msun_cgs = 1.98900e+33
H0_cgs   = 3.24086e-18
rhoc_ast = 2.77550e+11
eV2erg   = eV_cgs  = 1.60218e-12
G_cgs    = 6.6743e-8
k_cgs    = 1.38066e-16

class hpm_table:

    def __init__(self, cosmo, icm, igm, unit):

        self.cosmo  = cosmo
        self.icm    = icm
        self.igm    = igm
        self.unit   = unit

        # Mass range in M/(Msun/h)
        self.Mmin, self.Mmax, self.lgMdel = 1E8, 5E15, 1E-2
        self.lgMmin = np.log10(self.Mmin)
        self.lgMmax = np.log10(self.Mmax)
        self.Nmass  = int(1 + np.round((self.lgMmax - self.lgMmin)/self.lgMdel))

        # Radius range in r/R200c
        self.rmin, self.rmax, self.lgrdel = 1E-2, 4., 1E-2
        self.lgrmin = np.log10(self.rmin)
        self.lgrmax = np.log10(self.rmax)
        self.Nrad   = int(1 + np.round((self.lgrmax - self.lgrmin)/self.lgrdel))

        # Density range in rho/<rho>
        self.dmin   = 1e-5 #part.dmin/2 #????????????????????????????????????
        self.dmax   = 1e3  #part.dmax*2 #????????????????????????????????????
        self.lgddel = 5E-2
        self.lgdmin, self.lgdmax = np.log10(self.dmin), np.log10(self.dmax)
        self.Nrho   = int(1 + np.round((self.lgdmax - self.lgdmin)/self.lgddel))

        # psi range in particle units
        self.pmin   = 1e-5 #part.pmin/2 #????????????????????????????????????
        self.pmax   = 1e3  #part.pmax*2 #????????????????????????????????????
        self.lgpdel = 5E-2
        self.lgpmin = np.log10(self.pmin)
        self.lgpmax = np.log10(self.pmax)
        self.Npsi   = int(1 + np.round((self.lgpmax - self.lgpmin)/self.lgpdel))

        self.hpmtabA = {}
        self.hpmtabA['M']       = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['r']       = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['s']       = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['x']       = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['rho']     = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['rhosmth'] = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['psi']     = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['psismth'] = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['dm']      = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['dg']      = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['T']       = np.zeros((self.Nrad, self.Nmass))
        self.hpmtabA['Tsmth']   = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['vsq']     = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['P']       = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['Pnth']    = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['f']       = np.zeros((self.Nrad, self.Nmass))    
        self.hpmtabA['fp']      = np.zeros((self.Nrad, self.Nmass))

        self.hpmtabB = {}
        self.hpmtabB['M']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['r']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['s']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['x']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['rho']     = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['psi']     = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['dm']      = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['dg']      = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['T']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabB['Tsmth']   = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['vsq']     = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['P']       = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['Pnth']    = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['f']       = np.zeros((self.Nrho, self.Npsi))    
        self.hpmtabB['fp']      = np.zeros((self.Nrho, self.Npsi))

        self.hpmtabC = {}
        self.hpmtabC['T']       = np.zeros((self.Nrho, self.Npsi))
        self.hpmtabC['P']       = np.zeros((self.Nrho, self.Npsi))
        

        # Local and global tables
        print("Computing table_halo")
        for im in range(0,self.Nmass):
            self.table_halo(im)
    
        print("Computing table_icm")
        for ip in range(0, self.Npsi):
            self.table_icm(ip)

        #for ip in range(0, self.Npsi):
        #   self.table_igm(ip)
            
        
    def sigma_tophat(self, R):

        
        #cosmo_jax = jaxCosmology(Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96,
        #              Omega_k=0., w0=-1., wa=0.)


        #k = np.logspace(-3,-0.5)
        #pk_jax = linear_matter_power(cosmo_jax, k/cosmo_jax.h, a=1.0)

        '''
        var = 0
        for k in range(cosmo['Nkl']): #???????????????????????????????????????????
            kR   = cosmo['Plin'][0,k]*R #???????????????????????????????????????????
            dlnk = cosmo['Plin'][1,k]/cosmo['Plin'][0,k] #???????????????????????????????????????????
            var += cosmo['Plin'][2,k]*tophat_transform(kR)**2*dlnk #???????????????????????????????????????????
        sigma_tophat = cosmo['D1']*np.sqrt(var) #???????????????????????????????????????????
        return sigma_tophat
        ''';
        return 0.1

    def neff_pk(self,keff):

        #keff = float(keff)
        # Local variables
        gamma = self.cosmo['om']*self.cosmo['h']
        theta = self.cosmo['Tcmb0']/2.7
        # Eisenstein & Hu (1998)
        dlnP  = 0
        dlnk  = 0.01
        for i in [-1, 1]:
            k  = keff*np.exp(i*dlnk/2)
            q  = k*theta**2/gamma
            L0 = np.log(5.43656 + 1.8*q)
            C0 = 14.2 + 731/(1 + 62.5*q)
            T0 = L0/(L0 + C0*q**2)
            P  = k**self.cosmo['ns']*T0**2
            dlnP = dlnP + i*np.log(P)
        # Effective spectral index
        neff_pk = dlnP/dlnk
        return neff_pk


    def c200c_nfw(self, cosmo, M200c):
        # Local parameters
        kappa = 1.00
        alpha = 1.08
        beta  = 1.77
        phi0  = 6.58
        phi1  = 1.27
        eta0  = 7.28
        eta1  = 1.56
        # Lagrangian radius in comoving Mpc/h
        rho0      = self.cosmo['om']*self.cosmo['rhoc_ast']
        M         = M200c/Msun_cgs*self.cosmo['h']
        R         = (3*M/(4*np.pi*rho0))**(1./3)
        # Diemer & Kravtsov (2015), Diemer & Joyce (2019)
        k         = kappa*2*np.pi/R
        n         = self.neff_pk(k)
        nu        = 1.686/self.sigma_tophat(R) #????????????????????????????????????????
        cmin      = phi0 + phi1*n
        numin     = eta0 + eta1*n
        c200c_nfw = cmin/2*((numin/nu)**alpha + (nu/numin)**beta)
        return c200c_nfw

    def M500c_from_M200c(self, M200c, R200c, c200c):
        rs = R200c/c200c
        Am = M200c/(np.log(1+c200c) - c200c/(1+c200c))
        # Calc M500c iteratively
        x1  = 0
        x2  = c200c
        rho = 0
        while (abs(rho/self.cosmo['rho500c'] - 1) > 1E-4):
            # Calc average density
            x   = (x1 + x2)/2
            R   = x*rs
            M   = Am*(np.log(1+x) - x/(1+x))
            rho = M/(4*np.pi/3*R**3)
            # Check average density
            if (rho > self.cosmo['rho500c']):
                x1 = x
            else:
                x2 = x
        # Save
        M500c_from_M200c = M
        return M500c_from_M200c

    def P500c_gnfw(self,M):
        # Arnaud et al
        h70    = self.cosmo['h']/0.7
        hz     = self.cosmo['Hz']/(H0_cgs*self.cosmo['h'])
        P500c_gnfw = 1.65*eV2erg*(self.icm['mue']/self.icm['mu'])*np.power(hz, 8./3) \
                   * (M/(3E14*Msun_cgs/h70))**(2./3)*np.power(h70, 2)
        return P500c_gnfw

    def rho_nfw(self,x, M, R, c):
        return M / (4 * np.pi * R ** 3) * c ** 3 / (np.log(1 + c) - c / (1 + c)) / (x * (1 + x) ** 2)

    def acc_nfw(self,x, M, R, c):
        return -G_cgs*M/(R*x/c)**2*(np.log(1+x) - x/(1+x))/(np.log(1+c) - c/(1+c))


    def P_gnfw(self,x):
        # Dimensionless pressure profile
        P_gnfw = self.icm['p0']                           \
                 /(self.icm['c500']*x)**self.icm['gamma']         \
                 /(1+(self.icm['c500']*x)**self.icm['alpha'])     \
                 **((self.icm['beta']-self.icm['gamma'])/self.icm['alpha'])
        return P_gnfw

    def fnth_gnfw(self,x):
        L = 0.00
        b = 0.913
        k = 0.244
        d = 1.244
        fnth_gnfw = 1-L+(L-b)*np.exp(-(k*x)**d)
        return fnth_gnfw

    def Pnth_gnfw(self,x):
        def fnth_gnfw(x):
            pass
        def P_gnfw(x):
            pass
        fnth      = self.fnth_gnfw(x)
        Pnth_gnfw = fnth/(1 - fnth)*self.P_gnfw(x)
        return Pnth_gnfw


    def dPdx_gnfw(self,x):
        # Dimensionless pressure profile derivative
        return -self.P_gnfw(x) * (self.icm['gamma']/x + self.icm['c500']*(self.icm['beta']-self.icm['gamma'])*(self.icm['c500']*x)**(self.icm['alpha']-1)/(1+(self.icm['c500']*x)**self.icm['alpha']))


    def dPdx_tot(self,x):
        L, b, k, d = 0.00, 0.913, 0.244, 1.244
        fth = L-(L-b)*np.exp(-(k*x)**d)
        Pth = self.P_gnfw(x) # Assuming P_gnfw is defined elsewhere
        dfthdx   = (L-b)*np.exp(-(k*x)**d)*(k*x)**(d-1.0)*d*k     
        dPthdx   = self.dPdx_gnfw(x) # Assuming dPdx_gnfw is defined elsewhere
        dPdx_tot = (dPthdx*fth-dfthdx*Pth)/fth**2
        return dPdx_tot


    def psi_nfw(self,x,M,R,c):
        G_cgs = 6.6743e-8
        rhos  = M/(4*np.pi*R**3)*c**3/(np.log(1+c) - c/(1+c))
        rs    = R/c
        Apsi  = 4*np.pi*G_cgs*rhos*rs
        if abs(x-1) > 1E-6:
            psi_nfw = Apsi*np.log(x)/(x**2 - 1)
        else:
            psi_nfw = Apsi*(1-x/2)
        return psi_nfw


    def rho_ratio(x, spline):
        xs = [x]
        ys = [spline_interp(spline, xs)]
        # Spline
        rho_ratio = min(max(ys[0], 0.5), 2.0)
        return rho_ratio

    def psi_ratio(x, spline):
        xs = [x]
        ys = spline_interp(spline, xs)
        psi_ratio = min(max(ys[0], 0.5), 2.)
        return psi_ratio

    def T_ratio(x, spline):
        xs = [x]
        ys = spline_interp(spline, xs)
        T_ratio = min(max(ys[0], 0.1), 2.)
        return T_ratio


    def table_halo2(self):
        # Calls c200c_nfw(), M500c_from_M200c(), P500c_gnfw(), rho_nfw(), acc_nfw()
        im, ir        = np.arange(self.Nmass), np.arange(self.Nrad)
        im_arr,ir_arr = np.meshgrid(im,ir)
        im_arr = im_arr.flatten()
        ir_arr = ir_arr.flatten()
        
        M200c  = 10**((im_arr-1)*self.lgMdel + self.lgMmin) / self.cosmo['h'] * Msun_cgs
        R200c  = (M200c/(4*np.pi/3*self.cosmo['rho200c']))**(1./3)
        c200c  = self.c200c_nfw(self.cosmo,M200c)
        rs     = R200c/c200c

        pdb.set_trace()

        # M500c in cgs
        M500c  = self.M500c_from_M200c(M200c,R200c,c200c)
        R500c  = (M500c/(4*np.pi/3*self.cosmo['rho500c']))**(1./3)
        c500c  = R500c/rs
        P500c  = self.P500c_gnfw(M500c)
        

        # Loop over radius


    
    def table_halo(self,im):
        # Calls c200c_nfw(), M500c_from_M200c(), P500c_gnfw(), rho_nfw(), acc_nfw()
        k_cgs = 1.38066e-16
        M200c = 10**((im-1)*self.lgMdel + self.lgMmin) / self.cosmo['h'] * Msun_cgs
        R200c = (M200c/(4*np.pi/3*self.cosmo['rho200c']))**(1./3)
        c200c = self.c200c_nfw(self.cosmo,M200c)
        rs    = R200c/c200c

        # M500c in cgs
        M500c = self.M500c_from_M200c(M200c,R200c,c200c)
        R500c = (M500c/(4*np.pi/3*self.cosmo['rho500c']))**(1./3)
        c500c = R500c/rs
        P500c = self.P500c_gnfw(M500c)

        # Loop over radius

        for ir in range(self.Nrad):
            # Radius
            r = 10**((ir-1)*self.lgrdel + self.lgrmin)*R200c
            s = r/R500c
            x = r/rs
            
            # NFW
            dm = self.rho_nfw(x,M200c,R200c,c200c)
            f  = self.acc_nfw(x,M200c,R200c,c200c)
            f  = abs(f)

            # GNFW
            P       = P500c*self.P_gnfw(s)
            Pnth    = P500c*self.Pnth_gnfw(s)
            dPdr    = P500c*self.dPdx_gnfw(s)/R500c
            dPtotdr = P500c*self.dPdx_tot(s)/R500c
            dg      = abs(dPtotdr/f)
            fp      = abs(dPdr/dg)
            T       = P/(k_cgs*dg/self.icm['mu'])
            vsq     = 3*Pnth/dg
            
            # HPM
            rho = dm
            psi = abs(self.psi_nfw(x,M200c,R200c,c200c))
            
            # Spline???????????????????????????????????????????
            """
            if (hpmcal['on']):
                xs      = np.log10(r/unit['len']*unit['part_to_mesh'])
                i       = 1 + int((np.log10(M200c) - hpmcal['lgMmin'])/hpmcal['lgMdel'])
                i       = min(max(i,1),hpmcal['Nmass'])
                rhosmth = rho*self.rho_ratio(xs,hpmcal['spl'][i,0])
                psismth = psi*self.psi_ratio(xs,hpmcal['spl'][i,1])
                Tsmth   = T  *self.T_ratio(xs,hpmcal['spl'][i,2])
            else:
                rhosmth = rho
                psismth = psi
                Tsmth   = T
            """;
            rhosmth = rho
            psismth = psi
            Tsmth   = T         
            # Save in part units and log form
            #print(ir,im)
            self.hpmtabA['M'][ir,im]       = np.log10(M200c  /self.unit['mass']  )
            self.hpmtabA['r'][ir,im]       = np.log10(r      /R200c              )
            self.hpmtabA['s'][ir,im]       = np.log10(s                          )
            self.hpmtabA['x'][ir,im]       = np.log10(x                          )
            self.hpmtabA['rho'][ir,im]     = np.log10(rho    /self.cosmo['rhom'] )
            self.hpmtabA['rhosmth'][ir,im] = np.log10(rhosmth/self.cosmo['rhom'] )
            self.hpmtabA['psi'][ir,im]     = np.log10(psi    /self.unit['psi']   )
            self.hpmtabA['psismth'][ir,im] = np.log10(psismth/self.unit['psi']   )
            self.hpmtabA['dm'][ir,im]      = np.log10(dm     /self.cosmo['rhom'] )
            self.hpmtabA['dg'][ir,im]      = np.log10(dg     /self.cosmo['rhob'] )
            self.hpmtabA['T'][ir,im]       = np.log10(T      /self.unit['temp']  )
            self.hpmtabA['Tsmth'][ir,im]   = np.log10(Tsmth  /self.unit['temp']  )
            self.hpmtabA['vsq'][ir,im]     = np.log10(vsq    /self.unit['vel']**2)
            self.hpmtabA['P'][ir,im]       = np.log10(P      /self.unit['pres']  )
            self.hpmtabA['Pnth'][ir,im]    = np.log10(Pnth   /self.unit['pres']  )
            self.hpmtabA['f'][ir,im]       = np.log10(f      /self.unit['acc']   )
            self.hpmtabA['fp'][ir,im]      = np.log10(fp     /self.unit['acc']   )
    
       

    
    def table_icm(self,ip):
        # Local variables
        '''
        i = np.int32(0)
        j = np.int32(0)
        id = np.int32(0)
        im = np.int32(0)
        ir = np.int32(0)
        lgd = np.float64(0)
        lgp = np.float64(0)
        drho = np.float64(0)
        drhomin = np.finfo(np.float64).max
        dpsi = np.float64(0)
        dpsimin = np.finfo(np.float64).max
        ''';
        data = np.zeros(14, dtype=np.float64)
        # psi in particle units 
        lgp = (ip-1)*self.lgpdel + self.lgpmin
        
        # rho, psi original
        for id in range(0, self.Nrho):
            print(ip,id)
            # Particle density relative to average
            lgd = (id-1)*self.lgddel + self.lgdmin
            # Init
            #data = 0
            drhomin = np.finfo(np.float64).max
            dpsimin = np.finfo(np.float64).max
            # Find minimum rho difference
            #for im in range(0, self.Nmass):
            #for ir in range(0, self.Nrad):
            drho    = abs((self.hpmtabA['rho'] - lgd)/self.lgddel)
            drhomin = np.minimum(drhomin,drho)

            drhomin = np.maximum(drhomin,0.5)
            # Find minimum psi difference
            for im in range(0, self.Nmass):
                for ir in range(0, self.Nrad):
                    drho = abs((self.hpmtabA['rho'][ir,im] - lgd)/self.lgddel)
                    if (drho <= drhomin):
                        dpsi = abs((self.hpmtabA['psi'][ir,im] - lgp)/self.lgpdel)
                        dpsimin = min(dpsimin,dpsi)
            dpsimin = max(dpsimin,0.5)

            # Loop over mass and radius
            for im in range(0, self.Nmass):
                for ir in range(0, self.Nrad):
                    drho = abs((self.hpmtabA['rho'][ir,im] - lgd)/self.lgddel)
                    dpsi = abs((self.hpmtabA['psi'][ir,im] - lgp)/self.lgpdel)
                    if (drho <= drhomin and dpsi <= dpsimin):
                        data[0] += 1
                        data[1]  += self.hpmtabA['M'][ir,im] 
                        data[2]  += self.hpmtabA['r'][ir,im] 
                        data[3]  += self.hpmtabA['s'][ir,im] 
                        data[4]  += self.hpmtabA['x'][ir,im] 
                        data[5]  += self.hpmtabA['dm'][ir,im] 
                        data[6]  += self.hpmtabA['dg'][ir,im] 
                        data[7]  += self.hpmtabA['T'][ir,im] 
                        data[8]  += self.hpmtabA['Tsmth'][ir,im] 
                        data[9]  += self.hpmtabA['vsq'][ir,im] 
                        data[10] += self.hpmtabA['P'][ir,im] 
                        data[11] += self.hpmtabA['Pnth'][ir,im] 
                        data[12] += self.hpmtabA['f'][ir,im] 
                        data[13] += self.hpmtabA['fp'][ir,im]

            # Save in log form for interpolation
            self.hpmtabB['rho'][id,ip]   = lgd
            self.hpmtabB['psi'][id,ip]   = lgp
            self.hpmtabB['M'][id,ip]     = data[1]/data[0]
            self.hpmtabB['r'][id,ip]     = data[2]/data[0]
            self.hpmtabB['s'][id,ip]     = data[3]/data[0]
            self.hpmtabB['x'][id,ip]     = data[4]/data[0]
            self.hpmtabB['dm'][id,ip]    = data[5]/data[0]
            self.hpmtabB['dg'][id,ip]    = data[6]/data[0]
            self.hpmtabB['T'][id,ip]     = data[7]/data[0]
            self.hpmtabB['Tsmth'][id,ip] = data[8]/data[0]
            self.hpmtabB['vsq'][id,ip]   = data[9]/data[0]
            self.hpmtabB['P'][id,ip]     = data[10]/data[0]
            self.hpmtabB['Pnth'][id,ip]  = data[11]/data[0]
            self.hpmtabB['f'][id,ip]     = data[12]/data[0]
            self.hpmtabB['fp'][id,ip]    = data[13]/data[0]


    def table_igm(self,ip):
        gm1 = self.igm.gamma - 1 #?????????????
        # Smoothing range
        D1 = self.icm.Dmin/2     #????????????? 
        D2 = self.icm.Dmin*2     #?????????????
        lgD1 = np.log10(D1)
        lgD2 = np.log10(D2)
        lgdD = lgD2 - lgD1
        # Table b
        for id in range(self.Nrho):
            if hpmtabB['rho'][id,ip] < lgD1:
                # IGM
                d = 10**hpmtabB['rho'][id,ip]
                rho = d*cosmo['rhob']
                Tigm = igm.T0*d**gm1           #??????????????
                Pigm = (rho/igm.mu)*k_cgs*Tigm #??????????????
                hpmtabB['T'][id,ip] = np.log10(Tigm/unit['temp'])
                hpmtabB['P'][id,ip] = np.log10(Pigm/unit['pres'])

            elif hpmtabB['rho'][id,ip] < lgD2:
                # IGM
                d = 10**hpmtabB['rho'][id,ip]
                rho  = d*cosmo['rhob']
                Tigm = igm.T0*d**gm1 #???????
                Pigm = (rho/igm.mu)*k_cgs*Tigm #????
                # ICM
                Ticm = 10**hpmtabB['T'][id,ip]*unit['temp']
                Picm = 10**hpmtabB['P'][id,ip]*unit['pres']
                Ticm = max(Ticm,Tigm)
                Picm = max(Picm,Pigm)
                # Smooth IGM and ICM
                x = (np.log10(d) - lgD1)/lgdD
                T = (Ticm**x)*(Tigm**(1-x))
                P = (Picm**x)*(Pigm**(1-x))
                hpmtabB['T'][id,ip] = np.log10(T/unit['temp'])
                hpmtabB['P'][id,ip] = np.log10(P/unit['pres'])

        # Table c
        for id in range(self.Nrho):
            if hpmtabC['rho'][id,ip] < lgD1:
                # IGM
                d = 10**hpmtab.c[id,ip].rho
                rho  = d*cosmo['rhob']
                Tigm = igm.T0*d**gm1 ###??????????????????????????
                hpmtabC['T'][id,ip]     = log10(Tigm/unit['temp'])
                hpmtabC['Tsmth'][id,ip] = log10(Tigm/unit['temp'])

            elif hpmtabC['rho'][id,ip] < lgD2:
                # IGM
                d = 10**hpmtabC['rho'][id,ip]
                rho = d*cosmo['rhob']
                Tigm = igm.T0*d**gm1###????????????????????
                # ICM
                Ticm = max(10**hpmtabC['T'][id,ip]*unit['temp'],Tigm)
                Tsmth = max(10**hpmtabC['Tsmth'][id,ip]*unit['temp'],Tigm)
                # Smooth IGM and ICM
                x = (np.log10(d) - lgD1)/lgdD
                Ticm  = (Ticm**x)*(Tigm**(1-x))
                Tsmth = (Tsmth**x)*(Tigm**(1-x))
                hpmtabC['T'][id,ip]     = np.log10(Ticm/unit['temp'])
                hpmtabC['Tsmth'][id,ip] = np.log10(Tsmth/unit['temp'])


def T_from_hpmtable(rho, psi):
    import math
    # Function arguments
    # Local variables
    i1, i2, j1, j2 = 0, 0, 0, 0
    x, y, dx1, dx2, dy1, dy2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Bilinear interpolation
    x = (math.log10(rho) - hpmtab['lgdmin']) / hpmtab['lgddel']
    i1 = min(max(1 + int(x), 1), hpmtab['Nrho'] - 1)
    i2 = 1 + i1
    dx1 = min(max(i1 - x, 0.0), 1.0)
    dx2 = 1 - dx1
    y = (math.log10(psi) - hpmtab['lgpmin']) / hpmtab['lgpdel']
    j1 = min(max(1 + int(y), 1), hpmtab['Npsi'] - 1)
    j2 = 1 + j1
    dy1 = min(max(j1 - y, 0.0), 1.0)
    dy2 = 1 - dy1
    # Temperature in part units
    T_from_hpmtable = 10 ** (hpmtab['c'][i1][j1]['T'] * dx1 * dy1 +
                             hpmtab['c'][i2][j1]['T'] * dx2 * dy1 +
                             hpmtab['c'][i1][j2]['T'] * dx1 * dy2 +
                             hpmtab['c'][i2][j2]['T'] * dx2 * dy2)
    return T_from_hpmtable



def Tsmth_from_hpmtable(rho, psi):
    import math
    # Function arguments
    # Local variables
    i1, i2, j1, j2 = 0, 0, 0, 0
    x, y, dx1, dx2, dy1, dy2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Bilinear interpolation
    x = (math.log10(rho) - hpmtab['lgdmin']) / hpmtab['lgddel']
    i1 = min(max(1 + int(x), 1), hpmtab['Nrho'] - 1)
    i2 = 1 + i1
    dx1 = min(max(i1 - x, 0.0), 1.0)
    dx2 = 1 - dx1
    y = (math.log10(psi) - hpmtab['lgpmin']) / hpmtab['lgpdel']
    j1 = min(max(1 + int(y), 1), hpmtab['Npsi'] - 1)
    j2 = 1 + j1
    dy1 = min(max(j1 - y, 0.0), 1.0)
    dy2 = 1 - dy1
    # Temperature in part units
    Tsmth_from_hpmtable = 10 ** (hpmtab['c'][i1][j1]['Tsmth'] * dx1 * dy1 +
                                 hpmtab['c'][i2][j1]['Tsmth'] * dx2 * dy1 +
                                 hpmtab['c'][i1][j2]['Tsmth'] * dx1 * dy2 +
                                 hpmtab['c'][i2][j2]['Tsmth'] * dx2 * dy2)
    return Tsmth_from_hpmtable


