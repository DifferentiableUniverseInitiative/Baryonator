import hpmtable as hpmtable
import numpy as np

G_cgs    = 6.67259e-08
H0_cgs   = 3.24086e-18
rhoc_ast = 2.77550e+11
mH_cgs   = 1.67223e-24
rhoc_cgs = 1.87890e-29

a        = 1

cosmo       = {}
cosmo['h']  = 0.7
cosmo['H0'] = 70
cosmo['or'] = 8.5e-5
cosmo['om'] = 0.3
cosmo['ol'] = 0.7
cosmo['ob'] = 0.045
cosmo['rhoc_ast'] = rhoc_ast 
cosmo['ns'] = 0.96
cosmo['Tcmb0'] = 2.725
cosmo['fb'] = cosmo['ob']/cosmo['om']

#cosmo['Nkl']  =
#cosmo['Plin'] = np.loadtxt('')

#gf = growthfactors_of_z(0)
#cosmo['delta0'] = gf(1)

#gf = growthfactors_of_z(z)
#cosmo['D1']     = gf(1)/cosmo['delta0']

hsq      = cosmo['or']/a**4 + cosmo['om']/a**3 + cosmo['ol']
oma      = cosmo['om']/a**3/hsq**2
cosmo['H0'] = H0_cgs*cosmo['h']
cosmo['Hz'] = cosmo['H0']*np.sqrt(hsq)

cosmo['rhocrit0'] = 3*cosmo['H0']**2/(8*np.pi*G_cgs)
cosmo['rhocrit']  = 3*cosmo['Hz']**2/(8*np.pi*G_cgs)
cosmo['rho200c']  = 200*cosmo['rhocrit']
cosmo['rho500c']  = 500*cosmo['rhocrit']
cosmo['rhom0']    = rhoc_cgs*cosmo['om']*cosmo['h']**2
cosmo['rhob0']    = cosmo['fb']*cosmo['rhom0']
cosmo['rhom']     = cosmo['rhom0']/a**3
cosmo['rhob']     = cosmo['rhob0']/a**3

icm = {}
cosmo['XH']  = 0.76
cosmo['YHe'] = 0.24
icm['mu']   = mH_cgs/(2*cosmo['XH'] + 3*cosmo['YHe']/4)
icm['mue']  = mH_cgs/(  cosmo['XH'] + 2*cosmo['YHe']/4)
#icm['mue']  = mH_cgs/(  cosmo['XH'] + 4*cosmo['YHe']/4 + 676*icm%Zxry*cosmo%XH)
icm['p0']     = 8.403
icm['c500']   = 1.177 
icm['gamma']  = 0.3081 
icm['alpha']  = 1.0510
icm['beta']   = 5.4905 

igm={}

unit          = {}
sim_Np1d      = 128 # Why do we need this???
Mpc2cm        = 3.08560e+24
Lbox          = 128

unit['len']   = (a*Lbox/cosmo['h']/sim_Np1d)*Mpc2cm
unit['rho']   = rhoc_cgs*cosmo['om']*cosmo['h']**2/a**3
unit['mass']  = unit['rho']*unit['len']**3 
unit['time']  = 2/(3*H0_cgs*cosmo['h']*np.sqrt(cosmo['om']))*a**2
unit['acc']   = unit['len']/unit['time']**2
unit['psi']   = unit['acc']
unit['vel']   = unit['len']/unit['time']
unit['temp']  = unit['vel']**2
unit['pres']  = unit['rho']*unit['temp']

hpmtable.hpm_table(cosmo,icm,igm,unit)
