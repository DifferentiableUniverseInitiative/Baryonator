# Constants
G = 6.67e-11             # m^3/kg/s^2
G = G*1.99e30            # m^3/Msun/s^2
G = G/(3.086e22)**2/1000 # (km)(Mpc^2)/Msun/s^2
G = G*3.15e16            # (km)(Mpc^2)/Msun/s/Gyr

sigT   = 6.65e-29 # m^2
me     = 9.11e-31 # kg
c      = 3e8      # m^2/s^2
mH_cgs = 1.67223e-24

mp = 1.67262192e-27 # kg
kB = 1.380549e-23   # J/K, kg*m^2/s^2/K [ML^2/T^2/K]
kB_over_mp = kB  * (3.24e-23)**2 / mp

icm = {}
icm['XH']     = 0.76
icm['YHe']    = 0.24
icm['mu']     = mH_cgs/(2*icm['XH'] + 3*icm['YHe']/4)
icm['mue']    = mH_cgs/(  icm['XH'] + 2*icm['YHe']/4)
#icm['mue']  = mH_cgs/(  cosmo['XH'] + 4*cosmo['YHe']/4 + 676*icm%Zxry*cosmo%XH)
icm['p0']     = 8.403
icm['c500']   = 1.177
icm['gamma']  = 0.3081
icm['alpha']  = 1.0510
icm['beta']   = 5.4905