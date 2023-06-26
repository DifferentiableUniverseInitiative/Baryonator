
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint, cic_paint_2d, cic_read, compensate_cic
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.lensing import density_plane, convergence_Born
from jaxpm.kernels import fftk, gradient_kernel, laplace_kernel, longrange_kernel
from jax.scipy.ndimage import map_coordinates

import tensorflow_probability as tfp

from hpmtable2 import *

import numpy as np
import pdb

# this needs to be jaxed
from scipy.interpolate import RegularGridInterpolator

from jax import config
config.update("jax_enable_x64", True)

### input parameters (free)
# currently this is just controlling the EDG part, later we want to also connect it to the HPM table

params_camels_optimized = jnp.array([0.93629056,2.0582693,0.46008348])
params_camels_optimized_extreme = jnp.array([100.93629056,100.0582693,0.46008348])


### input parameters (fixed)

box_size = [200.,200.,4000.]    #[200.,200.,4000.] Transverse comoving size of the simulation volume Mpc/h
#nc = [32, 32, 64]             #[64, 64, 1280.] Number of transverse voxels in the simulation volume
nc = [64, 64, 1280]             #[64, 64, 1280.] Number of transverse voxels in the simulation volume

lensplane_width = 102.5         # Width of each lensplane
n_lens = int(box_size[-1] // lensplane_width)
field_size = 5                  # Size of the lensing field in degrees
field_npix = 64                # Number of pixels in the lensing field
z_source = jnp.linspace(0,2.5,39)    # Source planes -- this needs to change
r = jnp.linspace(0., box_size[-1], n_lens+1)
r_center = 0.5*(r[1:] + r[:-1])

z_start = 10                    # Redshift where initial conditions are generated
a_start = 1./(1+z_start)

# cosmology (can be freed if need)
Omega_c = 0.2589
Omega_b = 0.0486
Omega_m = Omega_c+Omega_b
sigma8 = 0.8159
h = 0.7

G = 6.67e-11             # m^3/kg/s^2
G = G*1.99e30            # m^3/Msun/s^2
G = G/(3.086e22)**2/1000 # (km)(Mpc^2)/Msun/s^2
G = G*3.15e16            # (km)(Mpc^2)/Msun/s/Gyr

# gas parameters
sigT  = 6.65e-29 # m^2
me    = 9.11e-31 # kg
c     = 3e8      # m^2/s^2

mH_cgs   = 1.67223e-24

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


seed = 100

import astropy.units as u
xgrid, ygrid = np.meshgrid(np.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                           np.linspace(0, field_size, field_npix, endpoint=False)) # range of Y coordinates

coords = np.stack([xgrid, ygrid], axis=0)*u.deg
coords2 = coords.reshape([2, -1]).T.to(u.rad)

# if we want to ignore some dynamical parameters in the functions
#from functools import partial
#@partial(jax.jit, static_argnums=[2])

def create_nbody(cosmo, cosmo2):

    # Retrieve the scale factor corresponding to these distances
    a = jc.background.a_of_chi(cosmo, r)
    a_center = jc.background.a_of_chi(cosmo, r_center)

    # Then one step per lens plane
    stages = a_center[::-1]

    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(nc, box_size, pk_fn, seed=jax.random.PRNGKey(seed))

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),axis=-1).reshape([-1,3])

    # Initial displacement (need to redefine cosmo?)
    dx, p, f = lpt(cosmo2, initial_conditions, particles, a_start)

    # Evolve the simulation forward
    res = odeint(make_ode_fn(nc), [particles+dx, p],
             jnp.concatenate([jnp.atleast_1d(a_start), stages]), cosmo, rtol=1e-5, atol=1e-5)

    return res, a_center

@jax.jit
def calculat_Msun_per_particle(particle_list, a_center, cosmo):
    Msun_per_particle = []
    Rho_m_mean = []
    rho_crit = 8.5 * 10**-27 #kg/m3 # replace with cosmology object

    for i in range(len(particle_list)):

        # rho_m_mean in kg/m^3
        rho_m_mean = rho_crit*jc.background.Omega_m_a(cosmo, a_center[i])
        # rho_m_mean in (Msun/h)/(Mpc/h)^3
        rho_m_mean = rho_m_mean *(3.086*10**22/h)**3 * (1.989*10**30/h)**(-1)  

        Mpart = rho_m_mean * box_size[0] * box_size[1] * box_size[2] / len(particle_list[i])
    
        Msun_per_particle.append(Mpart)
        Rho_m_mean.append(rho_m_mean)

    return Msun_per_particle, Rho_m_mean


@jax.jit
def egd_correction(delta, pos, params):
    """
    Will compute the EGD displacement as a function of density traced by the 
    input particles.
    params contains [amplitude, scale, gamma]
    """
    kvec = fftk(nc)
    alpha, kl, gamma = params

    # Compute a temperature-like map from density contrast
    T = (delta+1)**gamma
    
    # Apply FFT to apply filtering
    T_k = jnp.fft.rfftn(T)
    filtered_T_k = gaussian_kernel(kvec, kl) * T_k # This applies a Gaussian smoothing

    # Compute derivatives of this filtered T-like field at the position of particles
    dpos_egd = jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*filtered_T_k), pos) 
                      for i in range(3)],axis=-1)
    
    # Apply overal scaling of these displacements
    dpos_egd = - alpha * dpos_egd
   
    return dpos_egd

def gaussian_kernel(kvec, k_smooth):
  """
  Computes a gaussian kernel
  """
  kk = sum(ki**2 for ki in kvec)
  return jnp.exp(-0.5 * kk / k_smooth**2)

def create_kgrid(nx, ny, nz, lx, ly, lz):
    """
    Create a 3D k grid for Fourier space calculations
    """
    xres   = lx/nx #Mpc/h
    yres   = ly/ny #Mpc/h
    zres   = lz/nz #Mpc/h

    kx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, xres))# h/Mpc
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(ny, yres))# h/Mpc
    kz = jnp.fft.fftshift(jnp.fft.fftfreq(nz, zres))# h/Mpc
    
    mg = jnp.meshgrid(kx,ky,kz)

    km  = jnp.sqrt(mg[0]**2+mg[1]**2+mg[2]**2)
    #k2[nx/2,ny/2,nz/2]=1.

    return km

def make_HPM_table_interpolator():
    f  = open('../notebooks/hpm_table_z00.01.dat', 'rb')
    dd = np.fromfile(f, dtype=np.int32,count=2)
    X  = np.fromfile(f, dtype=np.float32) 
    Y  = X.reshape((dd[1],dd[0],14))

    # Set up an interpolator
    drho = Y[0,:,4]
    psi  = Y[:,0,5]

    Tt   = Y[:,:,8].astype(np.float64)  # [K]

    '''
    Pt   = Y[:,:,10].astype(np.float64) # [keV/cm^3]
    Pt   = Pt*1000                      # [eV/cm^3]
    Pt   = Pt*(3e8)**2                  # [(eV/c^2)*(m/s)^2/cm^3]
    Pt   = Pt*1.783e-36                 # [(kg)*(m/s)^2/cm^3]
    Pt   = Pt/1.99e30                   # [(Msun)*(m/s)^2/cm^3]
    Pt   = Pt/(1000)**2                 # [(Msun)*(km/s)^2/cm^3]
    Pt   = Pt*(3.086e24)**3             # [(Msun)*(km/s)^2/Mpc^3]
    Pt   = Pt/h**2                      # [h^2(Msun)*(km/s)^2/Mpc^3]
    '''

    Pt   = Y[:,:,10].astype(np.float64) # [keV/cm^3]
    Pt   = Pt*1000                      # [eV/cm^3]
    Pt   = Pt*(1.602e-19)               # [kg m^2/s^2 /cm^3]
    Pt   = Pt/1.98847e30                # [Msun m^2/s^2 /cm^3]
    Pt   = Pt/1000/1000                 # [Msun km^2/s^2 /cm^3]
    Pt   = Pt*(3.086e24)**3             # [Msun km^2/s^2 /Mpc^3]
    Pt   = Pt/h**2                      # [h^2(Msun)*(km/s)^2/Mpc^3]

    #--------------> Pt is x10^10 high

    intpT = RegularGridInterpolator((psi,drho), Tt, bounds_error=False, fill_value=0)
    intpP = RegularGridInterpolator((psi,drho), Pt, bounds_error=False, fill_value=0)
    return intpT, intpP

# @jax.jit
def HPM_GPmodel(rho,psi,a, logMmin=8,logMmax=16,NM=50,rmin=0.1,rmax=4,Nr=50):
    """Takes rho and psi and extracts the inverse mapping via
     HPM table to predict the value of T & P.

    Parameters
    ----------
    rho : float, arr
     density values 
    psi : float, arr
     fscalar valaues

    Returns
    ----------
    T   : float, arr
     Temperature in units of K
    P   : float, arr
     Pressure in units of ????
    """
    
    verbose = False

    # First construct a table to map M/r -> rho/psi
    batched_r  = jax.vmap(table_halo,in_axes=[None, None, None, None, 0])
    batched_Mr = jax.vmap(batched_r, in_axes=[None,None,None,0,None]); del batched_r
    m_grid     = jnp.logspace(logMmin,logMmax,NM) # Msun/h
    r_grid     = jnp.linspace(rmin,rmax,Nr)# unitless, to be multiplied by R200c
    res        = batched_Mr(cosmo,a,icm, m_grid.flatten(), r_grid.flatten());del batched_Mr
    M,rx,_,_,rho,psi,T,P = res
    del res

    # Use GP to make an interpolate to apply a reverse mapping
    tfb = tfp.bijectors
    tfd = tfp.distributions
    psd_kernels  = tfp.math.psd_kernels

    index_points = jnp.array([np.log10(rho), np.log10(psi)]).reshape([-1,2])
    kern         = psd_kernels.ExponentiatedQuadratic()

    model_T = tfd.GaussianProcessRegressionModel( kern,
                                                  index_points=index_points,
                                                  observation_index_points=jnp.stack([np.log10(rho).flatten(), np.log10(psi).flatten()], axis=1),
                                                  observations=np.log10(T).flatten(),
                                                 )

    model_P = tfd.GaussianProcessRegressionModel( kern,
                                                  index_points=index_points,
                                                  observation_index_points=jnp.stack([np.log10(rho).flatten(), np.log10(psi).flatten()], axis=1),
                                                  observations=np.log10(P).flatten(),
                                                )
    return 10**model_T.mean(), 10**model_P.mean()




# need to write an equivalent for tSZ
def tSZ_Born(cosmo,
            pressure_planes,
            coords,
            z_source):
  """
  Compute the Born tSZ
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """


  # /1e9*lensplane_width
  #   pressure_plane = pressure_plane*sigT/(3.0886e22)**2
  #   pressure_plane = pressure_plane/me*1.99e30
  #   pressure_plane = pressure_plane*(1000)**2/c**2*h

  # Compute constant prefactor:
  constant_factor = 1./1e9 * sigT/(3.0886e22)**2 /me*1.99e30 *(1000)**2/c**2*h 

  # Compute comoving distance of source galaxies
  r_s = jax_cosmo.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  tsz = 0
  for entry in pressure_planes:
    r = entry['r']; a = entry['a']; p = entry['plane']
    dx = entry['dx']; dz = entry['dz']
    # Normalize density planes
    normalization = dz * r / a
    p = p * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p, 
                         coords * r / dx - 0.5, 
                         order=1, mode="wrap")

    tsz += im * jnp.clip(1. - (r / r_s), 0, 1000).reshape([-1, 1, 1])

  return tsz

### main part of code ###############################

cosmo = jc.Planck15(Omega_c=Omega_c, Omega_b=Omega_b, sigma8=sigma8, h=h)
cosmo2 = jc.Planck15(Omega_c=Omega_c, Omega_b=Omega_b, sigma8=sigma8, h=h)

res, a_center = create_nbody(cosmo, cosmo2)
print('end of N body creation')

Msun_per_particle, Rho_m_mean = calculat_Msun_per_particle(res[0], a_center, cosmo)
print('end of calculation of M and rho')

# first loop to get all the 3D quantities
Total_mass = []
Total_delta = []
DM_mass = []
Temperature = []
Pressure = []

intpT, intpP = make_HPM_table_interpolator()

for i in range(n_lens):

    print('Lens redshift', i)

    Nparticles = len(res[0][i])
    inds = jax.random.shuffle(jax.random.PRNGKey(seed), jnp.arange(0,Nparticles-1))
    split = int(Omega_b/Omega_m*Nparticles)
    egd_pos_gas = res[0][i][inds[:split]]
    egd_pos_dm  = res[0][i][inds[split:]]

    # may be able to remove some of these that are not used later
    total_mass_dmo = cic_paint(jnp.zeros((nc[0],nc[1],nc[2])), res[0][i])
    total_delta_dmo = total_mass_dmo/total_mass_dmo.mean() - 1
    egd_rho_dm = cic_paint(jnp.zeros((nc[0],nc[1],nc[2])), egd_pos_dm) # this is still in number of particles
    egd_rho_gas = cic_paint(jnp.zeros((nc[0],nc[1],nc[2])), egd_pos_gas)

    total_particles_egd = cic_paint(egd_rho_dm,
                            egd_pos_gas + egd_correction(total_delta_dmo, egd_pos_gas, params_camels_optimized))
    total_mass_egd = total_particles_egd * Msun_per_particle[i]
    total_delta_egd = total_mass_egd/total_mass_egd.mean() - 1

    F_rhom = jnp.fft.fftshift(jnp.fft.fftn(total_mass_egd))
    kg = create_kgrid(total_mass_egd.shape[0], total_mass_egd.shape[1], total_mass_egd.shape[2], lx=box_size[0], ly=box_size[1], lz=box_size[2])

    kg = kg.at[kg==0].set(jnp.inf)
    F_fscalar = 2*jnp.pi**2*G*F_rhom/kg # m^3/kg/s^2 (Msun/h)/(Mpc/h)^2  
    R_fscalar = jnp.fft.ifftn(jnp.fft.ifftshift(F_fscalar)) 

    # we want km/s/Gyr [L/T^2] -- in the paper
    R_fscalar -= jnp.min(R_fscalar.real)  
    R_fscalar_new = R_fscalar * (10**(-3))**3 * (1.989* 10**30 / h)  * 3.1536 * 10**16 / (3.086*10**19/h)**2 

    # Tf = intpT( np.c_[(R_fscalar_new.real).flatten(), (total_mass_egd/np.mean(total_mass_egd)).flatten() ]).reshape((nc[0],nc[1],nc[2]))
    # Pf = intpP( np.c_[(R_fscalar_new.real).flatten(), (total_mass_egd/np.mean(total_mass_egd)).flatten() ]).reshape((nc[0],nc[1],nc[2]))
    
    # pdb.set_trace()

    # here we want to interpolate back to the particles, run the HPM tables from the particles, and later project it onto 2D 
    Tf,Pf = HPM_GPmodel((total_mass_egd/np.mean(total_mass_egd)).flatten(), (R_fscalar_new.real).flatten(), a_center[i])
    Tf = Tf.reshape((nc[0],nc[1],nc[2]))
    Pf = Pf.reshape((nc[0],nc[1],nc[2]))

    Total_mass.append(total_mass_egd)
    Total_delta.append(total_delta_egd)
    DM_mass.append(egd_rho_dm * Msun_per_particle[i])
    Temperature.append(Tf)
    Pressure.append(Pf)

# print("total mass", Total_mass[:3])
# print("total delta", Total_delta[:3])
# print("DM mass", DM_mass[:3])
print("temperature", Temperature[:3])
print("pressure", Pressure[:3])

# do first level of integrating, into slabs of density and pressure 
lensplanes = []
pressureplane = []

for i in range(n_lens):

    width = lensplane_width/box_size[-1]*nc[-1]
    center = (i+0.5)*lensplane_width/box_size[-1]*nc[-1]
    layer_min = int(center-0.5*width) 
    
    # replace this by the density plane function for both
    density_plane = Total_delta[i][:,:, int(center-0.5*width):int(center+0.5*width)].sum(axis=2)
    pressure_plane = Pressure[i][:,:, int(center-0.5*width):int(center+0.5*width)].sum(axis=2)
    # pdb.set_trace()

    lensplanes.append({'r': r_center[i],
                      'a': a_center[::-1][i],
                      'plane': density_plane,
                      'dx':box_size[0]/nc[0],
                      'dz':lensplane_width})
    
    # replace with proper coordiate transform
    pressureplane.append(pressure_plane)

    # pressureplane.append({'r': r_center[i],
    #                   'a': a_center[::-1],
    #                   'plane': pressure_plane,
    #                   'dx':box_size[0]/nc[0],
    #                   'dz':lensplane_width})

# now do integrated quantities: kappa planes, pressure planes

kappa = convergence_Born(cosmo,
                      lensplanes,
                      coords=jnp.array(coords2).T.reshape(2,field_npix,field_npix),
                      z_source=z_source)


# really we should write a function similar to convergence_Born here
# y = tsz_Born(cosmo,
#              lensplanes,
#              coords=jnp.array(coords2).T.reshape(2,field_npix,field_npix),
#              z_source=z_source)

y = []
for i in range(n_lens):
    y.append(np.sum(np.array(pressureplane)[:(i+1), :,:], axis=0))

print("kappa", kappa[:3])
print("y", y[:3])

np.savez('kappa.npz', kappa=kappa, z=z_source)
np.savez('y.npz', y=y, z=1./a_center-1)


