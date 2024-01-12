import os,sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.90'
import h5py, pickle, jax, jaxpm, numpyro, diffrax
import haiku as hk
import numpy as np
import astropy.units as u
import jax_cosmo as jc
import jax.numpy as jnp
import numpyro.distributions as dist
from pathlib import Path
from functools import partial
from jax_cosmo.scipy.integrate import simps
from jax.scipy.ndimage import map_coordinates
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint, cic_read, cic_paint_2d
from jaxpm.kernels import gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.utils import gaussian_smoothing
from numpyro.handlers import seed, trace, condition, reparam
#from diffrax import diffeqsolve, ODETerm, Dopri5, LeapfrogMidpoint, PIDController, SaveAt
from jax.scipy.linalg import solve
from jax.scipy.linalg import cho_solve, cho_factor

from hpmtable3 import *
sys.path.append('../')
from pspec_tools import *

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Joint loglikelihood of a given map.
def compute_logprob(conditioned_model, res):
    log_probs = []
    for i in range(res['sigma_8'].shape[0]):
        params_i = {k: v[i] for k, v in res.items()}
        log_prob, _ = numpyro.infer.util.log_density(conditioned_model, (), {}, params=params_i)
        log_probs.append(log_prob)
    return jnp.array(log_probs)

# 2D RBF kernel
def rbf_kernel(x1, x2, length_scale=1.4, variance=0.5):
    # x1, x2 have shape [num_samples, 2]
    delta = x1[:, None, :] - x2[None, :, :]
    squared_distance = jnp.sum(delta ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * squared_distance / (length_scale ** 2))

# GP posterior
def gp_posterior(X_train, y_train, X_test, kernel_func, noise_variance=4e-5):
    K   = kernel_func(X_train, X_train) + noise_variance * jnp.eye(X_train.shape[0])
    Ks  = kernel_func(X_train, X_test)
    Kss = kernel_func(X_test, X_test)
    K_inv = jnp.linalg.inv(K)
    mu_s  = jnp.dot(Ks.T, jnp.dot(K_inv, y_train))
    return mu_s

# Function to create k-grid
def create_kgrid(nx, ny, nz, lx, ly, lz):
    """
    Create a 3D k grid for Fourier space calculations
    """
    xres = lx/nx #Mpc/h
    yres = ly/ny #Mpc/h
    zres = lz/nz #Mpc/h

    kx   = jnp.fft.fftshift(jnp.fft.fftfreq(nx, xres))# h/Mpc
    ky   = jnp.fft.fftshift(jnp.fft.fftfreq(ny, yres))# h/Mpc
    kz   = jnp.fft.fftshift(jnp.fft.fftfreq(nz, zres))# h/Mpc
    
    mg   = jnp.meshgrid(kx,ky,kz)

    km   = jnp.sqrt(mg[0]**2+mg[1]**2+mg[2]**2)

    return km

def HPM_GPmodel(cosmo,a,delta,psi,logMmin=8,logMmax=16,NM=40,rmin=0.1,rmax=4,Nr=40):
    """Takes delta (overdensity) and psi (fscalar) and extracts the inverse mapping via
       HPM table to predict the value of T & P.

    Parameters
    ----------
    delta     : float, arr
     overdensity values (rho_m)/<rho_m>
    psi     : float, arr
     fscalar valaues
    a       : float
     cosmological scale factor
    logMmin : float (optional)
    logMmax : float (optional)
    NM      : int   (optional)
    rmin    : float (optional)
    rmax    : float (optional)
    Nr      : int   (optional)

    Returns
    ----------
    T   : float, arr
     Temperature in units of K
    P   : float, arr
     Pressure in units of [Msun/s^2/Mpc]
    """
    mH_cgs  = 1.67223e-24
    
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

    # Locations to interpolate AT 
    index_points = jnp.array([jnp.log10(delta), jnp.log10(psi)]).T

    # First construct a table to map M/r -> rho/psi
    batched_r    = jax.vmap(table_halo,in_axes=[None, None, None, None, 0])
    batched_Mr   = jax.vmap(batched_r, in_axes=[None,None,None,0,None])
    del batched_r
    
    m_grid       = jnp.logspace(logMmin,logMmax,NM) # Msun/h
    r_grid       = jnp.linspace(rmin,rmax,Nr)       # unitless, to be multiplied by R200c
    res          = batched_Mr(cosmo,a,icm, m_grid.flatten(), r_grid.flatten())
    del batched_Mr, m_grid, r_grid
    
    tabM,tabR,_,_,tabrho,tabpsi,tabT,tabP,tabrhogas = res
    del res
    
    #Compute mean matter density in Msun/Mpc^3 to convert rho->delta
    #rhom0    = jc.background.Omega_m_a(cosmo,1.0)*jc.constants.rhocrit*cosmo.h**2 # [(M_sun)/ (Mpc)^{3}]
    #rhommean = rhom0*(a)**-3
    
    _delta  = jnp.log10(tabrho).flatten(); del tabrho
    _psi    = jnp.log10(tabpsi).flatten(); del tabpsi
    _T      = jnp.log10(tabT).flatten()  ; del tabT
    _P      = jnp.log10(tabP).flatten()  ; del tabP #
    _rhogas = jnp.log10(tabrhogas).flatten(); del tabrhogas #

    X_train = jnp.array(jnp.c_[_delta,_psi])
    T_train = jnp.array(_T)
    rhogas_train = jnp.array(_rhogas)


    #delta   = jnp.where(jnp.log10(delta)<7, 1e7, delta)
    X_test  = jnp.array(jnp.c_[jnp.log10(delta), jnp.log10(psi)])
    
    interp_T      = gp_posterior(X_train, T_train, jnp.c_[X_test[:,0],X_test[:,1]], rbf_kernel)
    interp_rhogas = gp_posterior(X_train, rhogas_train, jnp.c_[X_test[:,0],X_test[:,1]], rbf_kernel)
    
    return  jnp.asarray(10**interp_T), jnp.asarray(interp_rhogas)


# Function to generate intial conditions
def linear_field(mesh_shape, box_size, pk, field ):
  """Generate initial conditions
  mesh_shape : list of 3 numbers e.g. [64,64,2000]
  box_size   : list of 3 numbers in units of Mpc/h [100,100,4000]
  pk         : power spectrum to generate initial condition from. 
  """
  kvec   = jaxpm.kernels.fftk(mesh_shape)
  kk     = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
  kmesh  = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])
  field  = jnp.fft.rfftn(field) * pkmesh**0.5
  field  = jnp.fft.irfftn(field)
  return field

# Funtion to return a dictionary that contains all the density plane info
def get_density_planes(cosmology, density_plane_width     = 100., # In Mpc/h
                                  density_plane_npix      = 16  , # Number of pixels in xy
                                  density_plane_smoothing = 3.  , # In Mpc/h
                                  box_size = [400., 400., 4000.], # In Mpc/h
                                  nc       = [16, 16, 128],
                                  neural_spline_params=None,
                                  field = None,
                                  return_temperature = False
                                  ):
    """ Function that returns tomographic density planes for a given cosmology from a lightcone.
        Args:
            cosmology: jax-cosmo object
            density_plane_width: width of the output density slices
            density_plane_npix: size of the output density slices
            density_plane_smoothing: Gaussian scale of plane smoothing
            box_size: [sx,sy,sz] size in Mpc/h of the simulation volume
            nc: number of particles/voxels in the PM scheme
            neural_spline_params: optional parameters for neural correction of PM scheme
        Returns:
            list of [r, a, plane], slices through the lightcone along with their
                comoving distance (r) and scale factors (a). Each slice "plane" is a
                2d array of size density_plane_npix^2
    """
    # Initial scale factor for the simulation
    a_init = 0.01
    #print("Setting initial redshift to %.2f"%a_init)

    # Planning out the scale factor stepping to extract desired lensplanes
    n_lens     = int(box_size[-1] // density_plane_width)
    
    # Define comoving distances 
    chi        = jnp.linspace(0., box_size[-1], n_lens + 1)
    chi_center = 0.5 * (chi[1:] + chi[:-1])
    
    # Make sure scale factor is within the range of the neural spline
    a_center = jc.background.a_of_chi(cosmology, chi_center)
    a_center = jnp.clip(a_center, 0.25, 0.97)

    # Create a small function to generate the matter power spectrum
    kh    = jnp.logspace(-4, 1, 256)                    # h/Mpc
    pk    = jc.power.linear_matter_power(cosmology, kh) # (Mpc/h)^3
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), kh, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(nc, box_size, pk_fn, field)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),axis=-1).reshape([-1, 3])
    
    # Displace of the particles at the initial condition
    cosmology._workspace = {}  # FIX ME: this a temporary fix
    dx, p, _             = lpt(cosmology, initial_conditions, particles, a=a_init)

    Ge30     = 4.517e-18                     # CHECKED 6.67e-11*1.99e30/(3.086e22)**3*1e30 = 4.516e-18 [Mpc^3/s^2/Msun] 
    H2e30    = (cosmology.h*100)**2*1.050e-9 # CHECKED (1/3.086e19)**2*1e30 = 1.050e-9 [1/s^2]
    rhocrit0 = 3/8/jnp.pi*H2e30/Ge30         # CHECKED [Msun/Mpc^3]

    # Total particle mass
    # rhom(z) = Omegam0*rhocrit0/a^3  [Msun/Mpc^3(physical)]
    # totmass = rhom(z)*vol_comov*a^3 -> Omegam0*rhocrit0*vol_comov
    Mpart    = cosmology.Omega_m*rhocrit0*box_size[0]*box_size[1]*box_size[2]/particles.shape[0] #[Msun]  
  

    # Some function to make the resolution of the simulation better
    # NEED TO UNDERSTAND THIS
    @jax.jit
    def neural_nbody_ode(a, state, args):
        """
        state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
        See this link for conversion rules: https://github.com/fastpm/fastpm#units
        """
        cosmo, params, _ = args
        pos = state[0]
        vel = state[1]

        ##
        #HPM table 
        ##

        kvec    = jaxpm.kernels.fftk(nc)
        delta   = cic_paint(jnp.zeros(nc), pos)
        delta_k = jnp.fft.rfftn(delta)

        # Computes gravitational potential
        pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

        # Apply a correction filter
        if params is not None:
            kk    = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
            pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

        # Computes gravitational forces
        forces = jnp.stack([ cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos)for i in range(3)],axis=-1)
        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel], axis=0)


    # Function that will save the density planes as we are going 
    # THIS CODE IS VERY CONFUSING
    def output_plane_fn(a_plane, y, args):# change name
        '''
        Args:
          t   : scale_fcator
          y   :
          args:
        '''
        cosmo, _,  return_temperature = args
        positions  = y[0]
        nx, ny, nz = nc

        # Converts time t to comoving distance in voxel coordinates
        w      = density_plane_width / box_size[2] * nc[2]
        center = jc.background.radial_comoving_distance(cosmo, a_plane) / box_size[2] * nc[2]

        #Traced<ShapedArray(float32[4096,3])>with<DynamicJaxprTrace(level=2/0)>
        xy = positions[..., :2] 
        d  = positions[..., 2]

        # Apply 2d periodic conditions
        xy = jnp.mod(xy, nx)

        # Rescaling positions to target grid
        xy = xy / nx * density_plane_npix

        # Selecting only particles that fall inside the volume of interest
        weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)), 1., 0.)

        # Painting density plane. This is number of particles per voxel.
        density_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, weight)

        # Apply density normalization. Still in units of particles/voxel
        density_plane = density_plane / ( (nx / density_plane_npix) * (ny / density_plane_npix) * w )
        
        # Add Temperature 
 
        # Computing fscalar and delta
        # We want to first compute the full 3D density and fscalar field
        ####---->total_mass_egd
        G     = 4.390334603328476e-12 # (km)(Mpc^2)/Msun/s/Gyr
        vol_voxel  = (box_size[0]/nc[0] * box_size[1]/nc[1]*box_size[2]/nc[2] )/cosmo.h**3 # [Mpc^3]
        
        
        # convert from particles xyz positions to 3D grid 
        N_grid     = cic_paint(jnp.zeros((nc[0],nc[1],nc[2])), positions) # Number of particles per voxel
        M_grid     = N_grid  * Mpart                                      # Number of particles x Mass per particle -> Mass [Msun/h]

        F_rhom     = jnp.fft.fftshift(jnp.fft.fftn((M_grid/vol_voxel) ))  #[(Msun)/(Mpc)^3]
        kg         = create_kgrid(M_grid.shape[0], M_grid.shape[1], M_grid.shape[2], lx=box_size[0], ly=box_size[1], lz=box_size[2])

        kg = jnp.where(kg == 0, jnp.inf, kg)

        F_fscalar  = 2*jnp.pi**2*(G)*F_rhom/kg*1.989*3.24**2*1e-16*100*cosmo.h   # [cm/s^2] -- in units of HPM table
        R_fscalar  = jnp.fft.ifftn(jnp.fft.ifftshift(F_fscalar)) 
        R_fscalar -= jnp.min(R_fscalar.real)  
        
        # We assign a value of density and fscalar to each particle from their possitions
        #nx,ny,nz=50,50,500
        oversamprate=1
        px=jnp.linspace(0,nx,nx*oversamprate+1)
        py=jnp.linspace(0,ny,ny*oversamprate+1)
        pz=jnp.linspace(0,nz,nz*oversamprate+1)
        tmp = jnp.meshgrid(px,py,pz) # oversampled x,y,z positions on a grid
        positions_samp = jnp.c_[tmp[0].flatten(), tmp[1].flatten(), tmp[2].flatten()]
        #import pdb; pdb.set_trace()
        fscalar_pos = cic_read(R_fscalar.real  , positions_samp)
        delta_pos   = cic_read(M_grid/vol_voxel, positions_samp)



        xy_samp = positions_samp[..., :2] 
        d_samp  = positions_samp[..., 2]
        del positions_samp
        xy_samp = jnp.mod(xy_samp, nx)
        xy_samp = xy_samp / nx * density_plane_npix
        weight_samp = jnp.where((d_samp > (center - w / 2)) & (d_samp <= (center + w / 2)), 1, 0.)

        if return_temperature==True:
            #Compute both temperature and rhogas on fine grid
            T, rhogas = HPM_GPmodel(cosmo, a_plane, delta_pos, fscalar_pos)# T-> [K], rhogas -> [Msun/Mpc^3]

            #import pdb; pdb.set_trace()
            temperature_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy_samp, T*weight_samp)
            rhogas_plane      = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy_samp, rhogas/Mpart*weight_samp)
            
            #mass_plane        = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, delta_pos)

            return density_plane*cosmo.Omega_c/(cosmo.Omega_c+cosmo.Omega_b)+rhogas_plane, temperature_plane

        else:
            # Return density plane only
            return density_plane
        
        return density_plane  

    # Evolve the simulation forward
    term   = diffrax.ODETerm(neural_nbody_ode)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=a_center[::-1], fn=output_plane_fn) 
    # stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

    #print('a_init:' ,  a_init)
    #print('a_center:', a_center[::-1])

    solution = diffrax.diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.05,
                                                 y0=jnp.stack([particles+dx, p], axis=0),
                                                 args=(cosmology, neural_spline_params, return_temperature),
                                                 saveat=saveat,
                                                 adjoint=diffrax.RecursiveCheckpointAdjoint(5),
                                                 max_steps=32)
                                                 #  stepsize_controller=stepsize_controller) 

    dx = box_size[0] / density_plane_npix
    dz = density_plane_width
    #print("voxel size in xy direction: %.1f Mpc/h"%dx )
    #print("voxel size in z  direction: %.1f Mpc/h"%dz )

    # Apply some amount of gaussian smoothing defining the effective resolution of the density planes
    print('Applying smoothing')
    density_plane = jax.vmap(lambda x: gaussian_smoothing(x,  density_plane_smoothing / dx ))(solution.ys[0])
    print('Done applying smoothing')

    if return_temperature == True:
        print('Extracting temperature planes')
        print('Applying smoothing')
        temperature_plane = solution.ys[1]#DEBUG
        print('Done applying smoothing')

    print('Saving dict of density (and temperature) planes')
    pdict = {'planes': density_plane[::-1],
             'a'     : solution.ts[::-1],
             'a2'    : a_center,
             'chi'   : chi_center,
             'dx'    : dx,
             'dz'    : dz,
             'Mpart' : Mpart
            }
    
    if return_temperature == True:
        pdict.update({'Tplanes': temperature_plane[::-1]})

    return pdict



def convergence_Born(cosmo, density_planes, coords , z_source):
    """
    Compute the Born convergence
    Args:
        cosmo         : `Cosmology`, cosmology object.
        density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
        coords        : a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
        z_source      : 1-D `Tensor` of source redshifts with shape [Nz] .
    Returns:
        `Tensor` of shape [batch_size, N, Nz], of convergence values.
    """
    # Compute constant prefactor:
    c = 299792458
    A = 3 / 2 * cosmo.Omega_m * (cosmo.h*100 / (c/1000) )**2

    # Compute comoving distance of source galaxies
    chi_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0
    n_planes    = len(density_planes['planes'])

    dx = density_planes['dx']
    dz = density_planes['dz']

    for i in range(n_planes):

        chi = density_planes['chi'][i]
        a   = density_planes['a'][i]
        p   = density_planes['planes'][i]

        # Normalize density planes
        p = (p - p.mean()) * A * dz * chi / a 

        # Interpolate at the density plane coordinates
        im = map_coordinates(p, coords * chi / dx - 0.5, order=1, mode="wrap")

        convergence += im * jnp.clip(1. - (chi / chi_s), 0, 1000).reshape([-1, 1, 1])

    return convergence


def comptony_Born(cosmo, planes, coords , z_source, ):
    """
    Compute the Born convergence
    Args:
        cosmo         : `Cosmology`, cosmology object.
        density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
        coords        : a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
        z_source      : 1-D `Tensor` of source redshifts with shape [Nz] .
    Returns:
        `Tensor` of shape [batch_size, N, Nz], of convergence values.
    """
    # Compute constant prefactor:
    '''
    mp     = 1.6726219e-27      # [kg]
    msun   = 1.98847e30         # [kg]
    mpc2m  = 3.085677581e22     # [m]
    kb     = 1.38064852e-23     # [m^2 kg s^-2 K^-1]
    mue    = 0.588              # mean molecular weight


    c = 299792458
    A = 3 / 2 * cosmo.Omega_m * (cosmo.h*100 / c)**2
    '''

    # Compute comoving distance of source galaxies
    chi_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source)) # [Mpc/h]

    comptony = 0
    n_planes = len(planes['planes'])

    dx    = planes['dx']    # [Mpc/h]
    dz    = planes['dz']    # [Mpc/h]
    Mpart = planes['Mpart'] # [Msun]

    print("Starting Compton-y integration")
    for i in range(n_planes):

        chi  = planes['chi'][i]     # [Mpc/h]
        a    = planes['a'][i]       # [unitless]
        d    = planes['planes'][i]  # [npart/(Mpc/h)^3 -- comoving]
        tgas = planes['Tplanes'][i] # [K]
        Mgas = d*Mpart*(cosmo.Omega_b/cosmo.Omega_m) # CHECKED [Msun/(Mpc/h)^3 -- comoving]    

        rhogas = Mgas/a**3              # CHECKED [Msun/(Mpc/h)^3 -- physical]
        const1 = 1.4728010050999952e-11 # CHECKED (mp/msun)/mue/(mpc2m)**3*kb*m2mpc   precomputed factor to avoid float32 overflow
                                        # -> 1.988e30/(3.086e22)**3*1.38e-23/1.17/1.6726e-27*3.086e22  [kg/Mpc/s^2/K]
                                        # note the number mue=1.17, which is the quoted number in Mead 2020.

        #pe     = rhogas/(mp/msun)/mue/(mpc2m)**3*cosmo.h**2*kb*tgas # [1/m^3 (phys) m^2/s^2 kg], no more factors of h
        pe     = rhogas*cosmo.h**3*const1*tgas # [kg/Mpc/s^2] , no more factors of h
        A      = 8.125459939612701e-16 # CHECKED sigT/(me*c*c)  6.6524e-29/9.1093837e-31/299792458**2 # [s^2/kg]
        
        # Interpolate at the density plane coordinates
        im = map_coordinates(A*pe*a*dz/cosmo.h, coords * chi / dx - 0.5, order=1, mode="wrap")
        #print('----------------------------',i,im.shape)
        #import pdb; pdb.set_trace()
        comptony += im * jnp.ones_like(chi_s).reshape([-1, 1, 1]) #<-------Anything bellow and boave 0, 1000 gets set to 0 ,1000
        #im = map_coordinates(tgas, coords * chi / dx - 0.5, order=1, mode="wrap")
        #comptony += im * jnp.clip(1. - (chi / chi_s), 0, 1000).reshape([-1, 1, 1])       

    print("Done Compton-y integration")
    return comptony



def forward_model():
    """
    This function defines the top-level forward model for our observations
    """
    box_size   = [200., 200., 2000.] # In Mpc/h
    nc         = [50, 50, 500]       # Number of pixels
    #nc        = [8, 8, 64]         # Number of pixels
    field_npix = 50                  # Number of pixels in the lensing field
    sigma_e    = 0.0000              # Standard deviation of galaxy ellipticities
    galaxy_density = 10.             # Galaxy density per arcmin^2, per redshift bin
    compy       = True

    if compy==True:
        return_temperature = True
    else:
        return_temperature = False

    # Assert that the resolution in each direction are equal
    # (required by the neural spline code)
    assert box_size[0]/nc[0] ==  box_size[1]/nc[1] ==  box_size[2]/nc[2], "Resolution in each direction must be equal"

    # Field size in degrees
    field_size = 5.71
    #field_size = jnp.arctan2(box_size[0],box_size[-1])/np.pi*180                 
    print('field size is %.2fdeg x %.2fdeg'%(field_size,field_size) )
    
    # Sampling cosmological parameters and defines cosmology
    # Note that the parameters are shifted so e.g. Omega_c=0 means Omega_c=0.25
    # This is to ensure that the parameter space is sampled evenly around the fiducial values.
    Omega_c = numpyro.sample('omega_c', dist.TruncatedNormal(0.,1, low=-1))*0.02 + 0.25 ####################*0.2 + 0.25
    sigma_8 = numpyro.sample('sigma_8', dist.Normal(0., 1.))*0.014 + 0.831              ####################*0.14 + 0.831
    Omega_b, h, n_s, w0 = 0.04, 0.7, 0.96, -1  # fixed parameters

    cosmo   = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma_8, Omega_b=Omega_b, h=h, n_s=n_s, w0=w0, Omega_k=0., wa=0.)

    '''
    # Get total particle mass
    Ge30     = 4.516e-18                     # 6.67e-11*1.99e30/(3.086e22)**3*1e30 = 4.516e-18 [Mpc^3/s^2/Msun]
    H2e30    = (cosmology.h*100)**2*1.050e-9 # (1/3.086e19)**2*1e30 = 1.050e-9 [1/s^2]
    rhocrit0 = 3/8/jnp.pi*H2e30/Ge30         # [Msun/Mpc^3]
    # total particle mass
    # rhom(z) = Omegam0*rhocrit0/a^3  [Msun/Mpc^3(physical)]
    # totmass = rhom(z)*vol_comov*a^3 -> Omegam0*rhocrit0*vol_comov
    Mparttot  = cosmo.Omega_m*rhocrit0*box_size[0]*box_size[1]*box_size[2] #[Msun]
    '''

    # Generate density planes through an nbody
    # Here the density_plane_npix doesn't have to match the npix of lensing map
    # but probably should be higher. Same with density_plane_width.

    field  = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(nc) ,jnp.ones(nc) ) )

    planes = get_density_planes(
                                cosmo, nc = nc,
                                box_size = box_size, 
                                neural_spline_params = params,
                                density_plane_npix = 512,
                                density_plane_smoothing = 0.15,
                                density_plane_width = 100.,
                                field = field,
                                return_temperature = return_temperature
                                )
    
    # Defining the coordinate grid for lensing map
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                                jnp.linspace(0, field_size, field_npix, endpoint=False)) # range of Y coordinates

    coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))*0.017453292519943295 ) # deg->rad

    # Generate convergence maps by integrating over nz and source planes
    for i,nz in enumerate(nz_shear):
        print("Integrating along the LOS to create convergence map")
        kappa =  simps(lambda z: nz(z).reshape([-1,1,1]) * convergence_Born(cosmo, planes, coords, z), 0.01, 1.0, N=32)
        numpyro.deterministic('noiseless_convergence_%d'%i, kappa)

        #sig = sigma_e/jnp.sqrt(galaxy_density*(field_size*60/field_npix)**2)
        sig = jnp.ones((field_npix,field_npix))*0.02 # FIX
        numpyro.sample('kappa_0', dist.Normal(kappa, sig)) 

        delta_ell = 50.
        ell_max   = 3000.
        pix_size  = (field_size*60)/field_npix
        N         = field_npix
        window = jnp.ones_like(kappa)
        bell, cls_kappa = pspec(kappa,kappa,window,window,delta_ell,ell_max,pix_size,N)
        numpyro.deterministic('ells', bell)
        numpyro.deterministic('cls_noiseless_convergence_%d'%i, cls_kappa)

    if compy==True:
        print("Integrating along the LOS to create Compton-y map")
        compy =  simps(lambda z: comptony_Born(cosmo, planes, coords, z), 0.01, 1.0, N=32)
        numpyro.deterministic('noiseless_comptony', compy)

        sig = jnp.ones((field_npix,field_npix))*0.02 # FIX Probably needs some fiddling around 
        numpyro.sample('compy_0', dist.Normal(compy, sig)) 
        bell, cls_compy = pspec(compy,compy,window,window,delta_ell,ell_max,pix_size,N)
        numpyro.deterministic('cls_noiseless_comptony', cls_compy)
    #return observed_maps


##########################################################################################################

parser  = argparse.ArgumentParser()
#parser.add_argument('resume_state', default=False, dest='resume', action='store_true')
parser.add_argument('resume_state', default=None, type=int, help='')
args         = parser.parse_args()
resume_state = args.resume_state

# Reading the DC2 tomographic bins into redshift distribution objects
with h5py.File("shear_photoz_stack.hdf5") as f:
    source   = f["n_of_z"]["source"]
    z_shear  = source['z'][::]
    nz       = np.exp(-0.5*((z_shear-0.73)/0.02)**2)/np.sum( np.exp(-0.5*((z_shear-0.73)/0.02)**2)  )/(z_shear[1] - z_shear[0])
    #nz_shear = [jc.redshift.kde_nz(z_shear,source[f"bin_3"][:],bw=0.01, zmax=2.5) for i in range(1)]
    nz_shear = [jc.redshift.kde_nz(z_shear,nz,bw=0.01, zmax=2.5) for i in range(1)]

# Loads some correction factors to improve the resolution of the simulation
params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )

# MISSING DOCSTRING -> what is this doing?????????
model = hk.without_apply_rng(hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,latent_size=32)(x, a)))

# Condition the model on a given set of parameters
# Condition here means fixing some of the cosmological parameters
# Here we are setting omega_c and sigma_8 to fiducial values to make a simulated data vector
fiducial_model = numpyro.handlers.condition(forward_model, {
                                                            'omega_c': 0., 
                                                            'sigma_8': 0.
                                                           })

#import pdb;pdb.set_trace()
# Sample a mass map and save corresponding true parameters
model_trace    = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace( )

np.save('test_fidcucial_kappa_0.npy',model_trace['noiseless_convergence_0']['value'])
np.save('test_fidcucial_compy_0.npy',model_trace['noiseless_comptony']['value'])
'''
# Condition the model (i.e. generate "data" model)
observed_model = numpyro.handlers.condition(forward_model, {
                                                            'kappa_0': model_trace['kappa_0']['value'],
                                                            'compy_0': model_trace['compy_0']['value']
                                                           }
                                            )

# Setup the NUT sampler
nuts_kernel = numpyro.infer.NUTS(
                                 model = observed_model,
                                 init_strategy  = partial(numpyro.infer.init_to_value, values={'omega_c': 0., 'sigma_8': 0., 'initial_conditions': model_trace['initial_conditions']['value'] }),
                                 max_tree_depth = 3,
                                 step_size      = 1.27e-02
                                )

# Run the sampling 
mcmc = numpyro.infer.MCMC(
                          nuts_kernel,
                          num_warmup=0,
                          num_samples=50,
                          chain_method="parallel",
                          num_chains=4,
                          # thinning=2,
                          progress_bar=True
                         )

if resume_state<0:

    print("---------------STARTING SAMPLING-------------------")
    mcmc.run( jax.random.PRNGKey(0))
    print("-----------------DONE SAMPLING---------------------")

    res = mcmc.get_samples()
    
    # Saving an intermediate checkpoint
    with open('lensing_fwd_mdl_nbody_0.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Also save loglikelihood of a given map.
    log_probs = compute_logprob(observed_model, res)
    np.save('logprobs_lensing_fwd_mdl_nbody_0.npy',log_probs)

    del res,log_probs

    final_state = mcmc.last_state
    with open('mcmc_state_0.pkl', 'wb') as f:
        pickle.dump(final_state, f)


    # Continue on
    for i in range(1,500):
        print('round',i,'done')
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        with open('lensing_fwd_mdl_nbody_%d.pickle'%(i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_probs = compute_logprob(observed_model, res)
        np.save('logprobs_lensing_fwd_mdl_nbody_%d.npy'%(i),log_probs)

        del res,log_probs

        final_state = mcmc.last_state
        with open('mcmc_state_%d.pkl'%i, 'wb') as f:
            pickle.dump(final_state, f)
        

else:
    # Save
    with open('mcmc_state_%d.pkl'%resume_state, 'rb') as f:
        mcmc.post_warmup_state = pickle.load(f)

    for i in range(resume_state+1,resume_state+500):
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        with open('lensing_fwd_mdl_nbody_%d.pickle'%(i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_probs = compute_logprob(observed_model, res)
        np.save('logprobs_lensing_fwd_mdl_nbody_%d.npy'%(i),log_probs)

        del res,log_probs

        final_state = mcmc.last_state
        with open('mcmc_state_%d.pkl'%i, 'wb') as f:
            pickle.dump(final_state, f)

    



# To open the pickle file use:
#with open('lensing_fwd_mdl_nbody_0.pickle', 'rb') as handle:
#    res2 = pickle.load(handle)
'''