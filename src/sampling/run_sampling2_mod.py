
import os,sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'
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

# Function to generate intial conditions
def linear_field(mesh_shape, box_size, pk):
  """Generate initial conditions
  mesh_shape : list of 3 numbers e.g. [64,64,2000]
  box_size   : list of 3 numbers in units of Mpc/h [100,100,4000]
  pk         : power spectrum to generate initial condition from. 
  """
  kvec   = jaxpm.kernels.fftk(mesh_shape)
  kk     = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
  kmesh  = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])

  # This is one of the variables (although it's a 3d grid) we are sampling over 
  field  = numpyro.sample('initial_conditions',dist.Normal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape)))

  field  = jnp.fft.rfftn(field) * pkmesh**0.5
  field  = jnp.fft.irfftn(field)
  return field

# Funtion to return a dictionary that contains all the density plane info
def get_density_planes(cosmology, density_plane_width     = 100., # In Mpc/h
                                  density_plane_npix      = 16  , # Number of pixels in xy
                                  density_plane_smoothing = 3.  , # In Mpc/h
                                  box_size = [400., 400., 4000.], # In Mpc/h
                                  nc       = [16, 16, 128],
                                  neural_spline_params=None):
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
    print("Setting initial redshift to %.2f"%a_init)

    # Planning out the scale factor stepping to extract desired lensplanes
    n_lens     = int(box_size[-1] // density_plane_width)
    print("Splitting box with depth %dMpc/h into shells with thickness %dMpc/h -> %d planes"%(box_size[-1], density_plane_width, n_lens) )

    chi        = jnp.linspace(0., box_size[-1], n_lens + 1)
    chi_center = 0.5 * (chi[1:] + chi[:-1])
    print("These planes correspond to comoving distances (in Mpc/h):")
    print(chi_center) 

    a_center = jc.background.a_of_chi(cosmology, chi_center)
    print("Converted into scale factor (unitless):")
    print(a_center)

    # Create a small function to generate the matter power spectrum
    kh    = jnp.logspace(-4, 1, 256)                    # h/Mpc
    pk    = jc.power.linear_matter_power(cosmology, kh) # (Mpc/h)^3
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), kh, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(nc, box_size, pk_fn)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),axis=-1).reshape([-1, 3])

    # Displace of the particles at the initial condition
    cosmology._workspace = {}  # FIX ME: this a temporary fix
    dx, p, f             = lpt(cosmology, initial_conditions, particles, a=a_init)

    # Some function to make the resolution of the simulation better
    # NEED TO UNDERSTAND THIS
    @jax.jit
    def neural_nbody_ode(a, state, args):
        """
        state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
        See this link for conversion rules: https://github.com/fastpm/fastpm#units
        """
        cosmo, params = args
        pos = state[0]
        vel = state[1]

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
    def density_plane_fn(t, y, args):
        '''
        Args:
          t   : scale_fcator
          y   :
          args:
        '''
        cosmo, _   = args
        positions  = y[0]
        nx, ny, nz = nc

        # Converts time t to comoving distance in voxel coordinates
        w      = density_plane_width / box_size[2] * nc[2]
        center = jc.background.radial_comoving_distance(cosmo, t) / box_size[2] * nc[2]

        xy = positions[..., :2] 
        d  = positions[..., 2]

        # Apply 2d periodic conditions
        xy = jnp.mod(xy, nx)

        # Rescaling positions to target grid
        xy = xy / nx * density_plane_npix

        # Selecting only particles that fall inside the volume of interest
        weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)), 1., 0.)

        # Painting density plane
        density_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, weight)

        # Apply density normalization
        density_plane = density_plane / ((nx / density_plane_npix) *
                                        (ny / density_plane_npix) * w)

        return density_plane

    # Evolve the simulation forward
    term   = diffrax.ODETerm(neural_nbody_ode)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=a_center[::-1], fn=density_plane_fn)
    # stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

    solution = diffrax.diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.05,
                                                 y0=jnp.stack([particles+dx, p], axis=0),
                                                 args=(cosmology, neural_spline_params),
                                                 saveat=saveat,
                                                 adjoint=diffrax.RecursiveCheckpointAdjoint(5),
                                                 max_steps=32)
                                                 #  stepsize_controller=stepsize_controller) 

    dx = box_size[0] / density_plane_npix
    dz = density_plane_width
    print("voxel size in xy direction: %.1f Mpc/h"%dx )
    print("voxel size in z  direction: %.1f Mpc/h"%dz )

    # Apply some amount of gaussian smoothing defining the effective resolution of the density planes
    print('Applying smoothing')
    density_plane = jax.vmap(lambda x: gaussian_smoothing(x,  density_plane_smoothing / dx ))(solution.ys)
    print('Done applying smoothing')

    print('Saving dict')
    return {'planes': density_plane[::-1],
            'a'     : solution.ts[::-1],
            'a2'    : a_center,
            'chi'   : chi_center,
            'dx'    : dx,
            'dz'    : dz
            }


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
    A = 3 / 2 * cosmo.Omega_m * (cosmo.h*100 / c)**2

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


def forward_model(data = None):
    """
    This function defines the top-level forward model for our observations
    """
    box_size   = [200., 200., 4000.] # In Mpc/h
    nc         = [32, 32, 256]       # Number of pixels
    field_npix = 256                 # Number of pixels in the lensing field
    sigma_e    = 0.26                # Standard deviation of galaxy ellipticities
    galaxy_density = 10.           # Galaxy density per arcmin^2, per redshift bin

    field_size = jnp.arctan2(boxsize[-1],boxsize[0])/np.pi*180                  # Size of the lensing field in degrees

    # Sampling cosmological parameters and defines cosmology
    # Note that the parameters are shifted so e.g. Omega_c=0 means Omega_c=0.25
    Omega_c = numpyro.sample('omega_c', dist.TruncatedNormal(0.,1, low=-1))*0.2 + 0.25
    sigma_8 = numpyro.sample('sigma_8', dist.Normal(0., 1.))*0.14 + 0.831
    Omega_b, h, n_s, w0 = 0.04, 0.7, 0.96, -1  # fixed parameters

    cosmo   = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma_8, Omega_b=Omega_b,
                           h=h, n_s=n_s, w0=w0, Omega_k=0., wa=0.)

    # Generate density planes through an nbody
    # Here the density_plane_npix doesn't have to match the npix of lensing map
    # but probably should be higher. Same with density_plane_width.
    density_planes = get_density_planes(cosmo, box_size=box_size, nc=nc,
                                               neural_spline_params = params,
                                               density_plane_npix = 512,
                                               density_plane_smoothing = 0.75,
                                               density_plane_width = 100.
                                        )

    # Defining the coordinate grid for lensing map
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                                jnp.linspace(0, field_size, field_npix, endpoint=False)) # range of Y coordinates

    coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))*0.017453292519943295 ) # deg->rad

    # Generate convergence maps by integrating over nz and source planes
    #convergence_maps = []
    #for i,nz in enumerate(nz_shear):
    #print("Making convergence map zbin %d"%i)
    kappa =  simps(lambda z: nz(z).reshape([-1,1,1]) * convergence_Born(cosmo, density_planes, coords, z), 0.01, 1.0, N=32)
    #import pdb; pdb.set_trace()
    #convergence_maps.append(kappa)

    # Apply noise to the maps (this defines the likelihood)
    #observed_maps = []
    #for i,k in enumerate(convergence_maps):
    #print("Adding noise to zbin %d"%i)
    numpyro.deterministic('latent_image', kappa)
    numpyro.sample('kappa_0', dist.Normal(kappa, sigma_e/jnp.sqrt(galaxy_density*(field_size*60/field_npix)**2)) ,obs=data) 

    #return observed_maps


#######################################################################################################################
# Reading the DC2 tomographic bins into redshift distribution objects
with h5py.File("shear_photoz_stack.hdf5") as f:
    source   = f["n_of_z"]["source"]
    z_shear  = source['z'][::]
    nz_shear = [jc.redshift.kde_nz(z_shear,source[f"bin_{i}"][:],bw=0.01, zmax=2.5) for i in range(1)]

# Loads some correction factors to improve the resolution of the simulation
params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )

# MISSING DOCSTRING -> what is this doing?????????
model = hk.without_apply_rng(hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,latent_size=32)(x, a)))

# Condition the model on a given set of parameters
# Condition here means fixing some of the cosmological parameters
# Here we are setting omega_c and sigma_8 to fiducial values to make a simulated data vector
fiducial_model = numpyro.handlers.condition(forward_model, {'omega_c': 0., 'sigma_8': 0.})

#import pdb;pdb.set_trace()
# Sample a mass map and save corresponding true parameters
model_trace    = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()

np.save('fidcucial_kappa.npy',model_trace['kappa_0']['value'])
#import pdb; pdb.set_trace()

# ok, cool, now let's sample this posterior
observed_model = numpyro.handlers.condition(forward_model, {'kappa_0': model_trace['kappa_0']['value']})
observed_model_reparam = observed_model # reparam(observed_model, config=config)

# Set up the NUT sampler
nuts_kernel = numpyro.infer.NUTS(
                                 model = observed_model_reparam,
                                 init_strategy  = partial(numpyro.infer.init_to_value, values={'omega_c': 0., 'sigma_8': 0. }),
                                 max_tree_depth = 3,
                                 step_size      = 2e-2
                                )

# Run the sampling 
mcmc = numpyro.infer.MCMC(
                          nuts_kernel,
                          num_warmup=1000,
                          num_samples=10000,
                          chain_method="parallel", num_chains=4,
                          # thinning=2,
                          progress_bar=True
                         )

print("---------------STARTING SAMPLING-------------------")
mcmc.run(jax.random.PRNGKey(0),model_trace['kappa_0']['value'])
print("-----------------DONE SAMPLING---------------------")

res = mcmc.get_samples()
#mcmc.print_summary()
#np.save('/pscratch/sd/y/yomori/omegac.npy',res['omega_c'])
#np.save('/pscratch/sd/y/yomori/sigma_8.npy',res['sigma_8'])

# Saving an intermediate checkpoint
with open('lensing_fwd_mdl_nbody_0.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Resuming from a checkpoint above
for i in range(4):
    print('round',i,'done')
    mcmc.post_warmup_state = mcmc.last_state
    mcmc.run(mcmc.post_warmup_state.rng_key)
    res = mcmc.get_samples()
    with open('lensing_fwd_mdl_nbody_%d.pickle'%(i+1), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)