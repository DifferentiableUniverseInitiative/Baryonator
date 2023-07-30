import h5py
import jax
import jax.numpy as jnp
import jax_cosmo as jc
#import astropy.units as u
import haiku as hk
import pickle
import diffrax
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.integrate import simps
from bpcosmo.pm import get_density_planes
from jaxpm.lensing import convergence_Born
from jax.experimental.ode import odeint
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import fftk
from jaxpm.kernels import gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.lensing import density_plane
from jaxpm.nn import NeuralSplineFourierFilter
from diffrax import diffeqsolve, ODETerm, Dopri5, LeapfrogMidpoint, PIDController, SaveAt

def linear_field(mesh_shape, box_size, pk):
  """
    Generate initial conditions.
    """
  kvec = fftk(mesh_shape)
  kmesh = sum(
      (kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
      box_size[0] * box_size[1] * box_size[2])

  field = numpyro.sample(
      'initial_conditions',
      dist.Normal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape)))

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field


def get_density_planes(
    cosmology,
    density_plane_width=100.,  # In Mpc/h
    density_plane_npix=256,  # Number of pixels
    density_plane_smoothing=3.,  # In Mpc/h
    box_size=[400., 400., 4000.],  # In Mpc/h
    nc=[32, 32, 320],
    neural_spline_params=None):
  """Function that returns tomographic density planes
  for a given cosmology from a lightcone.

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

  # Planning out the scale factor stepping to extract desired lensplanes
  n_lens = int(box_size[-1] // density_plane_width)
  r = jnp.linspace(0., box_size[-1], n_lens + 1)
  r_center = 0.5 * (r[1:] + r[:-1])
  a_center = jc.background.a_of_chi(cosmology, r_center)

  # Create a small function to generate the matter power spectrum
  k = jnp.logspace(-4, 1, 256)
  pk = jc.power.linear_matter_power(cosmology, k)
  pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk
                                                ).reshape(x.shape)

  # Create initial conditions
  initial_conditions = linear_field(nc, box_size, pk_fn)

  # Create particles
  particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),
                        axis=-1).reshape([-1, 3])

  # Initial displacement
  cosmology._workspace = {}  # FIX ME: this a temporary fix
  dx, p, f = lpt(cosmology, initial_conditions, particles, a=a_init)

  @jax.jit
  def neural_nbody_ode(a, state, args):
    """
      state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
      See this link for conversion rules: https://github.com/fastpm/fastpm#units
      """
    cosmo, params = args
    pos = state[0]
    vel = state[1]

    kvec = fftk(nc)

    delta = cic_paint(jnp.zeros(nc), pos)

    delta_k = jnp.fft.rfftn(delta)

    model = hk.without_apply_rng(hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,latent_size=32)(x, a)))


    # Computes gravitational potential
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

    # Apply a correction filter
    if params is not None:
      kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
      pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

    # Computes gravitational forces
    forces = jnp.stack([
        cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos)
        for i in range(3)
    ],
                       axis=-1)

    forces = forces * 1.5 * cosmo.Omega_m

    # Computes the update of position (drift)
    dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

    # Computes the update of velocity (kick)
    dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

    return jnp.stack([dpos, dvel], axis=0)

  # Evolve the simulation forward
  term = ODETerm(neural_nbody_ode)
  solver = Dopri5()
  saveat = SaveAt(ts=jnp.concatenate([jnp.atleast_1d(a_init), a_center[::-1]]))
  stepsize_controller = PIDController(rtol=1e-4, atol=1e-4)

  solution = diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.01,
                       y0=jnp.stack([particles+dx, p], axis=0),
                       args=(cosmology, neural_spline_params),
                       saveat=saveat,
                       adjoint=diffrax.RecursiveCheckpointAdjoint()
                       )
                      #  adjoint=diffrax.BacksolveAdjoint(solver=Dopri5()),
                      #  stepsize_controller=stepsize_controller)

  # res = odeint(neural_nbody_ode, [particles + dx, p],
  #              jnp.concatenate([jnp.atleast_1d(a_init), a_center[::-1]]),
  #              cosmology,
  #              neural_spline_params,
  #              rtol=1e-5,
  #              atol=1e-5)
  res = solution.ys[:,0]

  # Extract the lensplanes
  density_planes = []
  for i in range(n_lens):
    dx = box_size[0] / density_plane_npix
    dz = density_plane_width
    plane = density_plane(res[::-1][i],
                          nc, (i + 0.5) * density_plane_width / box_size[-1] *
                          nc[-1],
                          width=density_plane_width / box_size[-1] * nc[-1],
                          plane_resolution=density_plane_npix,
                          smoothing_sigma=density_plane_smoothing / dx)
    density_planes.append({
        'r': r_center[i],
        'a': a_center[i],
        'plane': plane,
        'dx': dx,
        'dz': dz
    })

  return density_planes


def forward_model(box_size=[256., 256., 2048.], # In Mpc/h
                  nc = [64, 64, 256],         # Number of pixels
                  field_size = 10,            # Size of the lensing field in degrees
                  field_npix = 256,           # Number of pixels in the lensing field
                  sigma_e = 0.0,             # Standard deviation of galaxy ellipticities
                  galaxy_density = 10.,       # Galaxy density per arcmin^2, per redshift bin
                  nz_shear = None
                  ):
    """
    This function defines the top-level forward model for our observations
    """
    # Sampling cosmological parameters and defines cosmology
    Omega_c = numpyro.sample('omega_c', dist.Normal(0.3, 0.05))
    sigma_8 = numpyro.sample('sigma_8', dist.Normal(0.831, 0.14))
    Omega_b = 0.04
    h = 0.7
    n_s = 0.96
    w0 = -1
    cosmo = jc.Cosmology(Omega_c = Omega_c,
                         sigma8  = sigma_8,
                         Omega_b = Omega_b,
                         h=h, n_s=n_s,
                         w0=w0, Omega_k=0., wa=0.)

    # Generate lightcone density planes through an nbody
    params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )
    density_planes = get_density_planes(cosmo, box_size=box_size, nc=nc,
                                        neural_spline_params=params,
                                        density_plane_npix=32,
                                        density_plane_smoothing=1.,
                                        density_plane_width=100.
                                       )


    print('Loading N(z)')
    with h5py.File("shear_photoz_stack.hdf5") as f:
        group = f["n_of_z"]
        # Read the z grid
        source = group["source"]
        z_shear = source['z'][::]
        # Read the true n(z)
        nz_shear = [jc.redshift.kde_nz(z_shear,
                                       source[f"bin_{i}"][:],
                                       bw=0.01, zmax=2.5) for i in range(4)]



    # Create photoz systematics parameters, and create derived nz
    nzs_s_sys = [jc.redshift.systematic_shift(nzi,
                                            numpyro.sample('dz%d'%i, dist.Normal(0., 0.01)),
                                            zmax=2.5)
                for i, nzi in enumerate(nz_shear)]


    # Defining the coordinate grid for lensing map
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                                jnp.linspace(0, field_size, field_npix, endpoint=False)
                               ) # range of Y coordinates

    coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))/57.29571556389985 )

    # Generate convergence maps by integrating over nz and source planes
    convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                              convergence_Born(cosmo, density_planes, coords, z), 0.01, 1.0, N=32)
                      for nz in nzs_s_sys]
    # Apply noise to the maps (this defines the likelihood)
    observed_maps = [numpyro.sample('kappa_%d'%i,
                                  dist.Normal(k, sigma_e/jnp.sqrt(galaxy_density*(field_size*60/field_npix)**2)))
                   for i,k in enumerate(convergence_maps)]

    return observed_maps
