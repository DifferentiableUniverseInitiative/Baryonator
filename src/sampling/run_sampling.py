import h5py
import utils #
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import astropy.units as u
import haiku as hk
import pickle
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace, condition, reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from jax_cosmo.scipy.integrate import simps
from bpcosmo.pm import get_density_planes
from jaxpm.lensing import convergence_Born
from functools import partial
from time import time

def config(x):
    if type(x['fn']) is dist.TransformedDistribution:
        return TransformReparam()
    elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']) and ('kappa' not in x['name']):
        return LocScaleReparam(centered=0)
    else:
        return None


# Reading the DC2 tomographic bins into redshift distribution objects
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

# Loads some correction factors to improve the resolution of the simulation
print('Loading camels loss params')
params = pickle.load( open( "/project/chihway/yomori/repo/Baryonator/notebooks/camels_25_64_pkloss.params", "rb" ) )


__all__ = ['get_density_planes']

model = hk.without_apply_rng(hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,latent_size=32)(x, a)))

# condition the model on a given set of parameters
fiducial_model = numpyro.handlers.condition(utils.forward_model, {'omega_c': 0.3,
                                                                  'sigma_8': 0.8
                                                                 })

# sample a mass map and save corresponding true parameters
model_trace = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()



@jax.jit
def timed_func(om, s8):
    # condition the model on a given set of parameters
    fiducial_model = numpyro.handlers.condition(utils.forward_model, {'omega_c': om,
                                                                      'sigma_8': s8})
    # sample a mass map and save corresponding true parameters
    model_trace = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()
    return jnp.stack([model_trace['kappa_%d'%i]['value'] for i in range(4)],axis=0)


print('Setting up sampling: condition')
observed_model = condition(utils.forward_model, {'kappa_0': model_trace['kappa_0']['value'],
                                                 'kappa_1': model_trace['kappa_1']['value'],
                                                 'kappa_2': model_trace['kappa_2']['value'],
                                                 'kappa_3': model_trace['kappa_3']['value']})
print('Setting up sampling: reparam')
observed_model_reparam = reparam(observed_model, config=config)

print('Setting up sampling: NUTS')
nuts_kernel = numpyro.infer.NUTS(
                                model         = observed_model_reparam,
                                init_strategy = partial(numpyro.infer.init_to_value, values={'omega_c': 0.3,'sigma_8': 0.8}),
                                max_tree_depth= 5,
                                step_size     = 0.01)

print('Setting up sampling: MCMC')
mcmc = numpyro.infer.MCMC(
                          nuts_kernel,
                          num_warmup  = 0,
                          num_samples = 30,
                          progress_bar=True
                         )

print('Running MCMC')
start = time()
mcmc.run(jax.random.PRNGKey(42))
end   = time()
print(f'It took {end - start} seconds!')
