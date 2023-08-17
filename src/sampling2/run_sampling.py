import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'
import numpy as np
import haiku as hk
import jax,h5py,pickle
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
import astropy.units as u

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace, condition, reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

from functools import partial
from bpcosmo.pm import get_density_planes
from jax_cosmo.scipy.integrate import simps

from jax.experimental.ode import odeint
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.lensing import density_plane,convergence_Born
from jaxpm.painting import cic_paint, cic_read, cic_paint_2d
from jaxpm.kernels import fftk,gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.utils import gaussian_smoothing
from jax.scipy.ndimage import map_coordinates

import sim_utils as su
import hpm

# Reading the DC2 tomographic bins into redshift distribution objects
# This file can be obtained using:
# !wget --quiet https://github.com/LSSTDESC/star-challenge/raw/main/cosmodc2-srd-sample/generation/shear_photoz_stack.hdf5
with h5py.File("shear_photoz_stack.hdf5") as f:
    #Read n(z) for the 4 source redshift bins
    source   = f["n_of_z"]["source"]
    z_shear  = source['z'][::]
    nz_shear = [jc.redshift.kde_nz(z_shear,source[f"bin_{i}"][:],bw=0.01, zmax=2.5) for i in range(4)]

# Loads some correction factors to improve the resolution of the simulation.
# This file can be obtained using:
# !wget --quiet https://github.com/LSSTDESC/bayesian-pipelines-cosmology/raw/main/notebooks/forward_model/camels_25_64_pkloss.params
params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )

# TODO: ADD COMMENT
model  = hk.without_apply_rng(hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,latent_size=32)(x, a)))

# Condition the model on Omega_c and sigma_8.
# Note here we are using the reparametrized values of Omega_c and sigma_8. 
fiducial_model = numpyro.handlers.condition(su.forward_model, {'omega_c': 0.,
                                                               'sigma_8': 0.
                                                              }
                                           )

# Trace the execution of a probabilistic function and return its execution trace
# Physically generating a mass map and save corresponding true parameters
model_trace = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()

def config(x):
    if type(x['fn']) is dist.TransformedDistribution:
        return TransformReparam()
    elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']) and ('kappa' not in x['name']):
        return LocScaleReparam(centered=0)
    else:
        return None

# Conditioning based on the test data 
obs_model = condition(
                      su.forward_model, {'kappa_0': model_trace['kappa_0']['value'],
                                         'kappa_1': model_trace['kappa_1']['value'],
                                         'kappa_2': model_trace['kappa_2']['value'],
                                         'kappa_3': model_trace['kappa_3']['value']
                                        }
                     )
obs_model_reparam = obs_model 

# Setting up NUTS sampler
# model         : probabilistic model to be sampled from
# init_strategy : function that takes a model and returns an initial state for the sampler
# max_tree_depth: maximum depth of the tree implicitly built during each iteration
# step_size     : step size for the dual averaging step size adaptation
nuts_kernel = numpyro.infer.NUTS(
                                 model          = obs_model_reparam,
                                 init_strategy  = partial(numpyro.infer.init_to_value, values={'omega_c': 0.,
                                                                                               'sigma_8': 0.,
							                                                                   'initial_conditions': model_trace['initial_conditions']['value']
                                                                                              }
                                                        ),
                                 max_tree_depth = 3,
                                 step_size      = 2e-2
                                 )

# Setup MCMC sampler
# kernel      : kernel to use for sampling
# num_warmup  : number of warmup steps
# num_samples : number of samples to generate
# num_chains  : number of MCMC chains to run in parallel
# chain_method: method to use for splitting work between chains
# progress_bar: whether to enable progress bar updates
mcmc = numpyro.infer.MCMC(
                          nuts_kernel,
                          num_warmup=0,
                          num_samples=10,
                          # chain_method="parallel", num_chains=8,
                          # thinning=2,
                          progress_bar=True
                         )

# run MCMC
mcmc.run(jax.random.PRNGKey(0))

# extract MCMC results
res = mcmc.get_samples()
with open('lensing_fwd_mdl_nbody_0.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


for i in range(4):
    print('round',i,'done')
    mcmc.post_warmup_state = mcmc.last_state
    mcmc.run(mcmc.post_warmup_state.rng_key)
    res = mcmc.get_samples()
    with open('lensing_fwd_mdl_nbody_%d.pickle'%(i+1), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)