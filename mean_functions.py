import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.config import config; config.update("jax_enable_x64", True)

def exponential_mean_fn(x, phi, sigma):
    return phi * x[-1] * jnp.exp(-jnp.sum(jnp.square(x[:-1])/(2*sigma)))

exponential_mean_fn_vmap = jit(vmap(exponential_mean_fn, (0, None, None)))
exponential_mean_fn_deriv = grad(exponential_mean_fn)
exponential_mean_fn_deriv_vmap = vmap(exponential_mean_fn_deriv, (0, None, None))