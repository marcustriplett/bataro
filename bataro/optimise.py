from functools import partial

import jax.numpy as jnp
from jax.lax import scan, while_loop
from jax import grad, jit, jacfwd, jacrev, vmap
from jax.config import config; config.update("jax_enable_x64", True)

def sigmoid(x):
    '''Numerically stable sigmoid function.'''
    return jnp.where(x >= 0., 1. / (1. + jnp.exp(-x)), jnp.exp(x) / (1. + jnp.exp(x)))

@partial(jit, static_argnums=(8, 9))
def backtracking_newton(g, y, x, prior_mean, cov_inv, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=10, newton_steps=20,
	lr_theta=1e-1, init_theta=5.):
	'''Laplace approximation to GP posterior.

		x: list of grid points (i.e. points of g) that were stimulated on each trial
	'''

	def negloglik(g, theta):
		pre_activ = _sum_and_subtract_vmap(g, x, theta)
		activ = sigmoid(pre_activ)
		return -jnp.sum(y * jnp.log(activ) + (1 - y) * jnp.log(1 - activ + 1e-5)) - jnp.sum(jnp.log(g))

	grad_fn, hess_fn = _get_derivs(negloglik)
	grad_fn_theta = grad(negloglik, argnums=1)

	def negloglik_with_prior(g, theta):
		return negloglik(g, theta) + 1/2 * (g - prior_mean) @ cov_inv @ (g - prior_mean)

	def backtrack_cond(carry):
		it, _, lhs, rhs, _, _, _, _ = carry
		return jnp.logical_and(it < max_backtrack_iters, jnp.logical_or(jnp.isnan(lhs), lhs > rhs))

	def backtrack(carry):
		it, step, lhs, rhs, v, J, g, theta = carry
		it += 1
		step *= backtrack_beta
		lhs, rhs = get_ineq(g, step, v, J, backtrack_alpha, theta)
		return (it, step, lhs, rhs, v, J, g, theta)

	def get_ineq(g, step, v, J, backtrack_alpha, theta):
		return negloglik_with_prior(g + step * v, theta), negloglik_with_prior(g, theta) + backtrack_alpha * step * J @ v

	def get_stepv(g, theta):
		J = grad_fn(g, theta) + cov_inv @ (g - prior_mean)
		H = hess_fn(g, theta)
		H_inv = jnp.linalg.inv(H + cov_inv)
		v = -H_inv @ J
		return v, J, H_inv, -H

	def newton_step(g_carry, it):
		g, _, _, theta, theta_hist, nll_hist, gdiff_hist = g_carry
		v, J, cov, hess = get_stepv(g, theta)
		step = 1.
		lhs, rhs = get_ineq(g, step, v, J, backtrack_alpha, theta)
		init_carry = (0, step, lhs, rhs, v, J, g, theta)
		carry = while_loop(backtrack_cond, backtrack, init_carry)
		_, step, lhs, _, _, _, _, theta = carry
		gprev = jnp.copy(g)
		g += step * v
		theta -= lr_theta * grad_fn_theta(g, theta)

		## record histories
		theta_hist = theta_hist.at[it].set(theta)
		gdiff_hist = gdiff_hist.at[it].set(jnp.sqrt(jnp.sum(jnp.square(g - gprev))))
		nll_hist = nll_hist.at[it].set(negloglik_with_prior(g, theta))

		return (g, cov, hess, theta, theta_hist, nll_hist, gdiff_hist), lhs

	theta_hist = jnp.zeros(newton_steps)
	nll_hist = jnp.zeros(newton_steps)
	gdiff_hist = jnp.zeros(newton_steps)
	g_carry = (g, jnp.zeros((g.shape[0], g.shape[0])), jnp.zeros((g.shape[0], g.shape[0])), init_theta, theta_hist, nll_hist, gdiff_hist)
	return scan(newton_step, g_carry, jnp.arange(newton_steps))

def _get_derivs(fn):
	return jit(grad(fn, argnums=0)), jit(jacfwd(jacrev(fn)))

@jit
def _sum_and_subtract(g, point, theta):
	g_pad = jnp.r_[g, jnp.array([0.])] # pad g and x to allow vmap ## add 0 at the end to sop up padded terms
	return jnp.sum(g_pad[point]) - theta
_sum_and_subtract_vmap = jit(vmap(_sum_and_subtract, (None, 0, None)))
