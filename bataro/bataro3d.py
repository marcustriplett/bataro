import numpy as np
from functools import partial
from scipy.linalg import cholesky
from tqdm import tqdm

import jax.numpy as jnp
from jax.nn import sigmoid
from jax.lax import scan, while_loop
from jax import grad, jit, jacfwd, jacrev, vmap
from jax.config import config; config.update("jax_enable_x64", True)

from mean_functions import exponential_mean_fn_vmap, exponential_mean_fn_deriv_vmap
from optimise import backtracking_newton

class Model:
	'''Some of the matrix operations for the Newton steps can be sped up.
	'''
	# def __init__(self, kernel, mean_fn_theta=5., mean_fn_I_max=70., mean_fn_sigma=1e2):
	def __init__(self, kernel, mean_fn_phi=1.25e-1, mean_fn_sigma=3e2):
		'''Arg kernel should be unscaled.
		'''
		self.kernel = kernel
		self.amplitude = kernel.get_params()['k1__constant_value']
		self.lengthscale = kernel.get_params()['k2__length_scale'] # assumes equal lengthscales

		self.mean_fn_phi = mean_fn_phi
		self.mean_fn_sigma = mean_fn_sigma

	def fit(self, x, y, grid, marginal_var=1e-5, newton_steps=20, lr_theta=1e-2, init_theta=5., max_backtrack_iters=10):
		'''Run Laplace EM for the GP Bernoulli model.

			Args: 
				grid: unique stimulus locations
				y: binary responses
				x: list of indices of grid stimulated on each trial

		'''

		self.D = grid.shape[0]
		self.grid = grid

		self.marginal_var = marginal_var
		self.prior_cov = self.kernel(grid) + marginal_var * np.eye(self.D)
		self.prior_cov_inv = np.linalg.inv(self.prior_cov)

		self.mean_fn = lambda x: exponential_mean_fn_vmap(x, self.mean_fn_phi, self.mean_fn_sigma)
		self.prior_mean = self.mean_fn(grid)

		g = _init_gp_device_array(self.D)

		# Pad x with dummary value to ensure inputs are of equal lengths (i.e., each stim hits J locations, but J-1 could be dummies)
		# This speeds up JIT compilation
		maxlen = np.max([len(_x) for _x in x])
		x_pad = jnp.array([np.r_[_x, len(g) * np.ones(maxlen - len(_x))] for _x in x]).astype(int)

		(g, posterior_cov, hess, theta, theta_hist, nll_hist, gdiff_hist), _ = backtracking_newton(g, y, x_pad, self.prior_mean, 
			self.prior_cov_inv, newton_steps=newton_steps, lr_theta=lr_theta, init_theta=init_theta, max_backtrack_iters=max_backtrack_iters)

		self.hess = hess
		self.posterior_cov = posterior_cov
		self.posterior_mean = g
		self.theta = theta

		diag = jnp.diag(self.hess)
		post_cov = self.hess.at[jnp.diag_indices_from(self.hess)].set(1/diag)
		self.inv_cov_diff = jnp.linalg.inv(self.prior_cov - post_cov)
		self.pred_mean_right_factor = self.prior_cov_inv @ (self.posterior_mean - self.prior_mean)

		self.theta_hist = theta_hist
		self.nll_hist = nll_hist
		self.gdiff_hist = gdiff_hist

	def predict(self, test_points, return_var=False):
		'''Evaluate posterior predictive mean and variance at test points.
		'''

		K_test = self.kernel(self.grid, test_points) # fast, ~4 ms
		pred_mean = self.mean_fn(test_points) + K_test.T @ self.pred_mean_right_factor

		if return_var:
			K_test_test = self.kernel(test_points, test_points) # this is slow for large test data, ~3 s
			pred_var = K_test_test - K_test.T @ self.inv_cov_diff @ K_test
			return pred_mean, pred_var
		else:
			return pred_mean

class Models:
	''' Collect models into single object for calibration.
	'''

	# def __init__(self, N, kernel, neuron_locations, mean_fn_theta=5., mean_fn_I_max=70., mean_fn_sigma=1e2):
	def __init__(self, N, kernel, neuron_locations, mean_fn_phi=1.25e-1, mean_fn_sigma=3e2, dims=4):
		self.N = N
		self.mean_fn_phi = mean_fn_phi
		self.mean_fn_sigma = mean_fn_sigma
		self.models = [Model(kernel=kernel, mean_fn_phi=mean_fn_phi, mean_fn_sigma=mean_fn_sigma) 
			for _ in range(N)]
		self.neuron_locations = jnp.array(neuron_locations).squeeze()
		self.dims = dims
		self._staged = False

	def fit(self, xs, ys, grids, **kwargs):
		for n in tqdm(range(self.N)):
			self.models[n].fit(xs[n], ys[n], grids[n], **kwargs)
		self._stage()

	def _stage(self):
		_grids = [mod.grid for mod in self.models]; max_grid_len = np.max([len(g) for g in _grids])
		self.grids = jnp.array([np.r_[g, np.zeros((max_grid_len - len(g), self.dims))] for g in _grids])
		self.amplitudes = jnp.array([mod.amplitude for mod in self.models])
		self.lengthscales = jnp.array([mod.lengthscale for mod in self.models])
		self.predictive_factors = jnp.array([
			np.r_[mod.pred_mean_right_factor, np.zeros(max_grid_len - len(mod.pred_mean_right_factor))] \
			for mod in self.models
		])
		self.thetas = jnp.array([mod.theta for mod in self.models])

	def _unpack(self):
		return self.grids, self.amplitudes, self.lengthscales, self.predictive_factors, self.thetas, self.neuron_locations,\
		self.mean_fn_phi, self.mean_fn_sigma

	# def _unpack(self):
	# 	return self.grids, self.amplitudes, self.lengthscales, self.predictive_factors, self.thetas, self.neuron_locations,\
	# 	self.mean_fn_theta, self.mean_fn_I_max, self.mean_fn_sigma

# Primary functions

def optimise_stimulus(target_neurons, models, learning_rate=3e1, iters=100, n_inits=5, init_spread=5, I_max=70.):
	N = models.N
	dims = models.dims
	target = np.zeros(N)
	target[target_neurons] = 1
	target = jnp.array(target)

	x = jnp.c_[
		models.neuron_locations[tuple(target_neurons), :] + np.random.normal(0, init_spread, [n_inits, len(target_neurons), dims-1]), 
		I_max * np.random.uniform(0.75, 0.95, [n_inits, len(target_neurons), 1]) # randomly init powers to 75-95% of max power
	]

	init_probs = jnp.stack([_predict_proba(x[i], *models._unpack()) for i in range(n_inits)])

	errs = jnp.zeros((n_inits, iters+1))
	errs = errs.at[:, 0].set(jnp.sum(jnp.square(target - init_probs), axis=1))
	x_path = jnp.zeros((n_inits, iters+1, len(target_neurons), dims))
	x_path = x_path.at[:, 0].set(x)
	for i in tqdm(range(iters)):
		x, x_path, errs = calibration_step(x, x_path, errs, learning_rate, target, i, I_max, *models._unpack())
	return x, x_path, errs

@jit
def _calibration_step(x, x_path, errs, learning_rate, target, i, I_max, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, 
	mean_fn_phi, mean_fn_sigma):

	pred_proba = lambda stim: _predict_proba(stim, grids, amplitudes, lengthscales, predictive_factors, thetas, 
		neuron_locations, mean_fn_phi, mean_fn_sigma)
	pred_grad = lambda stim: objective_gradient(target, stim, grids, amplitudes, lengthscales, predictive_factors,
		thetas, neuron_locations, mean_fn_phi, mean_fn_sigma)

	x += learning_rate * pred_grad(x)
	x = x.at[:, -1].set(jnp.where(x[:, -1] > I_max, I_max, x[:, -1])) # project
	errs = errs.at[i+1].set(jnp.sum(jnp.square(target - pred_proba(x))))
	x_path = x_path.at[i+1].set(x)
	return x, x_path, errs
calibration_step = jit(vmap(_calibration_step, tuple([0]*3 + [None]*12)))

@jit
def objective_gradient(target, x, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, 
	mean_fn_phi, mean_fn_sigma):
	pred_resp = _predict_proba(x, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, mean_fn_phi, mean_fn_sigma)

	pred_grad = _predict_gradient_population(x, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, mean_fn_phi, mean_fn_sigma)
	return -2 * jnp.sum(((target - pred_resp) * pred_resp * (1 - pred_resp))[:, None, None] * pred_grad, axis=0)

@jit
def objective(target, x, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, mean_fn_phi, mean_fn_sigma):
	pred_resp = _predict_proba(x, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, mean_fn_phi, mean_fn_sigma)
	return jnp.sum((target - pred_resp)**2)
objective_vmap = jit(vmap(objective, tuple([None] + [0] + [None] * 8)))

# Ancillary functions

# Fast kernel evaluation using iterated vmap
def _rbf_kernel(x1, x2, amplitude, lengthscale):
	Lambda = jnp.diag(1/jnp.array(lengthscale)**2)
	return amplitude * jnp.exp(-1/2 * (x1 - x2).T @ Lambda @ (x1 - x2))
_rbf_kernel_vmap = jit(vmap(vmap(_rbf_kernel, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None)))


@jit
def _rbf_gradient_ensemble(x_test, grid, amplitude, lengthscale):
	## OPPORTUNITY TO REDUCE COMPILATION TIME HERE
	N, D = grid.shape
	J = x_test.shape[0]
	Lambda = jnp.diag(1/jnp.array(lengthscale)**2)
	exp_factor = jnp.exp(-1/2 * _compute_exp_factor_arg_vmap(x_test, grid, Lambda))
	grad = jnp.zeros((J, D, N))
	for j in range(J):
		# stack over J targets
		arg = amplitude * jnp.r_[[(x_test[j, d] - grid[:, d])/lengthscale[d]**2 * exp_factor[:, j] for d in range(D)]]
		grad = grad.at[j].set(arg)
	return grad

def _compute_exp_factor_arg(x, g, Lambda):
	return (x - g) @ Lambda @ (x - g)
_compute_exp_factor_arg_vmap = vmap(vmap(_compute_exp_factor_arg, (0, None, None)), (None, 0, None))

@jit
def _predict_gradient(test_points, grid, amplitude, lengthscale, predictive_factor, theta, neuron_location, mean_fn_phi, mean_fn_sigma):
	''' Predict gradient of ORF GPs at array of test points
	'''
	_test_points_centered = jnp.c_[test_points[:, :-1] - neuron_location, test_points[:, -1]]
	K_test_obs = _rbf_gradient_ensemble(_test_points_centered, grid, amplitude, lengthscale)
	return -exponential_mean_fn_deriv_vmap(_test_points_centered, mean_fn_phi, mean_fn_sigma) + K_test_obs @ predictive_factor
_predict_gradient_population = jit(vmap(_predict_gradient, (None, 0, 0, 0, 0, 0, 0, None, None))) # vmaps over population of neurons
_predict_gradient_vmap = jit(vmap(_predict_gradient, (0, None, None, None, None, None, None, None, None))) # vmaps over test points

@jit
def _predict_proba(test_points, grids, amplitudes, lengthscales, predictive_factors, thetas, neuron_locations, mean_fn_phi, mean_fn_sigma):
	pred_gp = _predict_gp_population(test_points, grids, amplitudes, lengthscales, predictive_factors, neuron_locations, mean_fn_phi, mean_fn_sigma)
	return sigmoid(jnp.sum(pred_gp, 1) - thetas)

@jit
def _predict_gp(test_points, grid, amplitude, lengthscale, predictive_factor, neuron_location, mean_fn_phi, mean_fn_sigma):
	_test_points_centered = jnp.c_[test_points[:, :-1] - neuron_location, test_points[:, -1]]
	K_test = _rbf_kernel_vmap(_test_points_centered, grid, amplitude, lengthscale)
	return exponential_mean_fn_vmap(_test_points_centered, mean_fn_phi, mean_fn_sigma) + K_test.T @ predictive_factor
_predict_gp_population = jit(vmap(_predict_gp, (None, 0, 0, 0, 0, 0, None, None)))

def _init_gp_device_array(D):
	''' Randomly initialise Gaussian process
	'''
	g = np.random.normal(0, 1, [D]) # standard-normal
	g += np.abs(np.min(g)) + 1 # ensures feasible-start
	g = jnp.array(g)
	return g