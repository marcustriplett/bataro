import numpy as np
from sklearn.gaussian_process.kernels import RBF
from scipy.linalg import cholesky
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import reduce

from mean_functions import exponential_mean_fn_vmap
from jax import jit, vmap
from jax.config import config; config.update("jax_enable_x64", True)

class Simulation:
	def __init__(self, N=30, arena_dims=[250, 250], receptive_field_dim=80, core_grid_spacing=5., core_power_spacing=15., 
		min_centroid_distance=10, kernel=None, mean_fn_sigma=3e2, mean_fn_phi=1.25e-1, neuron_locations=None,
		power_fn_theta=3.5, marginal_var=1e-5, mean_fn_I_max=70.): #mean_fn_theta=5.,):

		# Build arena and place neurons
		self.N = N
		self.arena_dims = arena_dims
		xrange_arena, yrange_arena = [np.arange(0, arena_dims[0] + 1e-6, 1), \
			np.arange(0, arena_dims[1] + 1e-6, 1)]
		xs_arena, ys_arena = np.meshgrid(xrange_arena, yrange_arena)
		self.arena = np.c_[xs_arena.flatten(), ys_arena.flatten()]
		self.xrange_grid, self.yrange_grid = [np.arange(-receptive_field_dim//2, receptive_field_dim//2 + 1e-6, 1), 
			np.arange(-receptive_field_dim//2, receptive_field_dim//2 + 1e-6, 1)]
		xs_grid, ys_grid = np.meshgrid(self.xrange_grid, self.yrange_grid)
		self.grid = np.c_[xs_grid.flatten(), ys_grid.flatten()]
		if neuron_locations is None:
			self.neuron_locations = place_neurons(N, self.arena, min_centroid_distance=min_centroid_distance)
		else:
			self.neuron_locations = neuron_locations

		# Setup neuron receptive fields
		self.receptive_field_dim = receptive_field_dim
		self.extent = [-receptive_field_dim/2, receptive_field_dim/2, -receptive_field_dim/2, receptive_field_dim/2]
		assert kernel is not None; self.kernel = kernel
		self.core_grid = generate_grid(np.arange(-receptive_field_dim//2, receptive_field_dim//2 + 1e-6, 5), 
			np.arange(0, mean_fn_I_max + 1e-6, core_power_spacing))
		prior_cov = kernel(self.core_grid) + marginal_var * np.eye(self.core_grid.shape[0])
		prior_cov_inv = np.linalg.inv(prior_cov)
		chol = cholesky(prior_cov)

		self.mean_fn = lambda x: exponential_mean_fn_vmap(x, mean_fn_phi, mean_fn_sigma)
		self.power_fn_theta = power_fn_theta

		# Only define a `core` of the receptive fields using a small number of samples. Then use GP prediction to extrapolate the ORF 
		# 	when needed. Greatly improves efficiency.
		core_mean = self.mean_fn(self.core_grid)[:, None]
		receptive_field_cores = core_mean + (np.random.normal(0, 1, [N, self.core_grid.shape[0]]) @ chol).T
		self.prior_cov_inv_times_gp = prior_cov_inv @ (receptive_field_cores - core_mean)

	def get_gp(self, x, n):
		return self.mean_fn(x) + self.kernel(x, self.core_grid) @ self.prior_cov_inv_times_gp[:, n]

	def get_orf(self, x, n):
		return sigmoid(softplus(self.get_gp(x, n)))

	def power_fn(self, x):
		J = x.shape[0]
		drive = np.zeros(self.N)
		ones = np.ones(self.N)
		for j in range(J):
			xj = np.c_[x[j, :2] - self.neuron_locations, x[j, 2]*ones]
			lhs = self.kernel(self.core_grid, xj).T
			orf_response = self.mean_fn(xj) + eval_pred(lhs, self.prior_cov_inv_times_gp)
			drive += softplus(orf_response) # ensures non-negativity
		return sigmoid(drive - self.power_fn_theta)

	def next_trial(self, x):
		eval_powers = np.array(self.power_fn(x))
		resps = (np.random.rand(self.N) <= eval_powers).astype(float)
		resps[np.where(eval_powers < 0.05)[0]] = 0. # prevent abberant responses
		return resps

	def simulate(self, J=1, deterministic_stim_grid_range=[-20, 20.01], deterministic_stim_grid_step=10., 
		deterministic_stim_powers=[30, 50, 70]):
		''' Simulates an all-optical read-write experiment.

		'''

		# Setup deterministic stim grid
		detm_grid_min, detm_grid_max = deterministic_stim_grid_range
		detm_grid_xy = np.arange(detm_grid_min, detm_grid_max, deterministic_stim_grid_step)
		xs_detm, ys_detm, ps_detm = np.meshgrid(detm_grid_xy, detm_grid_xy, deterministic_stim_powers)
		deterministic_stim_grid = np.c_[xs_detm.flatten(), ys_detm.flatten(), ps_detm.flatten()]

		all_stimuli = np.vstack([np.c_[self.neuron_locations[n] - deterministic_stim_grid[:, :2], deterministic_stim_grid[:, -1]] 
			for n in range(self.N)])
		n_target_locs = all_stimuli.shape[0]

		arr = np.random.choice(n_target_locs, n_target_locs, replace=False)
		array_split = np.array([arr[i*J:(i+1)*J] for i in range(n_target_locs//J)])

		K = len(array_split)
		L = np.zeros((K, J, 2))
		responses = np.zeros((self.N, K))
		stim_matrix = np.zeros((self.N, J, K))

		print('Simulating experiment...')
		for k in tqdm(range(K)):
			x = all_stimuli[array_split[k]]
			L[k] = x[:, :2]
			for j in range(J):
				hit_neurons = np.intersect1d(
					*[np.where(np.abs(self.neuron_locations[:, i] - x[j, i]) <= self.receptive_field_dim//2)[0] for i in range(2)]
				)
				stim_matrix[hit_neurons, j, k] = x[j, 2]
			responses[:, k] = self.next_trial(x)

		print('Formatting training data...')

		stim_grid = [None for _ in range(self.N)]
		y_train = [None for _ in range(self.N)]
		x_train = [None for _ in range(self.N)]

		for n in tqdm(range(self.N)):
			stim_grid[n], y_train[n], x_train[n] = self.construct_training_data(n, responses, stim_matrix, L)

		return x_train, y_train, stim_grid, stim_matrix, responses, L

	def construct_training_data(self, n, responses, stim_matrix, L):
		''' Identifies which neurons were hit by multi-target laser on each trial, at what location and at what power.

			Returns:
				grid: the unique locations and powers at which the neuron was stimulated
				y_train: the responses to stimulation on each trial
				x_train: list of points on `grid` that were stimulated on each trial. Elements of x_train are of variable length
					as neurons are randomly hit by different numbers of lasers on different trials.
			
		'''
		_all_locs = []
		_all_stimuli = []
		J = stim_matrix.shape[1]
		for j in range(J):
			locs = np.where(stim_matrix[n, j])[0]
			_all_locs += [locs]
			_all_stimuli += [np.c_[L[locs, j] - self.neuron_locations[n], stim_matrix[n, j, locs]]]

		all_locs = np.concatenate(_all_locs)
		all_stimuli = np.concatenate(_all_stimuli)
		
		all_stimuli_tuples = [tuple(row) for row in all_stimuli]
		indices = np.unique(all_stimuli_tuples, axis=0, return_index=True)[1]
		grid = all_stimuli[sorted(indices)]
		
		x_train = []
		y_train = []
		stim_locs = np.where(np.sum(stim_matrix[n], axis=0))[0]
		for k in range(len(stim_locs)):
			stim_indices = get_grid_indices(np.c_[L[stim_locs[k]] - self.neuron_locations[n], stim_matrix[n, :, stim_locs[k]]], grid)
			x_train += [stim_indices[(1-np.isnan(stim_indices)).astype(bool)].astype(int)]
			y_train += [responses[n, stim_locs[k]]]
		y_train = np.array(y_train)
		
		return grid, y_train, x_train

	def test_calibration(self, target, power=70, K=100):
		responses = np.zeros((self.N, K))
		power_ar = np.array([power])
		for k in range(K):
			responses[:, k] = next_trial(power_ar, self.neuron_locations[target], self.receptive_fields, self.neuron_locations, self.grid,
				self.phi_0, self.phi_1)
		return responses

def _eval_pred(lhs_n, prior_cov_inv_times_gp_n):
	return lhs_n @ prior_cov_inv_times_gp_n
eval_pred = jit(vmap(_eval_pred, (0, 1)))

def generate_grid(xy_range, p_range):
	xs, ys, ps = np.meshgrid(xy_range, xy_range, p_range)
	return np.c_[xs.flatten(), ys.flatten(), ps.flatten()]

def softplus(x, beta=5.):
	return np.log(1 + np.exp(beta*x))/beta

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def edist(neurons, target):
	return np.sqrt(np.sum(np.square(neurons - target), 1))

def place_neurons(N, arena, n_attempts=100, min_centroid_distance=10):
	neuron_locations = np.zeros((N, 2))
	neuron_locations[0] = arena[np.random.choice(len(arena))]
	for n in range(1, N):
		for k in range(n_attempts):
			loc = arena[np.random.choice(len(arena))]
			if np.all(edist(neuron_locations[:n], loc) >= min_centroid_distance):
				neuron_locations[n, :] = loc
				break
		if k == n_attempts - 1:
			# could not place neuron, select random location
			neuron_locations[n, :] = loc
			
	return neuron_locations

def get_grid_idx(tar, grid, ndim=3):
	idx = reduce(np.intersect1d, [np.where(tar[i] == grid[:, i])[0] for i in range(ndim)])
	if len(idx) == 0:
		return np.nan
	else:
		return idx[0]
	
def get_grid_indices(tars, grid, ndim=3):
	raw_idx = np.array([get_grid_idx(tar, grid, ndim=ndim) for tar in tars])
	return raw_idx
