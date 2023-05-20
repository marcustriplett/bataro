import time
import bataro as bto
import argparse
import simulation
import numpy as np
import mean_functions
from scipy import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.gaussian_process.kernels import RBF

sigmoid = lambda x: 1/(1 + np.exp(-x))

def get_error(target_neurons, probs):
	target = np.zeros(N)
	target[target_neurons] = 1
	return np.sum(np.square(target - probs))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ensemble_size')
	parser.add_argument('--savedir')

	args = parser.parse_args()
	ensemble_size = int(args.ensemble_size)
	savedir = args.savedir

	# Setup GP vars
	mean_fn_sigma = 3e2
	mean_fn_phi = 1.25e-1

	kernel_orf = 2e-1 * RBF(length_scale=[8, 8, 20])
	kernel = 1e0 * RBF(length_scale=[5, 5, 16])

	# Params for sims, mapping
	N = 50
	J_mapping = 10
	I_max = 70.
	test_power = 70.
	n_inits = 5
	n_repeats = 20
	n_sims = 10
	results_array_nuclear_vary_pop = np.zeros((n_sims, n_repeats, N))
	results_array_calib_vary_pop = np.zeros((n_sims, n_repeats, N))
	target_vectors = np.zeros((n_sims, n_repeats, N))

	t_start = time.time()

	print('Population size %i. Time %.2f min'%(N, (time.time()-t_start)/60))
	for k in range(n_sims):
		sim = simulation.Simulation(N, kernel=kernel_orf)
		x_train, y_train, stim_grid, _, _, _ = sim.simulate(J=J_mapping)

		models = bto.Models(N, kernel, sim.neuron_locations, mean_fn_sigma=mean_fn_sigma, mean_fn_phi=mean_fn_phi)
		models.fit(x_train, y_train, stim_grid, lr_theta=3e-2, newton_steps=20, max_backtrack_iters=10)

		pred_proba = lambda x: bto._predict_proba(x, *models._unpack())
		get_nuclear_proba = lambda tars: sim.power_fn(np.c_[sim.neuron_locations[tars], test_power * np.ones(len(tars))])
		get_true_proba = lambda x: sim.power_fn(x)

		for i in range(n_repeats):
			tars = np.random.choice(N, ensemble_size, replace=False)
			tar_vector = np.zeros(N)
			tar_vector[tars] = 1

			x, x_path, err = bto.optimise_stimulus(tars, models, learning_rate=2e1, iters=500, n_inits=n_inits, init_spread=3, I_max=I_max)
			seln = np.argmin(err[:, -1]) # selection
			sel_x = x[seln].astype(int)

			nuclear_probs = get_nuclear_proba(tars)
			calib_probs = get_true_proba(sel_x)

			results_array_nuclear_vary_pop[k, i] = nuclear_probs
			results_array_calib_vary_pop[k, i] = calib_probs
			target_vectors[k, i] = tar_vector

	if savedir[-1] != '/': savedir[-1] += '/'
	README = 'Array signature: (n_simulations, n_example_ensembles, N). In this data 10 simulations are used, and 20 random ensembles are used for each size.'
	np.savez(savedir + 'fig3_results_error_vs_ensemble_size_%i'%ensemble_size, results_nuclear=results_array_nuclear_vary_pop, results_calibrated=results_array_calib_vary_pop, target_vectors=target_vectors, README=README)


