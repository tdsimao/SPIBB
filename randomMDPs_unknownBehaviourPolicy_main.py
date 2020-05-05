# authors: anonymized

import os
import sys
expname = sys.argv[1]
index = int(sys.argv[2])
import numpy as np
import pandas as pd
import garnets
import spibb
import spibb_utils
import modelTransitions
from RMDP import *
from SPI import *

from shutil import copyfile
from math import ceil, floor
spibb_utils.prt('Start of experiment')


def safe_save(filename, df):
	df.to_excel(filename + '.temp.xlsx')
	copyfile(filename + '.temp.xlsx', filename + '.xlsx')
	os.remove(filename + '.temp.xlsx')
	spibb_utils.prt(str(len(results)) + ' lines saved to ' + filename + '.xlsx')

N_wedges = [5, 7, 10, 15, 20]
delta = 1
epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000]

ratios = [0.1, 0.9]

seed = index
np.random.seed(seed)

gamma = 0.95
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4

mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
mask_0 = ~mask_0
rand_pi = np.ones((nb_states,nb_actions)) / nb_actions

filename = 'results/' + expname + '/results_' + str(index)

results = []
if not os.path.isdir('results'):
	os.mkdir('results')
if not os.path.isdir('results/' + expname):
	os.mkdir('results/' + expname)

while True:
	for ratio in ratios:
		garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition, self_transitions=0)

		softmax_target_perf_ratio = (ratio + 1) / 2
		baseline_target_perf_ratio = ratio
		pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf = \
								garnet.generate_baseline_policy(gamma,
																softmax_target_perf_ratio=softmax_target_perf_ratio,
																baseline_target_perf_ratio=baseline_target_perf_ratio)

		reward_current = garnet.compute_reward()
		current_proba = garnet.transition_function
		r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)

		for nb_trajectories in nb_trajectories_list:
			# Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
			trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b)
			spibb_utils.prt("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

			# Compute the maximal likelihood model for transitions and rewards.
			# NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
			# One should compute it fro mthe samples when it is stochastic.
			model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
			reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)

			policy_error = np.sum(abs(pi_b - model.policy), 1)
			# print("policy l1 error:", policy_error)
			print("policy divergence. mean: %05.4f; std: %05.4f" % (np.mean(policy_error), np.std(policy_error)))
			perf_pi_hat = spibb.policy_evaluation_exact(model.policy, r_reshaped, current_proba, gamma)[0][0]
			print("perf pi_hat: " + str(perf_pi_hat))

			# Estimates the values of the baseline policy with a monte-carlo estimation from the batch data:
			# q_pib_est = spibb_utils.compute_q_pib_est(gamma, nb_states, nb_actions, trajectories)

			# Computes the RL policy
			rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default')
			rl.fit()
			# Evaluates the RL policy performance
			perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
			print("perf RL: " + str(perfrl))


			# Computes the Reward-adjusted MDP RL policy:
			count_state_action = 0.00001 * np.ones((nb_states, nb_actions))
			kappa = 0.003
			for [action, state, next_state, reward] in batch_traj:
				count_state_action[state, action] += 1
			ramdp_reward_model = reward_model - kappa/np.sqrt(count_state_action)
			ramdp = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, ramdp_reward_model, 'default')
			ramdp.fit()
			# Evaluates the RL policy performance
			perf_RaMDP = spibb.policy_evaluation_exact(ramdp.pi, r_reshaped, current_proba, gamma)[0][0]
			print("perf RaMDP: " + str(perf_RaMDP))

			for N_wedge in N_wedges:
				# Computation of the binary mask for the bootstrapped state actions
				mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch_traj)
				# Computation of the model mask for the bootstrapped state actions
				masked_model = model.masked_model(mask)

				## Policy-based SPIBB ##

				# Computes the Pi_b_SPIBB policy:
				pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
				pib_SPIBB.fit()
				# Evaluates the Pi_b_SPIBB performance:
				perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Pi_b_SPIBB: " + str(perf_Pi_b_SPIBB))

				# Computes the Pi_b_SPIBB policy using estimated policy:
				pib_SPIBB_pi_hat = spibb.spibb(gamma, nb_states, nb_actions, model.policy, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
				pib_SPIBB_pi_hat.fit()
				# Evaluates the Pi_b_SPIBB performance:
				perf_Pi_b_SPIBB_pi_hat = \
				spibb.policy_evaluation_exact(pib_SPIBB_pi_hat.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Pi_b_SPIBB_pi_hat: " + str(perf_Pi_b_SPIBB_pi_hat))

				new_line = [
					seed, gamma, nb_states, nb_actions, 4, nb_trajectories, softmax_target_perf_ratio, baseline_target_perf_ratio, pi_b_perf, 0, pi_star_perf, perf_pi_hat, perfrl,
					perf_RaMDP, perf_Pi_b_SPIBB, perf_Pi_b_SPIBB_pi_hat, -1, -1, kappa, N_wedge, -1
				]
				results.append(new_line)



			for epsilon in epsilons:
				# Computation of the binary mask for the bootstrapped state actions
				mask = spibb.compute_mask(nb_states, nb_actions, epsilon, delta, batch_traj)[0]
				# Computation of the transition errors
				errors = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)

				# Computes the Soft-SPIBB-sort-Q policy
				soft_SPIBB_sort_Q = spibb.spibb(
				    gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_sort_Q',
				    errors=errors, epsilon=2 * epsilon
				)
				soft_SPIBB_sort_Q.fit()
				# Evaluates the Soft-SPIBB-sort-Q performance
				perf_soft_SPIBB_sort_Q = \
				spibb.policy_evaluation_exact(soft_SPIBB_sort_Q.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Approx-Soft-SPIBB:\t\t" + str(perf_soft_SPIBB_sort_Q))

				# Computes the Soft-SPIBB-sort-Q policyy using estimated policy:
				soft_SPIBB_sort_Q_pi_hat = spibb.spibb(
				    gamma, nb_states, nb_actions, model.policy, mask, model.transitions, reward_model, 'Soft_SPIBB_sort_Q',
				    errors=errors, epsilon=2 * epsilon
				)
				soft_SPIBB_sort_Q_pi_hat.fit()
				# Evaluates the Soft-SPIBB-sort-Q performance
				perf_soft_SPIBB_sort_Q_pi_hat = \
				spibb.policy_evaluation_exact(soft_SPIBB_sort_Q_pi_hat.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Approx-Soft-SPIBB_pi_hat:\t\t" + str(perf_soft_SPIBB_sort_Q_pi_hat))

				new_line = [
					seed, gamma, nb_states, nb_actions, 4, nb_trajectories, softmax_target_perf_ratio, baseline_target_perf_ratio, pi_b_perf, 0, pi_star_perf, perf_pi_hat, perfrl,
					perf_RaMDP, perf_Pi_b_SPIBB, perf_Pi_b_SPIBB_pi_hat, perf_soft_SPIBB_sort_Q, perf_soft_SPIBB_sort_Q_pi_hat, kappa, -1, epsilon
				]
				results.append(new_line)

		column_names = [
			'seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition', 'nb_trajectories', 'softmax_target_perf_ratio', 'baseline_target_perf_ratio',
			'baseline_perf', 'pi_rand_perf', 'pi_star_perf', 'perf_pi_hat', 'perfrl', 'perf_RaMDP', 'perf_Pi_b_SPIBB', 'perf_Pi_b_SPIBB_pi_hat', 'perf_soft_SPIBB_sort_Q', 'perf_soft_SPIBB_sort_Q_pi_hat', 'kappa', 'N_wedge',
			'epsilon'
		]
		df = pd.DataFrame(results, columns=column_names)

	# Save it to an excel file
	safe_save(filename, df)

