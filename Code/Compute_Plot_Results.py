from Experiments_Utils import *
from ABC_Algorithm import NSABCAlgorithm
from SA_Algorithm import SimulatedAnnealingAlgorithm
from Genetic_Algorithm import GeneticAlgorithm
import pickle


# ------ Run through all permutations for each experiment and calculate the results of the algorithms against each other

path = 'C:/Users/User/PycharmProjects/Computational_Intelligence/Results'
net_size = '/very_small_network'
sa_archives = []
nsabc_archives = []
ga_archives = []
permutations = 10
for p in range(permutations):
    seed = 42 + 3 * p
    nsabc_path = path + '/NSABC' + net_size + '/Permutation_Results/NSABC_Algorithm_Seed' + str(seed)
    sa_path = path + '/Simulated_Annealing' + net_size + '/Permutation_Results/SA_Algorithm_Seed' + str(seed)
    ga_path = path + '/Genetic_Algorithm' + net_size + '/Permutation_Results/GA_Algorithm_Seed' + str(seed)

    with open(sa_path, 'rb') as SA_exp:
        SA = pickle.load(SA_exp)
    with open(nsabc_path, 'rb') as NSABC_exp:
        NSABC = pickle.load(NSABC_exp)
    with open(ga_path, 'rb') as GA_exp:
        GA = pickle.load(GA_exp)

    nsabc_archives.append(NSABC)
    sa_archives.append(SA)
    ga_archives.append(GA)

results = compute_results(sa_archives, 'Simulated Annealing', nsabc_archives, 'Artificial Bee Colony', ga_archives, 'Genetiv Algorithm')

results.to_csv(path + '/Results_' + net_size[1:] + '_SA_NSABC_GA.csv', index=False)


# ------- Compute best results out of each combination of weights

weights_combination = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7],
                       [0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.1, 0.3], [0.3, 0.1, 0.3, 0.3], [0.1, 0.3, 0.3, 0.3],
                       [0.5, 0.3, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.5, 0.1, 0.1, 0.3], [0.3, 0.5, 0.1, 0.1],
                       [0.1, 0.5, 0.3, 0.1], [0.1, 0.5, 0.1, 0.3], [0.3, 0.1, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1],
                       [0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.3, 0.5], [0.1, 0.3, 0.1, 0.5], [0.3, 0.1, 0.1, 0.5],
                       [0.25, 0.25, 0.25, 0.25]]

path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results'
algo_name = '/Genetic_Algorithm'
net_size = '/big_network'

results = pd.read_csv(path + algo_name + net_size + '/All_runs_results.csv')
best_results = best_scores_extraction(results, weights_combination)

best_results.to_csv(path + algo_name + net_size + '/All_best_results.csv', index=False)


# -------- Find the results of each limit (ABC), decrease rate (SA) and repeat (GA)

sa_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\Simulated_Annealing'
very_small_results_sa = pd.read_csv(sa_path + '/very_small_network/All_runs_results.csv')
small_results_sa = pd.read_csv(sa_path + '/small_network/All_runs_results.csv')
medium_results_sa = pd.read_csv(sa_path + '/medium_network/All_runs_results.csv')
sa_results = [very_small_results_sa, small_results_sa, medium_results_sa]

nsabc_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\NSABC'
very_small_results_nsabc = pd.read_csv(nsabc_path + '/very_small_network/All_runs_results.csv')
small_results_nsabc = pd.read_csv(nsabc_path + '/small_network/All_runs_results.csv')
medium_results_nsabc = pd.read_csv(nsabc_path + '/medium_network/All_runs_results.csv')
nsabc_results = [very_small_results_nsabc, small_results_nsabc, medium_results_nsabc]

ga_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\Genetic_Algorithm'
very_small_results_ga = pd.read_csv(ga_path + '/very_small_network/All_runs_results.csv')
small_results_ga = pd.read_csv(ga_path + '/small_network/All_runs_results.csv')
medium_results_ga = pd.read_csv(ga_path + '/medium_network/All_runs_results.csv')
big_results_ga = pd.read_csv(ga_path + '/big_network/All_runs_results.csv')
ga_results = [very_small_results_ga, small_results_ga, medium_results_ga, big_results_ga]

parameter_results_sa = calculate_parameters_results(sa_results, 'Decrease_rate', [0.75, 0.85, 0.9])
parameter_results_nsabc = calculate_parameters_results(nsabc_results, 'Limit', [10, 50, 150])
parameter_results_ga = calculate_parameters_results(ga_results, 'Iterations', [100, 300, 500, 1000])

parameter_results_sa.to_csv(sa_path + '/Parameter_Results.csv', index=False)
parameter_results_nsabc.to_csv(nsabc_path + '/Parameter_Results.csv', index=False)
parameter_results_ga.to_csv(ga_path + '/Parameter_Results.csv', index=False)



