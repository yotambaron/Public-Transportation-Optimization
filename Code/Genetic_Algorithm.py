from Traffic_Network import *
from Experiments_Utils import *
import random as rnd
from itertools import zip_longest
import time
import pickle
import csv


class GeneticAlgorithm:

    def __init__(self, **kwargs):
        # ---------- Set network parameters ----------- #
        self.network = []
        self.stations_num = kwargs['stations_num']
        self.buses_num = kwargs['buses_num']
        self.min_num_stations = kwargs['min_num_stations']
        self.max_num_stations = kwargs['max_num_stations']
        self.obj_function_weights = kwargs['obj_function_weights']
        self.seed = kwargs["seed"]
        self.initial_solution = []
        self.best_solution = []
        self.archive = []
        self.total_run_time = 0
        self.total_num_of_iterations = 0

        # ---------- Set algorithm parameters ----------- #
        self.population_size = kwargs['population_size']
        self.repeat = kwargs['repeat']

        # ---------- Build network ----------- #
        network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                 self.max_num_stations,
                                 self.obj_function_weights)
        network.create_traffic_network_graph()
        self.network = network.network

        # ---------- Build population set ----------- #
        self.population = []
        scores_array = []
        for net in range(self.population_size):
            temp_network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                          self.max_num_stations, self.obj_function_weights)
            temp_network.network = network.network
            temp_network.find_starting_stations()
            temp_network.find_initial_paths()
            temp_network.check_all_stations_visited()
            temp_network.fix_unvisited_stations()
            temp_network.create_network_from_paths_dict()
            temp_network.check_all_stations_connections()
            temp_network.score_network()
            self.population.append(temp_network)
            self.update_archive(temp_network)
            # Save current population solution in a solution array with all the solutions
            scores_array.append(temp_network.network_score)
            # Find and save the best initial solution in population
        best_solution_index = np.argmin(scores_array)
        self.best_solution = copy.deepcopy(self.population[copy.deepcopy(best_solution_index)])

    def __call__(self):
        """
        Run genetic algorithm and return solve traffic network issue
        :return: solve graph
        """
        for i in range(self.repeat):
            new_population = []
            scores = normalize_scores([net.network_score for net in self.population])
            for j in range(self.population_size):
                x, y, scores = self.random_selection(scores)
                child_net = self.reproduce(x, y)
                if rnd.uniform(0, 1) < 0.1:  # mutate in 0.1 probability
                    child_net = self.mutate(child_net)
                new_population.append(child_net)
                # Update archive and set best solution
                self.archive.append(child_net)
                # self.update_archive(child_net)
                if child_net.network_score < self.best_solution.network_score:
                    self.best_solution = copy.deepcopy(child_net)
            self.population = new_population
            self.total_num_of_iterations += 1
            print("Best score after iteration: ", i + 1, " is: ", self.best_solution.network_score)

        # self.archive = clean_archive(self.archive)

    def random_selection(self, scores):
        """
        select randomly two different nets from population
        :return: x, y --> two random nets
        """
        x = rnd.choices(self.population, weights=scores, k=1)[0]
        x_index = self.population.index(x)
        self.population.remove(x)
        x_score = scores.pop(x_index)
        y = rnd.choices(self.population, weights=scores, k=1)[0]
        self.population.append(x)
        scores.append(x_score)
        return x, y, scores

    def reproduce(self, x, y):
        """
        Create child from x & y nets by paths
        :param x: random net
        :param y: random net
        :return: combination between x and y net by paths
        """
        n = self.buses_num
        c = rnd.randint(1, n)
        child_dict = {}
        for i in range(1, n + 1):
            if i < c:
                child_dict[i] = x.paths_dict[i]
            else:
                child_dict[i] = y.paths_dict[i]
        child_net = self.create_new_network_from_paths_dict(child_dict)
        return child_net

    def mutate(self, child_net):
        """
        add new path to randomly bus in child dict
        :param child_net: dict of bus paths
        :return: child with mutate
        """
        bus_path_to_change = rnd.randint(1, self.buses_num)  # randomly choose bus number
        num_of_stations_in_path_scores = []
        num_of_stations_in_path_networks = []
        current_best_score = child_net.network_score  # The old solution's score
        # Loop through all possible path lengths
        for num_of_stations_in_path in range(self.min_num_stations, self.max_num_stations + 1):
            # Change the entire selected path
            new_paths_dict = child_net.change_all_path_of_specific_bus(child_net.paths_dict, bus_path_to_change, num_of_stations_in_path)
            # Create a new network using the new paths
            new_network = self.create_new_network_from_paths_dict(new_paths_dict)
            # Save the new solution and its score to arrays
            num_of_stations_in_path_scores.append(new_network.network_score)
            num_of_stations_in_path_networks.append(new_network)

        # If there is no changed possible solution
        if len(num_of_stations_in_path_scores) == 0 or current_best_score <= min(num_of_stations_in_path_scores):
            return child_net
        # There is a better new solution
        else:
            max_score_index = np.argmin(num_of_stations_in_path_scores)  # Find the best solution's index
            return num_of_stations_in_path_networks[max_score_index]
        return child_net

    def create_new_network_from_paths_dict(self, paths):
        """
        Create a new network using the given paths, initialize and score it.
        param paths: a dictionary containing all paths
        :return: a new initialized network that has the given paths
        """
        # New network
        new_network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                     self.max_num_stations, self.obj_function_weights)
        new_network.network = self.network
        new_network.paths_dict = paths  # Assign the given paths
        # Fix unvisited stations
        new_network.check_all_stations_visited()
        new_network.fix_unvisited_stations()
        # Create a network containing only the paths
        new_network.create_network_from_paths_dict()
        # Score the network
        new_network.check_all_stations_connections()
        new_network.score_network()
        return new_network

    def update_archive(self, net):
        """
        Check if there is a dominant solution in the archive to the given network.
        If the network is not dominated - add it to the archive.
        If the network dominates an existing solution in the archive - delete this solution.

        param net: the network that is tested for dominance
        """
        dominance_flag = False
        # Loop through the solutions in the archive
        for solution in self.archive:
            # If the current solution in the archive dominates the given solution - flag it
            if check_dominance(solution, net):
                dominance_flag = True
            # If the given solution dominates the solution from the archive - remove the dominated solution
            elif check_dominance(net, solution):
                self.archive.remove(solution)
        # If no solution in the archive dominates the given solution - add it to the archive
        if not dominance_flag:
            self.archive.append(copy.deepcopy(net))


if __name__ == '__main__':

    for permutation in range(10):
        save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\Genetic_Algorithm'
        net_size = '/very_small_network'
        seed = 42 + permutation * 3
        random.seed(seed)
        stations_num = 10
        buses_num = 3
        min_num_stations = 4
        max_num_stations = 6
        population_size = 15

        # ---------- Set algorithm parameters ----------- #
        solutions_array = []
        paths_dict_array = []
        archive_flag = 1    # Use pareto dominant solutions (1) or try given weights for the objective function (0)

        if archive_flag:
            iterations_array = [500]
            weights_combination = [[0.25, 0.25, 0.25, 0.25]]
        else:
            combination_index = 0
            iterations_array = [100, 300, 500, 1000]
            weights_combination = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7],
                                   [0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.1, 0.3], [0.3, 0.1, 0.3, 0.3], [0.1, 0.3, 0.3, 0.3],
                                   [0.5, 0.3, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.5, 0.1, 0.1, 0.3], [0.3, 0.5, 0.1, 0.1],
                                   [0.1, 0.5, 0.3, 0.1], [0.1, 0.5, 0.1, 0.3], [0.3, 0.1, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1],
                                   [0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.3, 0.5], [0.1, 0.3, 0.1, 0.5], [0.3, 0.1, 0.1, 0.5],
                                   [0.25, 0.25, 0.25, 0.25]]

            # Create new dataframe to save all runs results
            results_columns = ['Option_number', 'Decrease_rate', 'Weight_distance', 'Weight_time', 'Weight_frequency',
                               'Weight_connectivity', 'Iterations', 'Total_run_time', 'Average_run_time', 'Score']
            All_runs_results = pd.DataFrame(np.zeros([len(weights_combination) * len(iterations_array), len(results_columns)]),
                                            columns=results_columns)

        # Loop though all the different weights combinations of the objective function (uniform if archive = 1)
        for weight in weights_combination:
            # Loop through all tested temperature decrease rate (0.85 if archive = 1)
            for iteration in iterations_array:

                args = {
                    "stations_num": stations_num,
                    "buses_num": buses_num,
                    "min_num_stations": min_num_stations,
                    "max_num_stations": max_num_stations,
                    "obj_function_weights": weight,
                    "population_size": population_size,
                    "repeat": iteration,
                    "seed": seed
                }

                start_time = time.time()
                GA_experiment = GeneticAlgorithm(**args)
                GA_experiment()
                GA_experiment.total_run_time = time.time() - start_time

                if archive_flag:
                    # Save archive results
                    GA_experiment.archive = clean_archive(GA_experiment.archive)
                    with open('Results/Genetic_Algorithm' + net_size + '/Permutation_Results/GA_Algorithm_Seed' + str(seed), 'wb') as GA:
                        pickle.dump(GA_experiment, GA)

                else:
                    # Save combinations results
                    All_runs_results.loc[combination_index, 'Option_number'] = combination_index + 1
                    All_runs_results.loc[combination_index, 'Population_size'] = GA_experiment.population_size
                    All_runs_results.loc[combination_index, 'Weight_distance'] = weight[0]
                    All_runs_results.loc[combination_index, 'Weight_time'] = weight[1]
                    All_runs_results.loc[combination_index, 'Weight_frequency'] = weight[2]
                    All_runs_results.loc[combination_index, 'Weight_connectivity'] = weight[3]
                    All_runs_results.loc[combination_index, 'Iterations'] = GA_experiment.total_num_of_iterations
                    All_runs_results.loc[combination_index, 'Total_run_time'] = GA_experiment.total_run_time
                    All_runs_results.loc[combination_index, 'Average_run_time'] = np.round(GA_experiment.total_run_time /
                                                                                           GA_experiment.total_num_of_iterations, 2)
                    All_runs_results.loc[combination_index, 'Score'] = np.round(GA_experiment.best_solution.network_score, 4)
                    # Write all results to a csv file
                    All_runs_results.to_csv(save_path + net_size + '/All_runs_results.csv', index=False)

                    # Write paths dictionary to a csv file
                    d = []
                    for P in GA_experiment.best_solution.paths_dict:
                        d.append(GA_experiment.best_solution.paths_dict[P])

                    with open(save_path + net_size + "/Paths/Paths" + str(combination_index + 1) + ".csv", "w+") as f:
                        writer = csv.writer(f)
                        for values in zip_longest(*d):
                            writer.writerow(values)
                    # pd.DataFrame.from_dict(GA_experiment.best_solution.paths_dict).to_csv(save_path + net_size + '/Paths' + str(combination_index + 1) + '.csv', index=False)

                    combination_index += 1


