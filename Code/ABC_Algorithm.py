from Traffic_Network import *
from Experiments_Utils import *
import time
import pickle


class NSABCAlgorithm:

    def __init__(self, seed_number, employed_bees, onlooker_bees, limit_num, iterations_num, num_of_stations,
                 num_of_buses, min_stations, max_stations, user_weights):
        self.seed = seed_number
        self.network = []
        self.solutions_array = []
        self.scores_array = []
        self.scores_array_normalized = []
        self.paths_dict_array = []
        self.num_of_employed_bees = employed_bees
        self.num_of_onlooker_bees = onlooker_bees
        self.limit = limit_num
        self.limit_array = np.zeros(employed_bees)
        self.iterations = iterations_num
        self.stations_num = num_of_stations
        self.buses_num = num_of_buses
        self.min_num_stations = min_stations
        self.max_num_stations = max_stations
        self.obj_function_weights = user_weights
        self.onlookers_current_solution_array = []
        self.best_solution = []
        self.archive = []
        self.total_run_time = 0
        self.total_num_of_iterations = 0

    def initialize_solutions(self):
        """
        Initialize solutions for each employee bee.
        Initial paths are set using a stochastic procedure (corresponding to nodes and edges weights).
        """
        # Create new network with the same weights
        network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                 self.max_num_stations, self.obj_function_weights)
        network.create_traffic_network_graph()
        self.network = network.network
        # For each employed bee initialize its network
        for net in range(self.num_of_employed_bees):
            temp_network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                          self.max_num_stations, self.obj_function_weights)
            temp_network.network = self.network
            temp_network.find_starting_stations()
            temp_network.find_initial_paths()
            temp_network.check_all_stations_visited()
            temp_network.fix_unvisited_stations()
            temp_network.create_network_from_paths_dict()
            temp_network.check_all_stations_connections()
            temp_network.score_network()
            # Save current employed bee's solution in a solution array with all the employed bees' solutions
            self.solutions_array.append(temp_network)
            self.scores_array.append(temp_network.network_score)
            self.paths_dict_array.append(temp_network.paths_dict)
        # Find and save the best initial solution of the employed bees
        best_solution_index = np.argmin(self.scores_array)
        self.best_solution = copy.deepcopy(self.solutions_array[copy.deepcopy(best_solution_index)])

    def normalize_scores(self):
        """
        Normalize the scores of the employed bees' solutions - high scores are worse

        param scores: scores of the current employed bees' solutions
        :return: normalized scores
        """
        self.scores_array_normalized = np.array(self.scores_array) - min(self.scores_array)
        self.scores_array_normalized = -1 * (self.scores_array_normalized - max(self.scores_array_normalized)) + 0.01
        self.scores_array_normalized = list(self.scores_array_normalized / sum(self.scores_array_normalized))

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

    def assign_onlookers_to_solutions(self):
        """
        Each onlooker bee chooses one employed bee solution (food source) to try and improve.
        The solution is chosen corresponding to how good the solution is compared to all existing solutions.
        """
        self.onlookers_current_solution_array = np.random.choice(list(range(self.num_of_employed_bees)),
                                                                 size=self.num_of_employed_bees,
                                                                 p=self.scores_array_normalized)

    def onlookers_improve_solutions(self):
        """
        Each onlooker bee goes to its chosen employed bee's solution and try to improve it by changing one of its paths.
        The path is chosen randomly.
        Every path length between the minimum and the maximum number of stations is tested and the best is chosen.
        """
        # Loop through every onlooker bee
        for onlooker in range(self.num_of_onlooker_bees):
            solution_to_change = self.onlookers_current_solution_array[onlooker]   # The employed bee solution to change
            path_dict_to_change = self.paths_dict_array[solution_to_change]     # The paths of the selected solution
            bus_path_to_change = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a random path
            current_best_score = self.solutions_array[solution_to_change].network_score     # The old solution's score
            num_of_stations_in_path_scores = []     # An array of all possible solutions' scores
            num_of_stations_in_path_networks = []   # An array of all possible solutions

            # Loop through all possible path lengths
            for num_of_stations_in_path in range(self.min_num_stations, self.max_num_stations + 1):
                # Change the entire selected path
                new_paths_dict = self.solutions_array[solution_to_change]. \
                    change_all_path_of_specific_bus(path_dict_to_change, bus_path_to_change, num_of_stations_in_path)
                # Create a new network using the new paths
                new_network = self.create_new_network_from_paths_dict(new_paths_dict)
                # Save the new solution and its score to arrays
                num_of_stations_in_path_scores.append(new_network.network_score)
                num_of_stations_in_path_networks.append(new_network)

            # If there is no changed possible solution
            if len(num_of_stations_in_path_scores) == 0:
                print("No solutions")
            # If there is no improvement in the newly found solutions
            elif current_best_score <= min(num_of_stations_in_path_scores):
                self.limit_array[solution_to_change] += 1   # Increase by 1 the limit count of the unimproved solution
                if self.limit_array[solution_to_change] >= self.limit:  # If the limit count exceeds the limit threshold
                    # Create a new solution
                    self.solutions_array[solution_to_change] = initialize_solution(
                        self.solutions_array[solution_to_change])
                    # Save new solution
                    self.scores_array[solution_to_change] = self.solutions_array[solution_to_change].network_score
                    self.paths_dict_array[solution_to_change] = self.solutions_array[solution_to_change].paths_dict
                    self.limit_array[solution_to_change] = 0    # Zero the limit count

            # There is a better new solution
            else:
                max_score_index = np.argmin(num_of_stations_in_path_scores)     # Find the best solution's index
                max_score = num_of_stations_in_path_scores[max_score_index]     # Find the best solution's score
                # Save new better solution
                self.solutions_array[solution_to_change] = num_of_stations_in_path_networks[max_score_index]
                self.scores_array[solution_to_change] = max_score
                self.paths_dict_array[solution_to_change] = self.solutions_array[solution_to_change].paths_dict
                # If the new improved solution is the best solution ever found
                if max_score < self.best_solution.network_score:
                    # Copy the best network to the best solution attribute
                    self.best_solution = copy.deepcopy(self.solutions_array[copy.deepcopy(solution_to_change)])

            # Update the archive
            self.update_archive(self.solutions_array[solution_to_change])
            self.update_archive(self.best_solution)

    def employees_improve_solutions(self):
        """
        Each employed bee tries to improve its solution by changing part of one of its paths.
        The path is chosen randomly.
        The starting station to change in the path is randomly selected.
        Every path length between the minimum and the maximum number of stations is tested and the best is chosen.
        """
        # Loop through every employed bee
        for employee in range(self.num_of_employed_bees):
            path_dict_to_change = self.paths_dict_array[employee]   # The paths of the current employed bee
            bus_path_to_change = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a random path
            current_path = path_dict_to_change[bus_path_to_change]
            current_best_score = self.solutions_array[employee].network_score    # The old solution's score
            num_of_stations_in_path_scores = []     # An array of all possible solutions' scores
            num_of_stations_in_path_networks = []   # An array of all possible solutions

            # Choose the index of the station to start changing the path from
            start_station_index = np.random.choice(list(range(1, len(current_path))))
            # Number of stations that will not be changed (before the start station)
            stations_unchanged = start_station_index

            # Loop through all possible path lengths
            for num_of_stations_in_path in range(max(1, self.min_num_stations - stations_unchanged + 1),
                                                 self.max_num_stations - stations_unchanged + 2):
                # Change partially the selected path starting with start station index
                new_paths_dict = self.solutions_array[employee].change_partial_path_of_specific_bus(
                    path_dict_to_change, bus_path_to_change, num_of_stations_in_path, start_station_index)
                # Create a new network using the new paths
                new_network = self.create_new_network_from_paths_dict(new_paths_dict)
                # Save the new solution and its score to arrays
                num_of_stations_in_path_scores.append(new_network.network_score)
                num_of_stations_in_path_networks.append(new_network)

            # If there is no changed possible solution
            if len(num_of_stations_in_path_scores) == 0:
                print("No solutions")
            # If there is no improvement in the newly found solutions
            elif current_best_score <= min(num_of_stations_in_path_scores):
                self.limit_array[employee] += 1     # Increase by 1 the limit count of the unimproved solution
                if self.limit_array[employee] >= self.limit:    # If the limit count exceeds the limit threshold
                    # Create a new solution
                    self.solutions_array[employee] = initialize_solution(self.solutions_array[employee])
                    # Save new solution
                    self.scores_array[employee] = self.solutions_array[employee].network_score
                    self.paths_dict_array[employee] = self.solutions_array[employee].paths_dict
                    self.limit_array[employee] = 0      # Zero the limit count

            # There is a better new solution
            else:
                max_score_index = np.argmin(num_of_stations_in_path_scores)     # Find the best solution's index
                max_score = num_of_stations_in_path_scores[max_score_index]     # Find the best solution's score
                # Save new better solution
                self.solutions_array[employee] = num_of_stations_in_path_networks[max_score_index]
                self.scores_array[employee] = max_score
                self.paths_dict_array[employee] = self.solutions_array[employee].paths_dict
                # If the new improved solution is the best solution ever found
                if max_score < self.best_solution.network_score:
                    # Copy the best network to the best solution attribute
                    self.best_solution = copy.deepcopy(self.solutions_array[copy.deepcopy(employee)])

            # Update the archive
            self.update_archive(self.solutions_array[employee])
            self.update_archive(self.best_solution)

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

    # ---------- Set network parameters ----------- #
    for permutation in range(1):
        save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\NSABC'
        net_size = '/very_small_network'
        seed = 42 + permutation * 3
        random.seed(seed)
        stations_num = 10
        buses_num = 3
        min_num_stations = 4
        max_num_stations = 6

        # ---------- Set algorithm parameters ----------- #
        num_bees_employed = 15
        num_bees_onlooker = 15
        iterations = 1000
        improvement_limit = 150
        solutions_array = []
        paths_dict_array = []
        archive_flag = 1    # Use pareto dominant solutions (1) or try given weights for the objective function (0)

        if archive_flag:
            limit_array = [50]
            weights_combination = [[0.25, 0.25, 0.25, 0.25]]
        else:
            combination_index = 0
            limit_array = [10, 30, 50, 100, 150]
            weights_combination = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7],
                                   [0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.1, 0.3], [0.3, 0.1, 0.3, 0.3], [0.1, 0.3, 0.3, 0.3],
                                   [0.5, 0.3, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.5, 0.1, 0.1, 0.3], [0.3, 0.5, 0.1, 0.1],
                                   [0.1, 0.5, 0.3, 0.1], [0.1, 0.5, 0.1, 0.3], [0.3, 0.1, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1],
                                   [0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.3, 0.5], [0.1, 0.3, 0.1, 0.5], [0.3, 0.1, 0.1, 0.5],
                                   [0.25, 0.25, 0.25, 0.25]]

            # Create new dataframe to save all runs results
            results_columns = ['Option_number', 'Employed_bees', 'Limit', 'Weight_distance', 'Weight_time', 'Weight_frequency',
                               'Weight_connectivity', 'Iterations', 'Total_run_time', 'Average_run_time', 'Score']
            All_runs_results = pd.DataFrame(np.zeros([len(weights_combination) * len(limit_array), len(results_columns)]),
                                            columns=results_columns)

        # Loop though all the different weights combinations of the objective function (uniform if archive = 1)
        for weight in weights_combination:
            # Loop through all tested limits (50 if archive = 1)
            for limit in limit_array:
                iteration = 1
                no_improvement = 1
                # ---------- Build experiment ----------- #
                NSABC_experiment = NSABCAlgorithm(seed, num_bees_employed, num_bees_onlooker, limit, iterations, stations_num,
                                                  buses_num, min_num_stations, max_num_stations, weight)

                # ---------- Find initial solutions ----------- #
                NSABC_experiment.initialize_solutions()
                best_score = NSABC_experiment.best_solution.network_score
                # Update the archive with each initial solution
                for sol in NSABC_experiment.solutions_array:
                    NSABC_experiment.update_archive(sol)

                start_time = time.time()    # Measure run time of the iteration
                # Make another iteration as long as max number of iterations and no improvement are not exceeded
                while iteration <= iterations and no_improvement <= improvement_limit:
                    # Normalize current scores to get solution's probability
                    NSABC_experiment.normalize_scores()
                    # Assign each onlooker bee to a solution (according to the computed probabilities)
                    NSABC_experiment.assign_onlookers_to_solutions()
                    # Onlooker bees try to improve their selected solutions
                    NSABC_experiment.onlookers_improve_solutions()
                    # Employed bees try to improve their solutions
                    NSABC_experiment.employees_improve_solutions()
                    # Save the best score of the iteration
                    iteration_best_score = NSABC_experiment.best_solution.network_score
                    # If there is an improvement after this iteration
                    if iteration_best_score < best_score:
                        best_score = iteration_best_score
                        no_improvement = 0  # Zero no improvement count
                    else:
                        # If there is no improvement in this iteration - increase by 1 the no improvement count
                        no_improvement += 1
                    print("Best score after iteration: ", iteration, " is: ",
                          np.round(NSABC_experiment.best_solution.network_score, 4))
                    iteration += 1

                total_run_time = time.time() - start_time
                print("Total run time is: ", np.round(total_run_time / 60, 2), "minutes")
                print("Average run time per iteration was: ", np.round(total_run_time / iteration, 2), "seconds")

                if archive_flag:
                    # Save archive results
                    clean_archive(NSABC_experiment)
                    NSABC_experiment.total_run_time = np.round(total_run_time / 60, 2)
                    NSABC_experiment.total_num_of_iterations = iteration
                    with open(save_path + net_size + '/NSABC_Algorithm_Seed' + str(seed), 'wb') as NSABC:
                        pickle.dump(NSABC_experiment, NSABC)

                else:
                    # Save combinations results
                    All_runs_results.loc[combination_index, 'Option_number'] = combination_index + 1
                    All_runs_results.loc[combination_index, 'Employed_bees'] = num_bees_employed
                    All_runs_results.loc[combination_index, 'Limit'] = limit
                    All_runs_results.loc[combination_index, 'Weight_distance'] = weight[0]
                    All_runs_results.loc[combination_index, 'Weight_time'] = weight[1]
                    All_runs_results.loc[combination_index, 'Weight_frequency'] = weight[2]
                    All_runs_results.loc[combination_index, 'Weight_connectivity'] = weight[3]
                    All_runs_results.loc[combination_index, 'Iterations'] = iteration
                    All_runs_results.loc[combination_index, 'Total_run_time'] = np.round(total_run_time / 60, 2)
                    All_runs_results.loc[combination_index, 'Average_run_time'] = np.round(total_run_time / iteration, 2)
                    All_runs_results.loc[combination_index, 'Score'] = np.round(NSABC_experiment.best_solution.network_score, 4)

                    pd.DataFrame.from_dict(NSABC_experiment.best_solution.paths_dict).to_csv(save_path + '/Paths' + str(combination_index + 1) + '.csv', index=False)
                    All_runs_results.to_csv(save_path + '/All_runs_results.csv', index=False)

                    combination_index += 1


    # Extract the best results from each weight combination and save the best results as a dataframe
    # save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\NSABC'
    # net_size = '/very_small_network'
    # results = pd.read_csv(save_path + net_size + '/All_runs_results.csv')
    # best_results = best_scores_extraction(results, weights_combination)
    # best_results['Option_number'] = range(1, len(weights_combination) + 1)
    # best_results.to_csv(save_path + net_size + '/All_best_results.csv', index=False)
    #
    # save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\NSABC'
    # net_size = '/big_network'
    # with open(save_path + net_size + '/Permutation_Results/NSABC_Algorithm_Seed63', 'rb') as NSABC_exp:
    #     NSA = pickle.load(NSABC_exp)
    #
    # with open(save_path + net_size + '/Permutation_Results/NSABC_Algorithm_Seed63', 'wb') as NSABC_exp:
    #     pickle.dump(NSA, NSABC_exp)


