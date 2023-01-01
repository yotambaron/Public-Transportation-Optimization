from Traffic_Network import *
from Experiments_Utils import *
from itertools import zip_longest
import time
import pickle
import csv


class SimulatedAnnealingAlgorithm:

    def __init__(self, seed_number, initial_temp, threshold_temp, iterations_num, decrease_rate, temperature_length,
                 accepted_proportion, num_of_stations, num_of_buses, min_stations, max_stations, user_weights):
        self.seed = seed_number
        self.initial_temp = initial_temp
        self.threshold_temp = threshold_temp
        self.iterations = iterations_num
        self.decrease_rate = decrease_rate
        self.temperature_length = temperature_length
        self.stations_num = num_of_stations
        self.buses_num = num_of_buses
        self.min_num_stations = min_stations
        self.max_num_stations = max_stations
        self.obj_function_weights = user_weights
        self.accepted_proportion = accepted_proportion
        self.initial_solution = []
        self.best_solution = []
        self.archive = []
        self.network = []
        self.total_run_time = 0
        self.total_num_of_iterations = 0

    def initialize_solution(self):
        """
        Initialize a network and a starting solution:
        - Create a network with the given nodes and weights
        - Find starting stations
        - Find initial paths
        - Make sure all stations are visited
        - Create a network including only the initialized paths
        - Score the new network
        """
        # Create new network with the same weights
        network = TrafficNetwork(self.seed, self.stations_num, self.buses_num, self.min_num_stations,
                                 self.max_num_stations, self.obj_function_weights)
        network.create_traffic_network_graph()
        self.network = network.network
        network.find_starting_stations()
        path_len = int(np.round((self.max_num_stations + self.min_num_stations + 0.1) / 2, 0))
        network.find_initial_paths(path_length=path_len)
        network.check_all_stations_visited()
        network.fix_unvisited_stations()
        network.create_network_from_paths_dict()
        network.check_all_stations_connections()
        network.score_network()
        self.initial_solution = copy.deepcopy(network)
        self.best_solution = copy.deepcopy(network)
        self.archive.append(network)

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

    def delete_random_station(self, solution):
        """
        Choose a random path from the given solution.
        Compute the best station to delete from the path.
        Delete the station and update the solution.

        param solution: a given solution to try to improve
        :return: the best solution found by deleting one station from the chosen path
        """
        chosen_bus = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a bus randomly
        path_to_delete_station = solution.paths_dict[chosen_bus]    # The chosen bus path
        best_score = solution.network_score     # Best score we have seen during the process
        best_solution = copy.deepcopy(solution)    # The solution that corresponds to the best score seen so far
        # Run through all the stations in the path and delete the one that gives the best score if deleted
        for station in range(len(path_to_delete_station)):
            score = 0
            temp_paths_dict = solution.paths_dict.copy()    # The original paths dictionary
            temp_path = path_to_delete_station.copy()   # The original path
            temp_path.pop(station)  # Delete the current station from the path
            temp_paths_dict[chosen_bus] = temp_path     # Update paths dictionary
            temp_network = self.create_new_network_from_paths_dict(temp_paths_dict)     # Create a new network after the change
            # Connectivity score
            score += temp_network.user_weights[3] * len(temp_network.stations_without_path) / temp_network.len_all_stations_combinations
            for bus in temp_network.paths_dict.keys():
                # Sum and normalize all weights
                bus_path = temp_network.paths_dict[bus]
                # Weight of the first node in the path
                score += temp_network.user_weights[2] * -1 * list(nx.get_node_attributes(temp_network.network, 'weight').values())[bus_path[0] - 1]
                for node in range(len(bus_path) - 1):
                    temp_score, dis_weight, travel_weight, sta_weight = temp_network.compute_neighbor_score([bus_path[node], bus_path[node + 1]])
                    score += temp_score
            # Update archive with the current changed solution
            self.update_archive(temp_network)
            # If the current change is the best seen score so far and is legal (more than the minimum stations)
            if score < best_score and len(temp_path) >= self.min_num_stations:
                # Make the current solution the best seen so far
                best_score = score
                best_solution = copy.deepcopy(temp_network)
        return best_solution

    def add_random_station(self, solution):
        """
        Choose a random path from the given solution.
        Choose a random station that is not already in the chosen path.
        Compute the best position to insert the station and add it to the path.

        param solution: a given solution to try to improve
        :return: the best solution found by inserting one station to the chosen path
        """
        chosen_bus = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a bus randomly
        path_to_add_station = solution.paths_dict[chosen_bus]    # The chosen bus path
        best_score = solution.network_score     # Best score we have seen during the process
        best_solution = copy.deepcopy(solution)    # The solution that corresponds to the best score seen so far
        # Possible stations to add to the chosen path
        optional_stations = [node for node in range(1, self.stations_num + 1) if node not in path_to_add_station]
        chosen_station = np.random.choice(optional_stations)    # Choose randomly a station from optional stations
        # Run through all the stations in the path and add the chosen station in the best position
        for position in range(len(path_to_add_station)):
            score = 0
            temp_paths_dict = solution.paths_dict.copy()    # The original paths dictionary
            temp_path = path_to_add_station.copy()   # The original path
            temp_path.insert(position, chosen_station)  # Add the chosen station to the current position
            temp_paths_dict[chosen_bus] = temp_path     # Update paths dictionary
            temp_network = self.create_new_network_from_paths_dict(temp_paths_dict)     # Create a new network after the change
            # Connectivity score
            score += temp_network.user_weights[3] * len(temp_network.stations_without_path) / temp_network.len_all_stations_combinations
            for bus in temp_network.paths_dict.keys():
                # Sum and normalize all weights
                bus_path = temp_network.paths_dict[bus]
                # Weight of the first node in the path
                score += temp_network.user_weights[2] * -1 * list(nx.get_node_attributes(temp_network.network, 'weight').values())[bus_path[0] - 1]
                for node in range(len(bus_path) - 1):
                    temp_score, dis_weight, travel_weight, sta_weight = temp_network.compute_neighbor_score([bus_path[node], bus_path[node + 1]])
                    score += temp_score
            # Update archive with the current changed solution
            self.update_archive(temp_network)
            # If the current change is the best seen score so far and is legal (less than the maximum stations)
            if score < best_score and len(temp_path) <= self.max_num_stations:
                # Make the current solution the best seen so far
                best_score = score
                best_solution = copy.deepcopy(temp_network)
        return best_solution

    def swap_random_station(self, solution):
        """
        Choose a random path from the given solution.
        Choose a random station that is not already in the chosen path.
        Choose a random position in the chosen path.
        Swap the current station in the selected position in the path with the new selected station.

        param solution: a given solution to try to improve
        :return: the best solution found by swapping one of the stations in the chosen path
        """
        chosen_bus = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a bus randomly
        path_to_swap_station = solution.paths_dict[chosen_bus]    # The chosen bus path
        chosen_station_index = np.random.choice(list(range(len(path_to_swap_station))))  # Choose a random station index
        # Possible stations to add to the chosen path
        optional_stations = [node for node in range(1, self.stations_num + 1) if node not in path_to_swap_station]
        chosen_station = np.random.choice(optional_stations)    # Choose randomly a station from optional stations
        path_to_swap_station[chosen_station_index] = chosen_station     # Insert the chosen station in the chosen index
        # Insert the new path to the given paths
        solution.paths_dict[chosen_bus] = path_to_swap_station
        # Update the network with the new solution
        new_network = self.create_new_network_from_paths_dict(solution.paths_dict)  # Create a new network after the change
        # Update archive with the current changed solution
        self.update_archive(new_network)
        return new_network

    def swap_two_stations_in_same_path(self, solution):
        """
        Choose a random path from the given solution.
        Choose a random position in the chosen path.
        Calculate which other position will give the best score if swapped with the first chosen station.
        Swap the two chosen stations in the path.

        param solution: a given solution to try to improve
        :return: the best solution found by swapping the two chosen stations in the chosen path
        """
        best_score = solution.network_score     # Best score we have seen during the process
        best_solution = copy.deepcopy(solution)    # The best solution found
        chosen_bus = np.random.choice(list(range(1, self.buses_num + 1)))   # Choose a bus randomly
        path_to_swap_station = solution.paths_dict[chosen_bus]    # The chosen bus path
        chosen_station_index = np.random.choice(list(range(len(path_to_swap_station))))  # Choose a random station index
        # Loop through all the stations in the path
        for station in range(len(path_to_swap_station)):
            temp_paths_dict = solution.paths_dict.copy()    # Copy the original paths dictionary
            temp_path = path_to_swap_station.copy()     # Copy the chosen path to swap
            # Compute the score of all stations to swap except the selected one
            if not station == chosen_station_index:
                score = 0
                temp_station = temp_path[station]   # First station to swap
                temp_path[station] = temp_path[chosen_station_index]    # Swap first station
                temp_path[chosen_station_index] = temp_station  # Swap second station
                temp_paths_dict[chosen_bus] = temp_path     # Save changed path
                temp_network = self.create_new_network_from_paths_dict(temp_paths_dict)  # Create a new network after the change
                # Connectivity score
                score += temp_network.user_weights[3] * len(temp_network.stations_without_path) / temp_network.len_all_stations_combinations
                for bus in temp_network.paths_dict.keys():
                    # Sum and normalize all weights
                    bus_path = temp_network.paths_dict[bus]
                    # Weight of the first node in the path
                    score += temp_network.user_weights[2] * -1 * list(nx.get_node_attributes(temp_network.network, 'weight').values())[bus_path[0] - 1]
                    for node in range(len(bus_path) - 1):
                        temp_score, dis_weight, travel_weight, sta_weight = temp_network.compute_neighbor_score(
                            [bus_path[node], bus_path[node + 1]])
                        score += temp_score
                # Update archive with the current changed solution
                self.update_archive(temp_network)
                # If the current change is the best seen score so far and is legal (less than the maximum stations)
                if score < best_score:
                    # Make the current solution the best seen so far
                    best_score = score
                    best_solution = copy.deepcopy(temp_network)
        return best_solution

    def swap_two_stations_in_different_paths(self, solution):
        """
        Choose a random path from the given solution.
        Choose a random position in the chosen path.
        choose another random path.
        Calculate which other position in the second path will give the best score if swapped with the first station.
        Swap the two chosen stations in the two chosen paths.

        param solution: a given solution to try to improve
        :return: the best solution found by swapping the two chosen stations in the chosen path
        """
        best_score = solution.network_score     # Best score we have seen during the process
        best_solution = copy.deepcopy(solution)    # The best solution found
        buses_list = list(range(1, self.buses_num + 1))
        bus_path1 = np.random.choice(buses_list)    # Choose a random bus path
        buses_list.remove(bus_path1)    # Remove the chosen path from the optional second bus path to choose
        bus_path2 = np.random.choice(buses_list)    # Choose randomly the second bus path
        station_index1 = np.random.choice(list(range(len(solution.paths_dict[bus_path1]))))  # Choose a random first station index
        temp_paths_dict = solution.paths_dict.copy()  # Copy the original paths dictionary
        # If the first chosen station is already in the second path - switch a bus path
        while temp_paths_dict[bus_path1][station_index1] in temp_paths_dict[bus_path2]:
            buses_list.remove(bus_path2)
            if len(buses_list) == 0:
                buses_list = list(range(1, self.buses_num + 1))
                buses_list.remove(bus_path1)
                bus_path1 = np.random.choice(buses_list)
                station_index1 = np.random.choice(list(range(len(solution.paths_dict[bus_path1]))))
                buses_list = list(range(1, self.buses_num + 1))
                buses_list.remove(bus_path1)
                bus_path2 = np.random.choice(buses_list)
            else:
                bus_path2 = np.random.choice(buses_list)
        # Loop through all the stations in the second path
        for station in range(len(solution.paths_dict[bus_path2])):
            temp_paths_dict = solution.paths_dict.copy()    # Copy the original paths dictionary
            # If the second station is not already in the first bus path
            if temp_paths_dict[bus_path2][station] not in temp_paths_dict[bus_path1]:
                first_path = temp_paths_dict[bus_path1]  # The first chosen bus path
                second_path = temp_paths_dict[bus_path2]  # The second chosen bus path
                temp_station = second_path[station]     # Copy the chosen station in the second path to swap
                second_path[station] = first_path[station_index1]   # Second path station = first path station
                first_path[station_index1] = temp_station   # First path station = second path station
                temp_paths_dict[bus_path1] = first_path     # Insert the first changed path
                temp_paths_dict[bus_path2] = second_path    # Insert the second changed path
                # Compute the score the new paths after the swap
                score = 0
                temp_network = self.create_new_network_from_paths_dict(temp_paths_dict)  # Create a new network after the change
                # Connectivity score
                score += temp_network.user_weights[3] * len(temp_network.stations_without_path) / temp_network.len_all_stations_combinations
                for bus in temp_network.paths_dict.keys():
                    # Sum and normalize all weights
                    bus_path = temp_network.paths_dict[bus]
                    # Weight of the first node in the path
                    score += temp_network.user_weights[2] * -1 * list(nx.get_node_attributes(temp_network.network, 'weight').values())[bus_path[0] - 1]
                    for node in range(len(bus_path) - 1):
                        temp_score, dis_weight, travel_weight, sta_weight = temp_network.compute_neighbor_score([bus_path[node], bus_path[node + 1]])
                        score += temp_score
                # Update archive with the current changed solution
                self.update_archive(temp_network)
                # If the current change is the best seen score so far
                if score < best_score:
                    # Make the current solution the best seen so far
                    best_score = score
                    best_solution = copy.deepcopy(temp_network)
        return best_solution

    def improve_solution(self, solution):
        solution = self.delete_random_station(solution)
        solution = self.add_random_station(solution)
        solution = self.swap_random_station(solution)
        solution = self.swap_two_stations_in_same_path(solution)
        # solution = self.swap_two_stations_in_different_paths(solution)
        return solution

    def find_solutions_using_simulated_annealing(self):
        T = self.initial_temp   # Save initial temperature
        iters = 0
        accepted = 0
        rejected = 0
        current_solution = self.initial_solution    # The current solution we are looking at
        self.update_archive(current_solution)   # Insert the initial solution to the archive
        # While stopping criteria run SA iterations
        while iters <= self.iterations and T > self.threshold_temp:
            # Run TL iterations with the current temperature
            for curr_temp in range(TL):
                temp_solution = copy.deepcopy(current_solution)
                improved_solution = self.improve_solution(temp_solution)     # Try to improve solution
                self.update_archive(improved_solution)  # Update the archive with the improved solution
                # If the new solution is the best ever found
                if improved_solution.network_score < self.best_solution.network_score:
                    self.best_solution = copy.deepcopy(improved_solution)    # Save the current solution to best solution
                # delta for the SA probability
                delta = improved_solution.network_score - current_solution.network_score
                if delta < 0:   # Current solution is better than the old one
                    accepted += 1   # Increase the accepted solutions by one
                    current_solution = copy.deepcopy(improved_solution)      # Accept current solution
                else:   # The new solution is not better than the old one
                    rejected += 1   # Increase the rejected solutions by one
                    iters += 1
                    p = np.power(np.e, -delta / T)    # Compute the probability for SA
                    if random.random() < p:     # Accept the new worse solution with the computed probability
                        current_solution = copy.deepcopy(improved_solution)

            print("Best score after iteration: ", accepted + rejected, " is: ", self.best_solution.network_score)
            T = T * self.decrease_rate  # Decrease the current temperature by 'decrease rate'
            self.total_num_of_iterations = accepted + rejected  # Save the number of iterations

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
    for permutation in range(10):
        save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\Simulated_Annealing'
        net_size = '/very_small_network'
        seed = 42 + permutation * 3
        random.seed(seed)
        stations_num = 10
        buses_num = 3
        min_num_stations = 4
        max_num_stations = 6

        # ---------- Set algorithm parameters ----------- #
        TI = 300
        TL = 150
        temp_threshold = 1
        iterations = 1000
        solutions_array = []
        paths_dict_array = []
        archive_flag = 1    # Use pareto dominant solutions (1) or try given weights for the objective function (0)

        if archive_flag:
            decrease_temp = [0.85]
            weights_combination = [[0.25, 0.25, 0.25, 0.25]]
        else:
            combination_index = 0
            decrease_temp = [0.75, 0.8, 0.85, 0.9]
            weights_combination = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7],
                                   [0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.1, 0.3], [0.3, 0.1, 0.3, 0.3], [0.1, 0.3, 0.3, 0.3],
                                   [0.5, 0.3, 0.1, 0.1], [0.5, 0.1, 0.3, 0.1], [0.5, 0.1, 0.1, 0.3], [0.3, 0.5, 0.1, 0.1],
                                   [0.1, 0.5, 0.3, 0.1], [0.1, 0.5, 0.1, 0.3], [0.3, 0.1, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1],
                                   [0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.3, 0.5], [0.1, 0.3, 0.1, 0.5], [0.3, 0.1, 0.1, 0.5],
                                   [0.25, 0.25, 0.25, 0.25]]

            # Create new dataframe to save all runs results
            results_columns = ['Option_number', 'Decrease_rate', 'Weight_distance', 'Weight_time', 'Weight_frequency',
                               'Weight_connectivity', 'Iterations', 'Total_run_time', 'Average_run_time', 'Score']
            All_runs_results = pd.DataFrame(np.zeros([len(weights_combination) * len(decrease_temp), len(results_columns)]),
                                            columns=results_columns)

        # Loop though all the different weights combinations of the objective function (uniform if archive = 1)
        for weight in weights_combination:
            # Loop through all tested temperature decrease rate (0.85 if archive = 1)
            for rate in decrease_temp:
                iteration = 1
                no_improvement = 1
                # ---------- Build experiment ----------- #
                SA_experiment = SimulatedAnnealingAlgorithm(seed, TI, temp_threshold, iterations, rate, TL, accept_proportion,
                                                            stations_num, buses_num, min_num_stations, max_num_stations, weight)

                # ---------- Initialize the experiment and run it ----------- #
                SA_experiment.initialize_solution()
                start_time = time.time()    # Measure run time of the iteration
                SA_experiment.find_solutions_using_simulated_annealing()
                total_run_time = time.time() - start_time

                SA_experiment.total_run_time = np.round(total_run_time / 60, 2)

                if archive_flag:
                    # Save archive results
                    clean_archive(SA_experiment)
                    with open('Results/Simulated_Annealing' + net_size + '/SA_Algorithm_Seed' + str(seed), 'wb') as SA:
                        pickle.dump(SA_experiment, SA)

                else:
                    # Save combinations results
                    All_runs_results.loc[combination_index, 'Option_number'] = combination_index + 1
                    All_runs_results.loc[combination_index, 'Decrease_rate'] = rate
                    All_runs_results.loc[combination_index, 'Weight_distance'] = weight[0]
                    All_runs_results.loc[combination_index, 'Weight_time'] = weight[1]
                    All_runs_results.loc[combination_index, 'Weight_frequency'] = weight[2]
                    All_runs_results.loc[combination_index, 'Weight_connectivity'] = weight[3]
                    All_runs_results.loc[combination_index, 'Iterations'] = SA_experiment.iterations
                    All_runs_results.loc[combination_index, 'Total_run_time'] = SA_experiment.total_run_time
                    All_runs_results.loc[combination_index, 'Average_run_time'] = np.round(SA_experiment.total_run_time /
                                                                                           SA_experiment.iterations, 2)
                    All_runs_results.loc[combination_index, 'Score'] = np.round(SA_experiment.best_solution.network_score, 4)
                    # Write all results to a csv file
                    All_runs_results.to_csv(save_path + net_size + '/All_runs_results.csv', index=False)

                    # Write paths dictionary to a csv file
                    d = []
                    for P in SA_experiment.best_solution.paths_dict:
                        d.append(SA_experiment.best_solution.paths_dict[P])

                    with open(save_path + net_size + "/Paths/Paths" + str(combination_index + 1) + ".csv", "w+") as f:
                        writer = csv.writer(f)
                        for values in zip_longest(*d):
                            writer.writerow(values)
                    # pd.DataFrame.from_dict(SA_experiment.best_solution.paths_dict).to_csv(save_path + net_size + '/Paths' + str(combination_index + 1) + '.csv', index=False)

                    combination_index += 1

    # Extract the best results from each weight combination and save the best results as a dataframe
    # save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\Simulated_Annealing'
    # net_size = '/very_small_network'
    # results = pd.read_csv(save_path + net_size + '/All_runs_results.csv')
    # best_results = best_scores_extraction(results, weights_combination)
    # best_results['Option_number'] = range(1, len(weights_combination) + 1)
    # best_results.to_csv(save_path + net_size + '/All_best_results.csv', index=False)
    #
    # save_path = r'C:\Users\User\PycharmProjects\Computational_Intelligence\Results\NSABC'
    # net_size = '/large_network'
    # with open(save_path + net_size + '/Permutation_Results/NSABC_Algorithm_Seed42', 'rb') as SA_exp:
    #     SA = pickle.load(SA_exp)






