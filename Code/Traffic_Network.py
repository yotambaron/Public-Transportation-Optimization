from Experiments_Utils import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations
import copy


class TrafficNetwork:

    def __init__(self, seed, num_of_stations, num_of_buses, min_num_of_stations, max_num_of_stations, user_weights):
        self.seed = seed
        random.seed(self.seed)
        self.network = []
        self.starting_stations = []
        self.paths_dict = {}
        self.stations_frequency = []
        self.unvisited_stations = []
        self.stations_without_path = []
        self.current_path_network = []
        self.network_score = 9999999
        self.distance_score = 0
        self.travel_time_score = 0
        self.stations_score = 0
        self.connectivity_score = 0
        self.num_of_stations = num_of_stations
        self.num_of_buses = num_of_buses
        self.min_num_of_stations = min_num_of_stations
        self.max_num_of_stations = max_num_of_stations
        self.user_weights = user_weights
        self.len_all_stations_combinations = len(list(combinations(range(num_of_stations), 2)))

    def create_traffic_network_graph(self):
        """
        Create a network with given amount of nodes, number of buses, min and max stations for each bus.
        Assign random weights for stations frequency, travel time and distance.
        """
        random.seed(self.seed)
        num_of_nodes = self.num_of_stations
        data = np.zeros([self.len_all_stations_combinations, 4], dtype=int)
        edges_weights = pd.DataFrame(data, columns=['Station1', 'Station2', 'Weight_Distance', 'Weight_Time'])
        counter = 0
        for i in range(1, num_of_nodes + 1):
            for j in range(i + 1, num_of_nodes + 1):
                # Add edge from every node to every node with a random weight for distance and travel time
                edges_weights.iloc[counter] = [i, j, random.random(), random.random()]
                counter += 1

        stations_weights = [random.random() for i in range(num_of_nodes)]   # random weight for station frequency

        # create the network
        G = nx.Graph()
        for node in range(1, num_of_nodes + 1):
            G.add_node(node, weight=stations_weights[node - 1])

        for ind in range(len(edges_weights)):
            edge = edges_weights.iloc[ind, :]
            G.add_edge(edge['Station1'], edge['Station2'],
                       weight={'Distance': edge['Weight_Distance'], 'Time': edge['Weight_Time']})
        self.network = G

    def draw_network(self, net):
        """
        Plot a given network with nodes' size that correspond to their weights
        """
        pos = nx.fruchterman_reingold_layout(nx.complete_graph(range(1, self.num_of_stations + 1)), k=5.0, iterations=20)
        nx.draw(net, node_size=(np.array(list(nx.get_node_attributes(net, 'weight').values())) * 2000), pos=pos,
                with_labels=True)
        ax = plt.gca()  # to get the current axis
        ax.collections[0].set_edgecolor("black")

    def find_starting_stations(self):
        """
        Find starting stations for every bus - stations are chosen by a probability corresponding to their weight
        """
        random.seed(self.seed)
        starting_stations = []
        weights_stations = list(nx.get_node_attributes(self.network, 'weight').values())
        weights_stations = list(np.array(weights_stations) / sum(weights_stations))     # normalize nodes' weights
        for bus_num in range(self.num_of_buses):
            starting_station = np.random.choice(list(range(1, self.num_of_stations + 1)), p=weights_stations)
            starting_stations.append(starting_station)
        self.starting_stations = starting_stations

    def find_starting_station(self):
        """
        Find a starting station - the station is chosen by a probability corresponding to the nodes' weights
        """
        random.seed(self.seed)
        weights_stations = list(nx.get_node_attributes(self.network, 'weight').values())
        weights_stations = list(np.array(weights_stations) / sum(weights_stations))
        starting_station = np.random.choice(list(range(1, self.num_of_stations + 1)), p=weights_stations)
        return starting_station

    def compute_neighbor_score(self, stations):
        """
        Calculate the weighted score that will cost us to include a path from a node to another node -
        The score is the distance and travel time weights (weights of the edge) and the to station's frequency weight.

        param stations: a list of two stations - from station and to station
        :return: the weighted score between from station and to station
        """
        user_weights = self.user_weights
        # get the weights (station weight is maximum so multiply by -1)
        station_weight = -1 * list(nx.get_node_attributes(self.network, 'weight').values())[stations[1] - 1]
        dist_weight = self.network[stations[0]][stations[1]]["weight"]['Distance']
        time_weight = self.network[stations[0]][stations[1]]["weight"]['Time']
        # weight the weights by the objective function weights
        score = (user_weights[0] * dist_weight) + (user_weights[1] * time_weight) + (user_weights[2] * station_weight)
        return score, dist_weight, time_weight, station_weight

    def find_initial_paths(self, path_length=0):
        """
        For every bus find its path with maximum number of stations in each path -
        choose next station with a probability of how good it is to make it the next station in the path.
        A specific bus cannot visit the same station twice.
        """
        random.seed(self.seed)
        if path_length == 0:
            path_length = self.max_num_of_stations
        for bus in range(1, self.num_of_buses + 1):     # Loop through every bus
            path = [self.starting_stations[bus - 1]]    # Assign starting station of the bus
            optional_stations = list(self.network.neighbors(self.starting_stations[bus - 1]))
            for i in range(path_length - 1):      # Find all stations in the current path
                # Compute the score to go to every possible station
                score_list = [self.compute_neighbor_score([int(path[-1]), int(station)])[0] for station in optional_stations]
                score_list_normalized = normalize_scores(score_list)    # normalize the scores
                # next_station = np.argmin(np.array(score_list))
                next_station = np.random.choice(optional_stations, p=score_list_normalized)     # Choose next station
                path.append(int(next_station))
                optional_stations.remove(next_station)  # remove the chosen station from the optional next stations
            self.paths_dict[bus] = path

    def change_partial_path_of_specific_bus(self, original_path_dict, bus_path_num, stations_to_add, start_station_index):
        """
        Change a given bus path starting from a given station and adding a given number of stations

        :param original_path_dict: the original paths
        :param bus_path_num: bus path number to partially change
        :param stations_to_add: number of stations to add to the path
        :param start_station_index: the station to change path from it
        :return: the new paths after the partial change
        """
        random.seed(self.seed)
        path = original_path_dict[bus_path_num][:start_station_index]   # Keep the stations before the starting station
        optional_stations = list(self.network.nodes)
        [optional_stations.remove(station) for station in path]     # We can't visit stations already in the path
        for i in range(stations_to_add):
            # Score all possible stations to go to
            score_list = [self.compute_neighbor_score([int(path[-1]), int(station)])[0] for station in optional_stations]
            score_list_normalized = normalize_scores(score_list)
            # next_station = np.argmin(np.array(score_list))
            next_station = np.random.choice(optional_stations, p=score_list_normalized)     # choose the next station
            path.append(int(next_station))
            optional_stations.remove(next_station)  # Remove the chosen station from the optional stations
        new_path_dict = copy.deepcopy(original_path_dict)
        new_path_dict[bus_path_num] = path  # Set the new changed path
        return new_path_dict

    def change_all_path_of_specific_bus(self, original_path_dict, bus_path_num, new_path_length):
        """
        Change a given bus path - the path will have 'new_path_length' stations

        :param original_path_dict: the original paths
        :param bus_path_num: bus path number to change
        :param new_path_length: number of stations in the new path
        :return: the new paths after the complete path change
        """
        random.seed(self.seed)
        path = []
        first_station = self.find_starting_station()    # Choose the starting station for the path
        path.append(first_station)
        optional_stations = list(self.network.neighbors(first_station))
        for i in range(new_path_length - 1):
            # Compute score of all possible next stations
            score_list = [self.compute_neighbor_score([int(path[-1]), int(station)])[0] for station in optional_stations]
            score_list_normalized = normalize_scores(score_list)
            # next_station = np.argmin(np.array(score_list))
            next_station = np.random.choice(optional_stations, p=score_list_normalized)     # Choose next station
            path.append(int(next_station))
            optional_stations.remove(next_station)  # We can't visit an already visited station
        new_path_dict = copy.deepcopy(original_path_dict)
        new_path_dict[bus_path_num] = path  # Set the new complete path
        return new_path_dict

    def check_all_stations_visited(self):
        """
        Check the frequency of stations visited in all paths and check which stations weren't visited at all.
        """
        stations_visited = []
        for path in range(len(self.paths_dict.values())):
            for s in range(len(list(self.paths_dict.values())[path])):
                stations_visited.append(list(self.paths_dict.values())[path][s])    # Save current station to visited
        visited_stations = np.unique(stations_visited)  # get all visited stations
        # Calculate stations visits frequency
        stations_freq = pd.DataFrame(np.array(stations_visited).reshape(-1, 1)).value_counts().sort_values()
        unvisited_stations = []
        for station in range(1, self.num_of_stations + 1):
            if station not in visited_stations:
                # save all stations not in visited stations to unvisited stations
                unvisited_stations.append(station)
        self.unvisited_stations = unvisited_stations
        self.stations_frequency = stations_freq

    def fix_unvisited_stations(self):
        """
        All stations that weren't visited at all will be embedded into one of the paths -
        stations with the lowest frequency (higher than 1) will be removed (they are worse stations).
        """
        random.seed(self.seed)
        stations_freq = copy.deepcopy(self.stations_frequency)  # Visited stations' frequency
        # Loop through all unvisited stations
        for station in self.unvisited_stations:
            stations_freq = stations_freq[stations_freq > 1]    # Get all stations that were visited more than once
            if len(stations_freq) == 0:
                # If no such stations exist - add the unvisited station randomly to the end of a path
                bus_list = list(range(1, self.num_of_buses + 1))
                bus_number = np.random.choice(bus_list)
                # Make sure we don't choose a path that is already at its maximum size
                while len(self.paths_dict[bus_number]) == self.max_num_of_stations:
                    bus_list.remove(bus_number)
                    bus_number = np.random.choice(bus_list)
                print("No stations to replace - adding the unvisited station randomly")
                self.paths_dict[bus_number].append(station)
            else:
                replace_station = stations_freq.index[0][0]     # station to remove
                stations_freq[replace_station] -= 1     # Update visiting frequency
                max_score = -999999999
                chosen_bus = []
                chosen_station_index = []
                for bus_number in self.paths_dict.keys():
                    bus_path = self.paths_dict[bus_number]
                    # Look if the removed station is in the bus path
                    if replace_station in bus_path:
                        station_index = bus_path.index(replace_station)
                        # Calculate the new score of the path after replacing the removed station with the unvisited one
                        if station_index == 0:
                            to_station = bus_path[1]
                            score = self.compute_neighbor_score([to_station, replace_station])[0]
                        elif station_index == len(bus_path) - 1:
                            from_station = bus_path[-2]
                            score = self.compute_neighbor_score([from_station, replace_station])[0]
                        else:
                            from_station = bus_path[station_index - 1]
                            to_station = bus_path[station_index + 1]
                            score = self.compute_neighbor_score([from_station, replace_station])[0]
                            score += self.compute_neighbor_score([to_station, replace_station])[0]
                    else:
                        score = -999999999
                    if score > max_score:   # Choose to replace the station that gives the best score
                        max_score = score
                        chosen_bus = bus_number
                        chosen_station_index = station_index
                self.paths_dict[chosen_bus][chosen_station_index] = station     # update path
            self.check_all_stations_visited()   # Update frequency and visited stations

    def create_network_from_paths_dict(self):
        """
        Create a new network with the same weights as the original network,
        but includes only edges that appear in the bus paths.
        """
        net = nx.Graph()
        # Add all nodes with their original weights
        for node in list(self.network.nodes(data='weight')):
            net.add_node(node[0], weight=node[1])
        # Add edges that appear in one of the paths with their original weight
        for bus in self.paths_dict.keys():
            bus_path = self.paths_dict[bus]
            for station in range(len(bus_path) - 1):
                net.add_edge(bus_path[station], bus_path[station + 1])
        self.current_path_network = net

    def check_all_stations_connections(self):
        """
        Check how many pairs of stations cannot be reached from one another in the found network.
        This is an objective function criteria that modulates the connectivity of the network.
        """
        stations_list = list(self.current_path_network.nodes)
        all_combinations = list(combinations(stations_list, 2))     # All possible stations (nodes) pairs
        stations_without_path = []
        for combination in all_combinations:
            # If there is no path between the two stations - add the pait to stations without path
            if not nx.has_path(self.current_path_network, combination[0], combination[1]):
                stations_without_path.append(combination)
        self.stations_without_path = stations_without_path

    def score_network(self):
        """
        Calculate the score of a given network - sum all the travel time and distance weights found on the edges,
        the nodes weights (multiplied by minus 1 and the number of times they were visited) and the connectivity score.
        Each weight is normalized by the objective function weights.
        """
        # Zero the scores of the network
        self.distance_score = 0
        self.travel_time_score = 0
        self.stations_score = 0
        # Connectivity score
        score = self.user_weights[3] * len(self.stations_without_path) / self.len_all_stations_combinations
        self.connectivity_score = score
        for bus in self.paths_dict.keys():
            # Sum and normalize all weights
            bus_path = self.paths_dict[bus]
            # Weight of the first node in the path
            score += self.user_weights[2] * -1 * list(nx.get_node_attributes(self.network, 'weight').values())[bus_path[0] - 1]
            for station in range(len(bus_path) - 1):
                temp_score, dis_weight, travel_weight, sta_weight = self.compute_neighbor_score([bus_path[station], bus_path[station + 1]])
                score += temp_score
                self.distance_score += dis_weight
                self.travel_time_score += travel_weight
                self.stations_score += sta_weight
        self.network_score = score

