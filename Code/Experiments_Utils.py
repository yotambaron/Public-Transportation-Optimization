import numpy as np
import pandas as pd


def check_dominance(sol1, sol2):
    """
    Check if sol1 dominates sol2 - if it is at least as good as sol 2 in all criteria

    :param sol1: the dominant solution
    :param sol2: the dominated solution
    :return: True if sol1 dominates sol 2 and False otherwise
    """
    if (sol1.distance_score <= sol2.distance_score and sol1.travel_time_score <= sol2.travel_time_score
        and sol1.stations_score <= sol2.stations_score and sol1.connectivity_score < sol2.connectivity_score) or (
            sol1.distance_score <= sol2.distance_score and sol1.travel_time_score <= sol2.travel_time_score
            and sol1.stations_score < sol2.stations_score and sol1.connectivity_score <= sol2.connectivity_score) or (
            sol1.distance_score <= sol2.distance_score and sol1.travel_time_score < sol2.travel_time_score
            and sol1.stations_score <= sol2.stations_score and sol1.connectivity_score <= sol2.connectivity_score) or (
            sol1.distance_score < sol2.distance_score and sol1.travel_time_score <= sol2.travel_time_score
            and sol1.stations_score <= sol2.stations_score and sol1.connectivity_score <= sol2.connectivity_score):
        return True
    else:
        return False


def clean_archive(archive):
    """
    Run through every solution in the archive and delete any dominated solution.
    """
    sol1_index = 0
    while sol1_index < len(archive) - 1:
        sol2_index = sol1_index + 1
        while sol2_index < len(archive):
            # If sol1 dominates sol2 in the archive - delete sol2
            if check_dominance(archive[sol1_index], archive[sol2_index]):
                archive.pop(sol2_index)
            # If sol2 dominates sol1 in the archive - delete sol1
            elif check_dominance(archive[sol2_index], archive[sol1_index]):
                archive.pop(sol1_index)
                sol1_index -= 1
                break
            else:
                sol2_index += 1
        sol1_index += 1
    return archive


def normalize_scores(scores):
    """
    Normalize the scores of the employed bees' solutions - high scores are worse

    param scores: scores of the current employed bees' solutions
    :return: normalized scores
    """
    scores_normalized = np.array(scores) - max(scores) - 0.01
    scores_normalized = -1 * scores_normalized
    scores_normalized = list(scores_normalized / sum(scores_normalized))
    return scores_normalized


def initialize_solution(net):
    """
    Given a network with nodes - initialize it:
    - Find starting stations
    - Find initial paths
    - Make sure all stations are visited
    - Create a network including only the initialized paths
    - Score the new network
    """
    net.find_starting_stations()
    net.find_initial_paths()
    net.check_all_stations_visited()
    net.fix_unvisited_stations()
    net.create_network_from_paths_dict()
    net.check_all_stations_connections()
    net.score_network()
    return net


def score_archive_dominance(archive1, archive2):
    """
    Goes through all the pairs of solutions in the two archives and counts separately the number of
    dominating solutions in archive 1 over archive 2 and the other way around.

    :param archive1: the first archive
    :param archive2: the second archive
    :return: the number of dominating solutions from archive 1 over 2
             and the number of dominating solutions from archive 2 over 1
    """
    score_archive1 = 0
    score_archive2 = 0
    for sol1 in archive1:
        for sol2 in archive2:
            if check_dominance(sol1, sol2):
                score_archive1 += 1
            elif check_dominance(sol2, sol1):
                score_archive2 += 2
    return score_archive1, score_archive2


def best_scores_extraction(results, weights_combination):
    """
    Find and save the best result of the algorithm in every weight combination
    (the best score out of the algorithm's parameters).

    :param results: all results with all weights combinations and algorithm's parameters
    :param weights_combination: all combinations of the objective function's weights
    :return: a dataframe with the best results under each combination of weights
    """
    # The new best scores dataframe
    best_scores = pd.DataFrame(np.zeros([len(weights_combination), len(results.columns)]), columns=results.columns)
    row_ind = 0
    # Loop through all weights combination
    for weights in weights_combination:
        distance_weight = weights[0]
        time_weight = weights[1]
        frequency_weight = weights[2]
        connectivity_weight = weights[3]
        # Results of the current weights combination
        temp_results = results[results['Weight_distance'] == distance_weight]
        temp_results = temp_results[temp_results['Weight_time'] == time_weight]
        temp_results = temp_results[temp_results['Weight_frequency'] == frequency_weight]
        temp_results = temp_results[temp_results['Weight_connectivity'] == connectivity_weight]
        # The index of the best score of the current combination of weights
        min_index = np.argmin(np.array(temp_results['Score']))
        # Save the best current score in the new dataframe
        best_scores.iloc[row_ind, :] = temp_results.iloc[min_index, :]
        row_ind += 1
    return best_scores


def compute_dominating_and_dominated_solutions(archive1, archive2):
    """
    Goes through all the solutions in archive 1 and archive 2
    and counts how many are dominated and dominating in each archive.

    :param archive1: the first archive with solutions to check
    :param archive2: the second archive with solutions to check
    :return: the number of dominating and dominated solutions in archive 1 over archive 2 and
    the number of dominating and dominated solutions in archive 2 over archive 1
    """
    dominant1 = 0
    dominated1 = 0
    dominant2 = 0
    dominated2 = 0
    for sol1 in archive1:
        for sol2 in archive2:
            if check_dominance(sol1, sol2):
                # If the solution from archive 1 dominates the solution from archive 2 -
                # increase by one dominant1 and dominated2
                dominant1 += 1
                dominated2 += 1
            elif check_dominance(sol2, sol1):
                # If the solution from archive 2 dominates the solution from archive 1 -
                # increase by one dominant2 and dominated1
                dominant2 += 1
                dominated1 += 1
    return dominant1, dominated1, dominant2, dominated2


def compute_dominating_and_dominated_solutions_three_algos(archive1, archive2, archive3):
    """
    Goes through all the solutions in archive 1 and archive 2 archive 3
    and counts how many are dominated and dominating in each archive.

    :param archive1: the first archive with solutions to check
    :param archive2: the second archive with solutions to check
    :param archive3: the third archive with solutions to check
    :return: the number of dominating and dominated solutions in archive 1, archive 2 and archive 3
    """
    dominant1 = 0
    dominated1 = 0
    dominant2 = 0
    dominated2 = 0
    dominant3 = 0
    dominated3 = 0
    for sol1 in archive1:
        for sol2 in archive2:
            for sol3 in archive3:
                if check_dominance(sol1, sol2):
                    # If the solution from archive 1 dominates the solution from archive 2 -
                    # increase by one dominant1 and dominated2
                    dominant1 += 1
                    dominated2 += 1
                elif check_dominance(sol2, sol1):
                    # If the solution from archive 2 dominates the solution from archive 1 -
                    # increase by one dominant2 and dominated1
                    dominant2 += 1
                    dominated1 += 1
                if check_dominance(sol1, sol3):
                    # If the solution from archive 1 dominates the solution from archive 3 -
                    # increase by one dominant1 and dominated3
                    dominant1 += 1
                    dominated3 += 1
                elif check_dominance(sol3, sol1):
                    # If the solution from archive 3 dominates the solution from archive 1 -
                    # increase by one dominant3 and dominated1
                    dominant3 += 1
                    dominated1 += 1
                if check_dominance(sol2, sol3):
                    # If the solution from archive 2 dominates the solution from archive 3 -
                    # increase by one dominant2 and dominated3
                    dominant2 += 1
                    dominated3 += 1
                elif check_dominance(sol3, sol2):
                    # If the solution from archive 3 dominates the solution from archive 2 -
                    # increase by one dominant3 and dominated2
                    dominant3 += 1
                    dominated2 += 1
    return dominant1, dominated1, dominant2, dominated2, dominant3, dominated3


def compute_results(algorithm1, name1, algorithm2, name2, algorithm3, name3):
    """
    Calculate the results of the two given algorithms.

    :param algorithm1: the first algorithm's permutations to score
    :param name1: the name of the first algorithm
    :param algorithm2: the second algorithm's permutations to score
    :param name2: the name of the second algorithm
    :param algorithm3: the third algorithm's permutations to score
    :param name3: the name of the third algorithm
    :return: a dataframe with the results of all three algorithms - avg, min, max run time,
    avg best score, avg dominated and dominating solutions.
    """
    columns = ['Algorithm', 'Avg_Best_score', 'Dominating_Solutions', 'Dominated_Solutions', 'Avg_RunTime',
               'Min_RunTime', 'Max_RunTime']
    All_results = pd.DataFrame(np.zeros([3, len(columns)]), columns=columns)
    run_time_array1 = []
    run_time_array2 = []
    run_time_array3 = []
    avg_score1 = 0
    avg_score2 = 0
    avg_score3 = 0
    avg_dominant1 = 0
    avg_dominant2 = 0
    avg_dominant3 = 0
    avg_dominated1 = 0
    avg_dominated2 = 0
    avg_dominated3 = 0
    for permutation in range(len(algorithm1)):
        run_time_array1.append(algorithm1[permutation].total_run_time)
        run_time_array2.append(algorithm2[permutation].total_run_time)
        run_time_array3.append(algorithm3[permutation].total_run_time)
        avg_score1 += algorithm1[permutation].best_solution.network_score
        avg_score2 += algorithm2[permutation].best_solution.network_score
        avg_score3 += algorithm3[permutation].best_solution.network_score
        dominant1, dominated1, dominant2, dominated2, dominant3, dominated3 = \
            compute_dominating_and_dominated_solutions_three_algos(
                algorithm1[permutation].archive, algorithm2[permutation].archive, algorithm3[permutation].archive)
        avg_dominant1 += dominant1
        avg_dominant2 += dominant2
        avg_dominant3 += dominant3
        avg_dominated1 += dominated1
        avg_dominated2 += dominated2
        avg_dominated3 += dominated3
        print("Finished scoring permutation: ", permutation)

    avg_runtime1 = np.mean(run_time_array1)
    avg_runtime2 = np.mean(run_time_array2)
    avg_runtime3 = np.mean(run_time_array3)
    min_runtime1 = np.min(run_time_array1)
    min_runtime2 = np.min(run_time_array2)
    min_runtime3 = np.min(run_time_array3)
    max_runtime1 = np.max(run_time_array1)
    max_runtime2 = np.max(run_time_array2)
    max_runtime3 = np.max(run_time_array3)
    avg_score1 = avg_score1 / len(algorithm1)
    avg_score2 = avg_score2 / len(algorithm2)
    avg_score3 = avg_score3 / len(algorithm3)
    avg_dominant1 = avg_dominant1 / len(algorithm1)
    avg_dominant2 = avg_dominant2 / len(algorithm2)
    avg_dominant3 = avg_dominant3 / len(algorithm3)
    avg_dominated1 = avg_dominated1 / len(algorithm1)
    avg_dominated2 = avg_dominated2 / len(algorithm2)
    avg_dominated3 = avg_dominated3 / len(algorithm3)

    results1 = [name1, avg_score1, avg_dominant1, avg_dominated1, avg_runtime1, min_runtime1, max_runtime1]
    results2 = [name2, avg_score2, avg_dominant2, avg_dominated2, avg_runtime2, min_runtime2, max_runtime2]
    results3 = [name3, avg_score3, avg_dominant3, avg_dominated3, avg_runtime3, min_runtime3, max_runtime3]
    All_results.iloc[0, :] = results1
    All_results.iloc[1, :] = results2
    All_results.iloc[2, :] = results3
    return All_results


def calculate_parameters_results(results_array, parameter_name, parameter_options):
    """
    Aggregate the results of the wanted parameter of the algorithm over different runs.

    :param results_array: dataframes with the results from all the runs.
    :param parameter_name: the name of the parameter to be aggregated.
    :param parameter_options: the options of the parameter's value to aggregate on.
    :return: a dataframe with the aggregated results over the runs for each value of the parameter -
    avg score and average run time.
    """
    cols = ['Parameter_Name', 'Parameter_Value', 'Average_Score', 'Average_Run_time']
    parameter_results = pd.DataFrame(np.zeros([len(parameter_options), 4]), columns=cols)
    for ind, parameter in enumerate(parameter_options):
        temp_score = 0
        temp_run_time = 0
        length = 0
        for result in results_array:
            temp_result = result[result[parameter_name] == parameter]
            temp_score += np.sum(temp_result['Score'])
            temp_run_time += np.sum(temp_result['Total_run_time'])
            length += len(temp_result)
        average_score = temp_score / length
        average_run_time = temp_run_time / length
        parameter_results.iloc[ind, :] = [parameter_name, parameter, average_score, average_run_time]
    return parameter_results


