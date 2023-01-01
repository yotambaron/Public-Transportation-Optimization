# Public-Transportation-Optimization
Computational Intelligence - public transportation planning optimization using nature-based algorithms.

This project explores three different nature-based search algorithms with the task of planning a multi-objective public transportation traffic network, 
withstanding several constraints.

The constraints:
1. All stations must be visited at least once.
2. Every bus must not revisit a station in its route.
3. A passanger can reach all stations from every station.

The algorithms:
1. Genetic Algorithm (GA) - an iterative search algorithm based on principles from human genetics.
2. Non-dominated Sorting-based multi-objective artificial bee colony (NSABC) - a search algorithm based on the behaviour of a bee colony.
3. Simulated Annealing (SA) - a search algorithm using a technique that helps escaping a local minima\maxima.

The objectives:
1. Minimize the total distance driven (edge weight 1).
2. Minimize the total time driven (edge weight 2).
3. Maximize arrivals to popular stations (node weight).

Networks of growing size were tested and weights were randomly assigned to each edge (2 different weights) and each node.
Code files are provided together with the results of all algorithms.

"Computational Intelligence Report" word file is the summarizing report of the entire project.
