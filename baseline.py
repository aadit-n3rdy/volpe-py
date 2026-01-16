import random
import numpy as np
import tsplib95 as tsplib
from deap import base, creator, tools, algorithms
import multiprocessing
import os
import csv
import time

# --- 1. Configuration & Constants ---
# Problem specific
PROBLEM_FILE = 'att532.tsp'
NDIM = 532  # Problem dimension

# GA Parameters (Ported from original main.py)
MUTATION_RATE = 0.4
SWAP_PROB = 0.2
REVERSAL_PROB = 2/532
CXPROB = 0.9
BASE_POPULATION_SIZE = 4000

# Load the problem
# Note: This needs to be loadable by worker processes. 
# Ideally, tsplib loads are fast enough to happen on import.
try:
    problem = tsplib.load_problem(PROBLEM_FILE)
except Exception as e:
    print(f"Error loading {PROBLEM_FILE}: {e}")
    problem = None

# --- 2. DEAP Setup ---

# Fitness is minimization (TSP distance)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# --- 3. Parallelism with SCOOP ---
# --- 4. Initialization ---
# Original code: np.random.permutation(NDIM) + 1  (1-based indexing)
toolbox.register("indices", random.sample, range(0, NDIM), NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 5. Evaluation ---
def eval_tsp(individual):
    """
    Evaluates the total distance of the tour.
    tsplib95 expects a list of tours.
    """
    if problem:
        # trace_tours returns a list of distances, we take the first one
        distance = problem.trace_tours([[i+1 for i in individual]])[0]
        return (distance,)
    return (float('inf'),)

toolbox.register("evaluate", eval_tsp)

# --- 6. Custom Operators (Matching original main.py) ---

def custom_mutate(individual):
    """
    Custom mutation replicating the logic in original main.py:
    1. Binomial number of Swaps
    2. Poisson number of Reversals (2-optish)
    """
    # 1. Swaps
    # np.random.binomial(n, p)
    n_swaps = np.random.binomial(NDIM, SWAP_PROB)
    for _ in range(n_swaps):
        i1 = random.randint(0, NDIM - 1)
        i2 = random.randint(0, NDIM - 1)
        individual[i1], individual[i2] = individual[i2], individual[i1]

    # 2. Reversals
    n_reversals = np.random.poisson(lam=REVERSAL_PROB * NDIM)
    for _ in range(n_reversals):
        i1 = random.randint(0, NDIM - 1)
        i2 = random.randint(0, NDIM - 1)
        if i1 > i2:
            i1, i2 = i2, i1
        # Reverse the slice
        individual[i1:i2] = individual[i1:i2][::-1]
    return (individual,)

# Register Operators
toolbox.register("mate", tools.cxOrdered)  # cxOrdered is the standard implementation of the original 'cross_raw' logic
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)  # Matches logic: choices=3, min(choices)

# --- 7. Main Execution ---
def main():
    print(f"Starting Unlimited Distributed TSP Optimization on {PROBLEM_FILE}")

    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)
    
    # 1. Initialize CSV file and write header if it doesn't exist
    output_filename = "BASELINE.csv"
    if not os.path.exists(output_filename):
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_Seconds", "Best_Fitness"])

    # 2. Initialize Population and Hall of Fame
    start_time = time.time()
    last_log_time = start_time
    pop = toolbox.population(n=BASE_POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    # 3. Initial Evaluation
    print("Performing initial evaluation...")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    print("Evolution started. Press Ctrl+C to stop.")
    
    # 4. Unlimited Generational Loop
    generation = 0
    try:
        while True:
            # Standard DEAP generational step (varAnd logic)
            offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPROB, mutpb=MUTATION_RATE)
            
            # Evaluate offspring with SCOOP
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update population and Hall of Fame
            pop = toolbox.select(offspring + pop, k=BASE_POPULATION_SIZE)
            hof.update(pop)
            
            # 5. Time-based Logging (Every 5 seconds)
            current_time = time.time()
            if current_time - last_log_time >= 5:
                elapsed = current_time - start_time
                best_fit = hof[0].fitness.values[0]
                
                with open(output_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([round(elapsed, 2), best_fit])
                
                print(f"Gen {generation} | Time: {int(elapsed)}s | Best: {best_fit}")
                last_log_time = current_time
            
            generation += 1

    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
        print(f"Final Best Fitness: {hof[0].fitness.values[0]}")
        print(f"Final Best Genotype: {np.array(hof[0])+1}")

if __name__ == "__main__":
    main()
