import random
import math

# ==============================================================================
# --- 1. CORE GENETIC ALGORITHM COMPONENTS (REUSABLE FOR ALL PARTS) ---
# ==============================================================================

def create_individual(genome_length, variable_range):
    """
    Creates a single random individual (a potential solution).
    
    Args:
        genome_length (int): The number of variables in the solution (e.g., 2 for [x, y]).
        variable_range (tuple): A tuple (min, max) for the random values.
        
    Returns:
        list: A list of random floats representing the genome.
    """
    return [random.uniform(variable_range[0], variable_range[1]) for _ in range(genome_length)]

def tournament_selection(population, fitnesses, k=5):
    """
    Selects a parent from the population using tournament selection.
    
    Args:
        population (list): The list of all individuals.
        fitnesses (list): The list of fitness scores for each individual.
        k (int): The number of individuals to compete in the tournament.
        
    Returns:
        list: The winning individual (genome) from the tournament.
    """
    # Select k random individuals from the population
    tournament_indices = random.sample(range(len(population)), k)
    
    # Find the individual with the best fitness among the selected ones
    best_index_in_tournament = -1
    best_fitness_in_tournament = -1
    
    for index in tournament_indices:
        if fitnesses[index] > best_fitness_in_tournament:
            best_fitness_in_tournament = fitnesses[index]
            best_index_in_tournament = index
            
    return population[best_index_in_tournament]

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Performs arithmetic crossover between two parents to produce two children.
    
    Args:
        parent1 (list): The first parent's genome.
        parent2 (list): The second parent's genome.
        crossover_rate (float): The probability that crossover will occur.
        
    Returns:
        tuple: A tuple containing two new children (genomes).
    """
    if random.random() < crossover_rate:
        # Create a random blending factor
        alpha = random.random()
        child1 = [p1 * alpha + p2 * (1 - alpha) for p1, p2 in zip(parent1, parent2)]
        child2 = [p1 * (1 - alpha) + p2 * alpha for p1, p2 in zip(parent1, parent2)]
        return child1, child2
    else:
        # If crossover doesn't happen, return the parents unchanged
        return parent1, parent2

def mutate(individual, mutation_rate=0.1, mutation_strength=0.5):
    """
    Mutates an individual by adding a small random value to its genes.
    
    Args:
        individual (list): The genome to mutate.
        mutation_rate (float): The probability of mutation for each gene.
        mutation_strength (float): The maximum magnitude of the random change.
        
    Returns:
        list: The mutated genome.
    """
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Add a small random value
            change = random.uniform(-mutation_strength, mutation_strength)
            mutated_individual[i] += change
    return mutated_individual

# ==============================================================================
# --- 2. PROBLEM-SPECIFIC FITNESS FUNCTIONS ---
# ==============================================================================

# --- Fitness Function for Part 1 ---
def fitness_function_part1(genome):
    """
    Calculates fitness for the 2x2 linear system.
    Equations:
        x + 2y = 4
        4x + 4y = 12
    """
    x, y = genome[0], genome[1]
    
    err1 = (x + 2*y) - 4
    err2 = (4*x + 4*y) - 12
    
    total_error = err1**2 + err2**2
    return 1 / (total_error + 1e-9) # Add epsilon to avoid division by zero

# --- Fitness Function for Part 2 ---
def fitness_function_part2(genome):
    """
    Calculates fitness for the 3x3 non-linear system.
    Equations:
        6x - 2y + 8z = 20
        y + 8xz = -1
        12z/x + 1.5y = 6
    """
    x, y, z = genome[0], genome[1], genome[2]
    
    # Add a large penalty if x is close to zero to prevent division by zero
    if abs(x) < 1e-6:
        return 1e-9 # Return a very low fitness
    
    err1 = (6*x - 2*y + 8*z) - 20
    err2 = (y + 8*x*z) + 1
    err3 = ((12*z) / x + 1.5*y) - 6
    
    total_error = err1**2 + err2**2 + err3**2
    return 1 / (total_error + 1e-9)

# --- Fitness Function for Part 3 ---
def fitness_function_part3(genome):
    """
    Calculates fitness for the 4x4 linear system.
    Equations:
        (1/15)x - 2y - 15z - (4/5)t = 3
        -2.5x - 2.25y + 12z - t = 17
        -13x + 0.3y - 6z - 0.4t = 17
        0.5x + 2y + 1.75z + (4/3)t = -9
    """
    x, y, z, t = genome[0], genome[1], genome[2], genome[3]
    
    err1 = (1/15)*x - 2*y - 15*z - (4/5)*t - 3
    err2 = -2.5*x - 2.25*y + 12*z - t - 17
    err3 = -13*x + 0.3*y - 6*z - 0.4*t - 17
    err4 = 0.5*x + 2*y + 1.75*z + (4/3)*t + 9
    
    total_error = err1**2 + err2**2 + err3**2 + err4**2
    return 1 / (total_error + 1e-9)

# ==============================================================================
# --- 3. MAIN GA SOLVER ENGINE ---
# ==============================================================================

def run_genetic_algorithm(part_name, fitness_func, genome_length, generations, pop_size, variable_range, correct_solution=None):
    """
    A general-purpose function to run the genetic algorithm for a specific problem.
    """
    print(f"\n{'='*20} Running GA for {part_name} {'='*20}")
    
    # 1. Initialization
    population = [create_individual(genome_length, variable_range) for _ in range(pop_size)]
    
    best_solution_so_far = None
    best_fitness_so_far = -1

    # 2. Main Loop
    for generation in range(generations):
        # Evaluate fitness of the entire population
        fitnesses = [fitness_func(ind) for ind in population]
        
        # Find the best individual in the current generation
        current_best_index = fitnesses.index(max(fitnesses))
        
        # Update the overall best solution found
        if fitnesses[current_best_index] > best_fitness_so_far:
            best_fitness_so_far = fitnesses[current_best_index]
            best_solution_so_far = population[current_best_index]

        # 3. Create the next generation
        next_population = []
        
        # Elitism: The best individual automatically moves to the next generation
        next_population.append(best_solution_so_far)
        
        # Fill the rest of the new population
        while len(next_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            next_population.append(mutate(child1))
            if len(next_population) < pop_size:
                next_population.append(mutate(child2))

        population = next_population
        
        # Print progress report
        if generation % (generations // 10) == 0:
            error = 1/best_fitness_so_far if best_fitness_so_far > 0 else float('inf')
            print(f"Generation {generation:4d}: Best Fitness = {best_fitness_so_far:,.2f}, Error = {error:.8f}")

    # 4. Final Result
    final_error = 1/best_fitness_so_far if best_fitness_so_far > 0 else float('inf')
    print(f"\n--- Final Result for {part_name} ---")
    print(f"GA finished after {generations} generations.")
    
    var_names = ['x', 'y', 'z', 't']
    solution_str = ", ".join([f"{var_names[i]} = {val:.4f}" for i, val in enumerate(best_solution_so_far)])
    print(f"Best solution found: {solution_str}")
    
    if correct_solution:
        correct_solution_str = ", ".join([f"{var_names[i]} = {val:.4f}" for i, val in enumerate(correct_solution)])
        print(f"Correct solution is: {correct_solution_str}")
        
    print(f"Final Error (Sum of Squares): {final_error:.8f}")
    print("=" * (42 + len(part_name)))


# ==============================================================================
# --- 4. EXECUTION BLOCK ---
# ==============================================================================

if __name__ == '__main__':
    # --- Parameters for Part 1 ---
    run_genetic_algorithm(
        part_name="Part 1: 2x2 Linear System",
        fitness_func=fitness_function_part1,
        genome_length=2,
        generations=200,
        pop_size=100,
        variable_range=(-100, 100),
        correct_solution=[2.0, 1.0]
    )
    """
     #--- Parameters for Part 2 ---
     #To run this part, uncomment the block below
    run_genetic_algorithm(
         part_name="Part 2: 3x3 Non-Linear System",
         fitness_func=fitness_function_part2,
         genome_length=3,
         generations=500,
         pop_size=200,
         variable_range=(-20, 20),
         correct_solution=[2/3, -5.0, 3/4]
     )"""

    # --- Parameters for Part 3 ---
    # To run this part, uncomment the block below
    """run_genetic_algorithm(
         part_name="Part 3: 4x4 Linear System",
         fitness_func=fitness_function_part3,
         genome_length=4,
         generations=1000,
         pop_size=400,
         variable_range=(-50, 50),
         correct_solution=[-3/2, -7/2, 1/3, -11/8]
     )"""