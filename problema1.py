import numpy as np

# Helper functions for the genetic algorithm

def initialize_population(pop_size, dim):
    """Initialize a population with random values between [0, 1]."""
    return np.random.rand(pop_size, dim)

def fitness(individual, obj_func, constraints):
    """Calculate the fitness of an individual based on objective function and constraints."""
    if all(constraint(individual) for constraint in constraints):
        return obj_func(individual)
    else:
        return float('-inf')

def selection(population, fitness_scores):
    """Select parents based on fitness scores using tournament selection."""
    k = 3  # tournament size
    selected_indices = np.argsort(np.random.choice(fitness_scores, k))[-2:]
    return population[selected_indices[0]], population[selected_indices[1]]

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    child = np.zeros(parent1.shape)
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(child, mutation_rate=0.05):
    """Introduce mutations in a child."""
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.rand()
    return child

# Problem definitions 
def objective_A(x):
    return 15*x[0] + 30*x[1] + 4*x[0]*x[1] - 2*x[0]**2 - 4*x[1]**2

constraints_A = [
    lambda x: x[0] + 2*x[1] <= 30,
    lambda x: x[0] >= 0,
    lambda x: x[1] >= 0
]

def objective_B(x):
    return 3*x[0] + 5*x[1]

constraints_B = [
    lambda x: 3*x[0] + 2*x[1] <= 18,
    lambda x: x[0] >= 0,
    lambda x: x[1] >= 0
]

def objective_C(x):
    return 5*x[0] - x[0]**2 + 8*x[1] - 2*x[1]**2

constraints_C = [
    lambda x: 3*x[0] + 2*x[1] <= 6,
    lambda x: x[0] >= 0,
    lambda x: x[1] >= 0
]


def genetic_algorithm_v2(obj_func, constraints, dim, pop_size=100, max_gen=1000, mutation_rate=0.05):
    """Implement a genetic algorithm to maximize obj_func."""
    population = initialize_population(pop_size, dim)
    best_fitness = float('-inf')  # Initialize best fitness as negative infinity
    best_individual = None

    for generation in range(max_gen):
        fitness_scores = [fitness(ind, obj_func, constraints) for ind in population]
        new_population = []
        
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1, mutation_rate), mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
        
        # Update best fitness and best individual
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)]
    
    return best_individual, best_fitness




# Apply the genetic algorithm to solve the problems
solution_A, max_value_A = genetic_algorithm_v2(objective_A, constraints_A, dim=2)
solution_B, max_value_B = genetic_algorithm_v2(objective_B, constraints_B, dim=2)
solution_C, max_value_C = genetic_algorithm_v2(objective_C, constraints_C, dim=2)

print("1a - Solution:", solution_A)
print("   - Max Value:", max_value_A)

print("---------------------------------")

print("1b - Solution:", solution_B)
print("   - Max Value:", max_value_B)

print("---------------------------------")

print("1c - Solution:", solution_C)
print("   - Max Value:", max_value_C)
