import numpy as np

def createPopulation(pop_size, dim):
    """Create the population with values randomly between [0,1]"""
    return np.random.rand(pop_size, dim)

def fitness(individual, obj_func, constraints):
    """Calculate fitness based on the objective and constraints"""
    violation_penalty = 1000
    total_violation = sum([max(0, constraint(individual)) for constraint in constraints])
    return obj_func(individual) - violation_penalty * total_violation

def selection(population, fitness_scores):
    """Select parents based on fitness"""
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        # Handle the case when all fitnesses are zero or negative
        indices = np.random.choice(len(population), size=2, replace=False)
    else:
        probs = [f/total_fitness for f in fitness_scores]
        indices = np.random.choice(len(population), size=2, p=probs, replace=False)
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2):
    """Perform crossover between selected parents"""
    child = np.zeros(parent1.shape)
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(child, mutation_rate=0.05):
    """Introduce mutations in the child"""
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.rand()
    return child

# Problem definitions
def objective_A(x):
    return 15*x[0] + 30*x[1] + 4*x[0]*x[1] - 2*x[0]**2 - 4*x[1]**2

constraints_A = [
    lambda x: x[0] + 2*x[1] - 30,
    lambda x: x[0],
    lambda x: x[1]
]

def objective_B(x):
    return 3*x[0] + 5*x[1]

constraints_B = [
    lambda x: 3*x[0] + 2*x[1] - 18,
    lambda x: x[0],
    lambda x: x[1]
]

def objective_C(x):
    return 5*x[0] - x[0]**2 + 8*x[1] - 2*x[1]**2

constraints_C = [
    lambda x: 3*x[0] + 2*x[1] - 6,
    lambda x: x[0],
    lambda x: x[1]
]

def geneticAlg(obj_func, constraints, dim, pop_size=200, max_gen=2000, mutation_rate=0.05):
    """Improved Genetic Algorithm implementation."""
    best_fitness = float('-inf')
    best_individual = None
    
    for _ in range(max_gen):
        population = createPopulation(pop_size, dim)
        fitness_scores = [fitness(ind, obj_func, constraints) for ind in population]
        new_population = []
        
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1, mutation_rate), mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        # Update the population with the new one
        population = np.array(new_population)
        current_best_fitness = max(fitness_scores)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)]
    
    return best_individual, best_fitness

solution_A, max_value_A = geneticAlg(objective_A, constraints_A, dim=2)
solution_B, max_value_B = geneticAlg(objective_B, constraints_B, dim=2)
solution_C, max_value_C = geneticAlg(objective_C, constraints_C, dim=2)

print("1a - Solution:", solution_A)
print("   - Max Value:", max_value_A)

print("---------------------------------")

print("1b - Solution:", solution_B)
print("   - Max Value:", max_value_B)

print("---------------------------------")

print("1c - Solution:", solution_C)
print("   - Max Value:", max_value_C)
