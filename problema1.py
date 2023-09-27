"""
Este código implementa un algoritmo genético para resolver problemas de optimización. 

Pasos clave:
1. **Inicialización**: Se genera una población inicial de soluciones aleatorias.
2. **Evaluación de Aptitud**: Se calcula la aptitud de cada individuo basándose en una función objetivo y restricciones.
3. **Selección**: Se seleccionan los individuos más aptos para la reproducción.
4. **Cruce**: Los individuos seleccionados son cruzados para producir descendencia.
5. **Mutación**: Se aplica una mutación aleatoria a la descendencia con cierta probabilidad.
6. **Actualización de la Población**: Se reemplaza la población anterior con la nueva generada.
7. **Actualización de la Mejor Solución**: Si se encuentra una mejor solución en una generación, se actualiza.
8. **Iteración**: Se repiten los pasos 2-7 durante un número determinado de generaciones o hasta que se cumpla un criterio de parada.

El código presenta definiciones para tres problemas (A, B, C) con diferentes funciones objetivo y conjuntos de restricciones. Al final, se aplica el algoritmo a estos tres problemas y se muestran las soluciones obtenidas.
"""

import numpy as np

def createPopulation(pop_size, dim, lower_bound=0, upper_bound=20):
    """Create a population with random values in [lower_bound, upper_bound]"""
    return lower_bound + (upper_bound - lower_bound) * np.random.rand(pop_size, dim)

def targetedInitialization(pop_size, dim, target, variation=1.0):
    """Create a population around a target solution."""
    base_population = np.array([target for _ in range(pop_size)])
    variations = (2 * np.random.rand(pop_size, dim) - 1) * variation
    return base_population + variations

def fitness(individual, obj_func, constraints):
    """Calculate the fitness based on the objective and constraints."""
    if all(constraint(individual) for constraint in constraints):
        return obj_func(individual)
    else:
        return float('-inf')

def selection(population, fitness_scores):
    """Select parents using tournament selection."""
    k = 4  # Tournament size
    selected_indices = np.argsort(np.random.choice(range(len(fitness_scores)), k))[-2:]
    return population[selected_indices[0]], population[selected_indices[1]]

def crossover(parent1, parent2):
    """Perform crossover between the selected parents."""
    child = np.zeros(parent1.shape)
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(child, mutation_rate=0.05, mutation_amount=0.1):
    """Generate mutations in the child."""
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] += mutation_amount * (2 * np.random.rand() - 1)
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


def geneticAlg(obj_func, constraints, dim, pop_size=100, max_gen=1000, mutation_rate=0.05, mutation_amount=0.1, target=None, variation=1.0):
    """Enhanced Genetic Algorithm with optional targeted initialization."""
    
    if target:
        population = targetedInitialization(pop_size, dim, target, variation)
    else:
        population = createPopulation(pop_size, dim)
    
    best_fitness = float('-inf') 
    best_individual = None

    for generation in range(max_gen):
        fitness_scores = [fitness(ind, obj_func, constraints) for ind in population]
        elite_indices = np.argsort(fitness_scores)[-5:]
        elites = [population[i] for i in elite_indices]
        
        new_population = elites[:]
        for _ in range((pop_size - len(elites)) // 2):
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1, mutation_rate, mutation_amount), mutate(child2, mutation_rate, mutation_amount)
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
        current_best_fitness = max(fitness_scores)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)]
    
    return best_individual, best_fitness



solution_A = geneticAlg(objective_A, constraints_A, dim=2, pop_size=200, max_gen=300)
solution_B = geneticAlg(objective_B, constraints_B, dim=2, pop_size=200, max_gen=300, target=[0, 9], variation=1.0)
solution_C = geneticAlg(objective_C, constraints_C, dim=2, pop_size=200, max_gen=300)

print("1a - Solution:", solution_A[0])
print("   - Max Value:", solution_A[1])
print("---------------------------------")
print("1b - Solution:", solution_B[0])
print("   - Max Value:", solution_B[1])
print("---------------------------------")
print("1c - Solution:", solution_C[0])
print("   - Max Value:", solution_C[1])
