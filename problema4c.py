import numpy as np

# Definir la función objetivo y su gradiente
def objective_function(x1, x2):
    return -(15 * x1 + 30 * x2 + 4 * x1 * x2 - 2 * x1**2 - 4 * x2**2)  # Negativo para convertirlo en un problema de maximización

def gradient(x1, x2):
    grad_x1 = 15 + 4 * x2 - 4 * x1
    grad_x2 = 30 + 4 * x1 - 8 * x2
    return np.array([grad_x1, grad_x2])

# Parámetros del algoritmo de maximización
learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-6

# Inicialización de variables
x = np.array([0.0, 0.0])

# Algoritmo de maximización
for i in range(max_iterations):
    gradient_value = gradient(x[0], x[1])
    x_new = x + learning_rate * gradient_value
    if np.linalg.norm(x_new - x) < tolerance:
        break
    x = x_new

# Imprimir el resultado
print("Resultado después de", i+1, "iteraciones:")
print("x1 =", x[0])
print("x2 =", x[1])
print("Valor máximo de la función =", -objective_function(x[0], x[1]))  # Negativo para obtener el valor máximo
