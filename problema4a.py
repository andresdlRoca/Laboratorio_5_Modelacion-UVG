import numpy as np

# Definir la función objetivo y su gradiente
def objective_function(x1, x2):
    return ((x1 - 10) ** 2) / 25 + ((x2 - 20) ** 2) / 16

def gradient(x1, x2):
    grad_x1 = (2/25) * (x1 - 10)
    grad_x2 = (2/16) * (x2 - 20)
    return np.array([grad_x1, grad_x2])

# Parámetros del algoritmo de descenso de gradiente
learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-6

# Inicialización de variables
x = np.array([0.0, 0.0])

# Algoritmo de descenso de gradiente
for i in range(max_iterations):
    gradient_value = gradient(x[0], x[1])
    x_new = x - learning_rate * gradient_value
    if np.linalg.norm(x_new - x) < tolerance:
        break
    x = x_new

# Imprimir el resultado
print("Resultado después de", i+1, "iteraciones:")
print("x1 =", x[0])
print("x2 =", x[1])
print("Valor mínimo de la función =", objective_function(x[0], x[1]))
