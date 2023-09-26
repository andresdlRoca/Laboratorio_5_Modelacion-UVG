"""
Este código se centra en el análisis y visualización de sistemas de ecuaciones diferenciales de primer orden en un plano. 

Pasos clave:
1. **Método de Runge-Kutta**: Se utiliza este método numérico para resolver ecuaciones diferenciales ordinarias. La solución se aproxima utilizando cuatro estimaciones en cada paso.
2. **Definiciones de Ecuaciones Diferenciales**: Se definen tres conjuntos de ecuaciones diferenciales, etiquetados como 'a', 'b' y 'c'.
3. **Búsqueda de Puntos Críticos**: Se determinan puntos críticos del sistema utilizando 'fsolve', donde el sistema no experimenta cambios.
4. **Representación Gráfica del Campo Vectorial**: Se visualiza la dirección y magnitud de los cambios del sistema a través de campos vectoriales.
5. **Condiciones Iniciales y Parámetros**: Se establecen condiciones iniciales y parámetros para cada sistema de ecuaciones diferenciales.
6. **Visualización**: Se crean gráficos que muestran el campo vectorial de cada sistema, y se superponen las trayectorias obtenidas con el método de Runge-Kutta.

El resultado es una serie de gráficos que muestran la evolución de las soluciones para cada sistema de ecuaciones diferenciales en el plano.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Metodo Runge-Kutta 
def rungeKutta(f, g, x0, y0, h, num_steps):
    x_values = [x0]
    y_values = [y0]
    
    for _ in range(num_steps):
        k1x = h * f(x_values[-1], y_values[-1])
        k1y = h * g(x_values[-1], y_values[-1])
        
        k2x = h * f(x_values[-1] + 0.5 * h, y_values[-1] + 0.5 * k1y)
        k2y = h * g(x_values[-1] + 0.5 * h, y_values[-1] + 0.5 * k1x)
        
        k3x = h * f(x_values[-1] + 0.5 * h, y_values[-1] + 0.5 * k2y)
        k3y = h * g(x_values[-1] + 0.5 * h, y_values[-1] + 0.5 * k2x)
        
        k4x = h * f(x_values[-1] + h, y_values[-1] + k3y)
        k4y = h * g(x_values[-1] + h, y_values[-1] + k3x)
        
        x_new = x_values[-1] + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y_new = y_values[-1] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return x_values, y_values

# Encontrar puntos criticos
def find_critical_points(f, g):
    def equations(p):
        x, y = p
        return [f(x, y), g(x, y)]

    critical_points = []
    
    for guess in [(1, 1), (-1, -1)]:
        critical_point = fsolve(equations, guess)

        critical_point_tuple = tuple(critical_point)
        
        if critical_point_tuple not in critical_points:
            critical_points.append(critical_point_tuple)

    return critical_points

# Checkeo de stability analysis para el sistema
def stability_analysis(A):
    eigenvalues = np.linalg.eigvals(A)
    
    if all(np.real(eigenvalues) < 0):
        stability = "Stable (Attracting)"
    elif all(np.real(eigenvalues) > 0):
        stability = "Unstable (Repelling)"
    else:
        stability = "Saddle Point"
    
    return stability

# Graficar
def plot_vector_field(f, g, x_range, y_range, ax):
    x, y = np.meshgrid(x_range, y_range)
    dx = f(x, y)
    dy = g(x, y)
    ax.quiver(x, y, dx, dy, scale=20)


# Definir las funciones

# a
def f_a(x, y):
    return -x + y

def g_a(x, y):
    return 4 * x - y

# b
def f_b(x, y):
    return -x + y

def g_b(x, y):
    return -y

# c
def f_c(x, y):
    return -x + y

def g_c(x, y):
    return -9 * x - y


# Definir condiciones

# a
x0_a = -1
y0_a = 1
h_a = 0.01
num_steps_a = 1000
critical_points_a = find_critical_points(f_a, g_a)
x_values_a, y_values_a = rungeKutta(f_a, g_a, x0_a, y0_a, h_a, num_steps_a)
print("Critical Points for Problem a:", critical_points_a)

# b
x0_b = 1
y0_b = 1
h_b = 0.01
num_steps_b = 1000
critical_points_b = find_critical_points(f_b, g_b)
print("Critical Points for Problem b:", critical_points_b)
x_values_b, y_values_b = rungeKutta(f_b, g_b, x0_b, y0_b, h_b, num_steps_b)

# c
x0_c = 1
y0_c = 1
h_c = 0.01
num_steps_c = 1000
critical_points_c = find_critical_points(f_c, g_c)
print("Critical Points for Problem c:", critical_points_c)
x_values_c, y_values_c = rungeKutta(f_c, g_c, x0_c, y0_c, h_c, num_steps_c)


# Plot the vector fields for each problem
x_range = np.linspace(-2, 2, 20)
y_range = np.linspace(-2, 2, 20)

plt.figure(figsize=(15, 5))

# a
plt.subplot(131)
plot_vector_field(f_a, g_a, x_range, y_range, plt.gca())
plt.plot(x_values_a, y_values_a, 'r-', label='Trajectory')  # Plot the trajectory from Runge-Kutta
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem a')
plt.legend()

# b
plt.subplot(132)
plot_vector_field(f_b, g_b, x_range, y_range, plt.gca())
plt.plot(x_values_b, y_values_b, 'r-', label='Trajectory')  # Plot the trajectory from Runge-Kutta
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem b')
plt.legend()

# c
plt.subplot(133)
plot_vector_field(f_c, g_c, x_range, y_range, plt.gca())
plt.plot(x_values_c, y_values_c, 'r-', label='Trajectory')  # Plot the trajectory from Runge-Kutta
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem c')
plt.legend()

plt.tight_layout()
plt.show()
