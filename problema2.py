import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the Runge-Kutta method
def runge_kutta(f, g, x0, y0, h, num_steps):
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

# Define the differential equations

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

# Define a function to find the critical points using fsolve
def find_critical_points(f, g):
    def equations(p):
        x, y = p
        return [f(x, y), g(x, y)]

    critical_points = []
    
    # Iterate through initial guesses to find multiple critical points
    for guess in [(1, 1), (-1, -1)]:
        critical_point = fsolve(equations, guess)
        
        # Convert the critical point array to a tuple for easier handling
        critical_point_tuple = tuple(critical_point)
        
        if critical_point_tuple not in critical_points:
            critical_points.append(critical_point_tuple)

    return critical_points


# Perform stability analysis for a given system and critical points
def stability_analysis(A):
    eigenvalues = np.linalg.eigvals(A)
    
    if all(np.real(eigenvalues) < 0):
        stability = "Stable (Attracting)"
    elif all(np.real(eigenvalues) > 0):
        stability = "Unstable (Repelling)"
    else:
        stability = "Saddle Point"
    
    return stability

# Plot the vector field for a given system of equations
def plot_vector_field(f, g, x_range, y_range, ax):
    x, y = np.meshgrid(x_range, y_range)
    dx = f(x, y)
    dy = g(x, y)
    ax.quiver(x, y, dx, dy, scale=20)

# Define the initial conditions and parameters for each problem

# a
x0_a = -1
y0_a = 1
h_a = 0.01
num_steps_a = 1000
critical_points_a = find_critical_points(f_a, g_a)
print("Critical Points for Problem a:", critical_points_a)

# b
x0_b = 1
y0_b = 1
h_b = 0.01
num_steps_b = 1000
critical_points_b = find_critical_points(f_b, g_b)
print("Critical Points for Problem b:", critical_points_b)

# c
x0_c = 1
y0_c = 1
h_c = 0.01
num_steps_c = 1000
critical_points_c = find_critical_points(f_c, g_c)
print("Critical Points for Problem c:", critical_points_c)



# Plot the vector fields for each problem
x_range = np.linspace(-2, 2, 20)
y_range = np.linspace(-2, 2, 20)

plt.figure(figsize=(15, 5))

# a
plt.subplot(131)
plot_vector_field(f_a, g_a, x_range, y_range, plt.gca())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem a')

# b
plt.subplot(132)
plot_vector_field(f_b, g_b, x_range, y_range, plt.gca())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem b')

# c
plt.subplot(133)
plot_vector_field(f_c, g_c, x_range, y_range, plt.gca())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for Problem c')

plt.tight_layout()
plt.show()
