import pulp
import numpy as np

'''
Para resolver los problemas primal y dual de programación lineal, se utilizan métodos de optimización lineal, 
como el método simplex o el método de las dos fases. Estos métodos encuentran las soluciones óptimas al ajustar 
gradualmente los valores de las variables de decisión para maximizar o minimizar la función objetivo sujeta a las restricciones.

Resolución del Problema Primal:

El objetivo en el problema primal es maximizar la función objetivo Z = x1 - x2 + x3 sujeta a las restricciones dadas. 
Para resolverlo, se puede utilizar el método simplex, que es un algoritmo iterativo que comienza con una solución factible y 
se mueve de forma iterativa hacia una solución óptima mejorando la función objetivo.

El algoritmo simplex se ejecuta en varias iteraciones. En cada iteración, selecciona una variable de decisión 
(en este caso, x1, x2, o x3) para aumentar su valor o reducir su valor, lo que mejora la función objetivo. 
Se sigue iterando hasta que no sea posible mejorar más la función objetivo, y se obtiene una solución óptima.
'''


# Crear el problema primal
lp_problem_primal = pulp.LpProblem("Primal", pulp.LpMaximize)

# Variables de decisión
x1 = pulp.LpVariable('x1', lowBound=0)
x2 = pulp.LpVariable('x2', lowBound=0)
x3 = pulp.LpVariable('x3', lowBound=0)

# Función objetivo
lp_problem_primal += x1 - x2 + x3, "Z"

# Restricciones
constraints = [
    x1 + x2 + 2*x3 <= 5,
    2*x1 + x2 + x3 <= 7,
    2*x1 - x2 + 3*x3 <= 8,
    x1 + 2*x2 + 5*x3 <= 9
]

for constraint in constraints:
    lp_problem_primal += constraint

# Resolver el problema primal
lp_problem_primal.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Problema 3a (Maximizar) ---")
# Imprimir resultados del problema primal
print("Resultado del problema primal:")
print("Estado:", pulp.LpStatus[lp_problem_primal.status])
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)
print("x3 =", x3.varValue)
print("Valor óptimo (Z) =", pulp.value(lp_problem_primal.objective))

# Crear el problema dual
lp_problem_dual = pulp.LpProblem("Dual", pulp.LpMinimize)

# Variables de decisión para el dual
y1 = pulp.LpVariable('y1', lowBound=0)
y2 = pulp.LpVariable('y2', lowBound=0)
y3 = pulp.LpVariable('y3', lowBound=0)
y4 = pulp.LpVariable('y4', lowBound=0)
print("-----------------------------------")

print("\n--- Problema 3b (Maximizar) ---")

# Crear el problema primal
lp_problem_primal = pulp.LpProblem("Primal", pulp.LpMaximize)

# Variables de decisión
x1 = pulp.LpVariable('x1', lowBound=0)
x2 = pulp.LpVariable('x2', lowBound=0)
x3 = pulp.LpVariable('x3', lowBound=0)
x4 = pulp.LpVariable('x4', lowBound=0)

# Función objetivo
lp_problem_primal += 5*x1 + 7*x2 + 15*x3 + 6*x4, "Z"

# Restricciones
constraints = [
    x1 + 2*x2 + x4 <= 1,
    x1 + 3*x2 + x3 <= 2,
    x1 + 4*x2 + 3*x3 + 2*x4 <= 3,
    x1 + 5*x3 + 3*x4 <= 4
]

for constraint in constraints:
    lp_problem_primal += constraint

# Resolver el problema primal
lp_problem_primal.solve(pulp.PULP_CBC_CMD(msg=False))

# Imprimir resultados del problema primal
print("Resultado del problema primal:")
print("Estado:", pulp.LpStatus[lp_problem_primal.status])
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)
print("x3 =", x3.varValue)
print("x4 =", x4.varValue)
print("Valor óptimo (Z) =", pulp.value(lp_problem_primal.objective))
print("-----------------------------------")

print("\n--- Problema 3c (Minimizar) ---")
# Crear el problema dual
lp_problem_dual = pulp.LpProblem("Dual", pulp.LpMinimize)

# Variables de decisión duales
y1 = pulp.LpVariable('y1', lowBound=0)
y2 = pulp.LpVariable('y2', lowBound=0)
y3 = pulp.LpVariable('y3', lowBound=0)
y4 = pulp.LpVariable('y4', lowBound=0)

# Función objetivo del dual
lp_problem_dual += 3*y1 + y2 + 5*y3 + 12*y4, "Z"

# Restricciones del dual
dual_constraints = [
    y1 + y2 + y3 + y4 >= 10,
    2*y1 - y2 + y3 + 2*y4 >= 14,
    5*y1 - 8*y2 - 3*y3 + 3*y4 >= 5,
    2*y1 - y2 - 5*y3 + 3*y4 >= 0
]

for constraint in dual_constraints:
    lp_problem_dual += constraint

# Resolver el problema dual
lp_problem_dual.solve(pulp.PULP_CBC_CMD(msg=False))

# Imprimir resultados del problema dual
print("Resultado del problema dual:")
print("Estado:", pulp.LpStatus[lp_problem_dual.status])
print("y1 =", y1.varValue)
print("y2 =", y2.varValue)
print("y3 =", y3.varValue)
print("y4 =", y4.varValue)
print("Valor óptimo (Z) =", pulp.value(lp_problem_dual.objective))