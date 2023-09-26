import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

'''
Situacion: Efectos de la urbanizacion en la naturaleza local
El proceso de urbanizacion es la transformacion de areas rurales en areas urbanas, lo que conlleva a la construccion de infraestructura, casas, carreteras, etc. Esto tiene un impacto natural en las plantas y animales y la biodiversidad del area en general.
Variables:
1.  Nivel de urbanizacion: Indica cuanta tierra ha sido convertida en areas urbanas. Cuando aumenta, mas tierra se convierte de lo natural a lo urbanizado.
2.  Biodiversidad Local: Representa la variedad de vida en un area particular: plantas, animales, etc.
3.  Zonas verdes: areas dentro de un entorno urbano que se mantienen con vegetacion, como parques y jardines. Ofrecen refugio a especies y conectan fragmentos de habitat.
4.  Contaminacion ambiental: Aumenta la contaminacion al urbanizar debido a los vehiculos, industrias, desechos.
5.  Educacion Ambiental: Programas diseñados para enseñar a la gente sobre el medio ambiente y como sus acciones pueden afectarlo.
6.  Crecimiento Poblacional: Representa el aumento de la poblacion en un area determinada. Esto podria influir directamente en el nivel de urbanizacion.
7.  Percepcion Publica: Refleja como la sociedad percibe y valora el medio ambiente y la biodiversidad. Podria ser influenciada por la biodiversidad local y, a su vez, influir en la educacion ambiental.
'''

# ****************** Diagrama de influencia ******************
'''
Es una herramienta grafica que muestra las relaciones causales entre diferentes variables o elementos en un sistema. Estos diagramas suelen usarse en la toma de decisiones y analisis de sistemas,
y muestran como los cambios en una variable pueden influir en otras. Las variables se representan mediante nodos, y las relaciones causales se indican mediante flechas que apuntan de la variable 
causante a la variable afectada.

Relaciones
- El aumento en el Nivel de Urbanizacion tiene un impacto negativo en la Biodiversidad Local (menos habitat, fragmentacion, etc.).
- Las Zonas Verdes tienen un impacto positivo en la Biodiversidad Local, proporcionando espacios para la fauna y flora.
- La Contaminacion Ambiental tiene un efecto negativo en la Biodiversidad Local (afecta la salud y supervivencia de las especies).
- La Educacion Ambiental tiene efectos positivos indirectos: puede llevar a la creacion de mas Zonas Verdes y a la reduccion de la Contaminacion Ambiental al concientizar a la poblacion.
- El Crecimiento Poblacional tiene un impacto positivo en el Nivel de Urbanizacion.
- La disminucion en la Biodiversidad Local podria tener un impacto positivo en la Educacion Ambiental debido a una mayor conciencia.
- La Percepcion Publica es influenciada positivamente por la Biodiversidad Local y, a su vez, influye positivamente en la Educacion Ambiental.
- Las Zonas Verdes podrian influir positivamente en la Percepcion Publica al mejorar la calidad de vida en las areas urbanas.
- La Contaminacion Ambiental podria tener un efecto negativo en las Zonas Verdes.
'''
# Crear un grafo dirigido ampliado
G_extended = nx.DiGraph()

# Definir los nodos (variables)
nodes = ['Nivel de Urbanizacion', 'Biodiversidad Local',
         'Zonas Verdes', 'Contaminacion Ambiental', 'Educacion Ambiental']

# Definir los nodos (variables) adicionales
nodes_extended = ['Crecimiento Poblacional', 'Percepcion Publica']

# Añadir todos los nodos al grafo
G_extended.add_nodes_from(nodes + nodes_extended)

# Definir las aristas (relaciones de influencia)
edges = [
    ('Nivel de Urbanizacion', 'Biodiversidad Local',
     {'weight': -1, 'label': '-'}),
    ('Zonas Verdes', 'Biodiversidad Local', {'weight': 1, 'label': '+'}),
    ('Contaminacion Ambiental', 'Biodiversidad Local',
     {'weight': -1, 'label': '-'}),
    ('Educacion Ambiental', 'Zonas Verdes', {'weight': 1, 'label': '+'}),
    ('Nivel de Urbanizacion', 'Zonas Verdes', {'weight': -1, 'label': '-'}),
    ('Nivel de Urbanizacion', 'Contaminacion Ambiental',
     {'weight': 1, 'label': '+'}),
    ('Educacion Ambiental', 'Contaminacion Ambiental',
     {'weight': -1, 'label': '-'})
]
# Definir las aristas (relaciones de influencia) adicionales
edges_extended = [
    ('Crecimiento Poblacional', 'Nivel de Urbanizacion',
     {'weight': 1, 'label': '+'}),
    ('Biodiversidad Local', 'Educacion Ambiental',
     {'weight': 1, 'label': '+'}),
    ('Biodiversidad Local', 'Percepcion Publica', {'weight': 1, 'label': '+'}),
    ('Percepcion Publica', 'Educacion Ambiental', {'weight': 1, 'label': '+'}),
    ('Zonas Verdes', 'Percepcion Publica', {'weight': 1, 'label': '+'}),
    ('Contaminacion Ambiental', 'Zonas Verdes', {'weight': -1, 'label': '-'})
]

# Añadir todas las aristas al grafo
G_extended.add_edges_from([(u, v, attr)
                          for u, v, attr in edges + edges_extended])

# Dibujar el grafo ampliado
# Ajustar la posicion de los nodos para una distribucion mas ordenada y centrada
pos_adjusted = {
    'Nivel de Urbanizacion': (0, 1),
    'Biodiversidad Local': (0, 0),
    'Zonas Verdes': (-1, 0.5),
    'Contaminacion Ambiental': (1, 0.5),
    'Educacion Ambiental': (0, -1),
    'Crecimiento Poblacional': (-1, -0.5),
    'Percepcion Publica': (1, -0.5)
}

# Dibujar el grafo con la posicion ajustada
plt.figure(figsize=(15, 10))
nx.draw(G_extended, pos_adjusted, with_labels=True, node_size=3500,
        node_color='green', font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G_extended, pos_adjusted, edge_labels={(
    u, v): G_extended[u][v]['label'] for u, v in G_extended.edges()}, font_size=12)
plt.title("Diagrama de Influencia Ajustado: Efectos de la urbanizacion en la biodiversidad local")
plt.show()


# ****************** Diagrama de flujo de nivel ******************
'''
Es una representacion mas detallada de como cambian las variables con el tiempo. Este diagrama incluira ecuaciones diferenciales que describen las tasas de cambio de cada variable.

"U": "Nivel de Urbanizacion: Representa cuanto se ha urbanizado una zona."
"B": "Biodiversidad Local: Representa la salud ecologica del area."
"Z": "Zonas Verdes: Representa areas verdes en la ciudad."
"C": "Contaminacion Ambiental: Representa el nivel de contaminacion."
"E": "Educacion Ambiental: Representa los esfuerzos para educar al publico sobre temas ambientales."
"P": "Crecimiento Poblacional: Representa el aumento de la poblacion en el area."
"R": "Percepcion Publica: Representa como la sociedad valora y percibe el medio ambiente."
'''

# Definicion de variables y sus descripciones
variables = {
    "U": "Nivel de Urbanizacion: Representa cuanto se ha urbanizado una zona.",
    "B": "Biodiversidad Local: Representa la salud ecologica del area.",
    "Z": "Zonas Verdes: Representa areas verdes en la ciudad.",
    "C": "Contaminacion Ambiental: Representa el nivel de contaminacion.",
    "E": "Educacion Ambiental: Representa los esfuerzos para educar al publico sobre temas ambientales.",
    "P": "Crecimiento Poblacional: Representa el aumento de la poblacion en el area.",
    "R": "Percepcion Publica: Representa como la sociedad valora y percibe el medio ambiente."
}


# Ecuaciones diferenciales
def ecuacionesDiferenciales(t, y, k):
    """
    Define el sistema de ecuaciones diferenciales basado en las interacciones descritas.

    Parametros:
    - y: Un array con los valores actuales de las variables [U, B, Z, C, E, P, R]
    - k: Un array con los coeficientes de las ecuaciones [k1, k2, ..., k12]

    Retorna:
    - dydt: Un array con las derivadas de cada variable respecto al tiempo

    Las constantes ki (donde i varia de 1 a 12) representan coeficientes que determinan la fuerza de las interacciones entre las variables. 
    Estos coeficientes pueden obtenerse a partir de datos reales o ser estimados segun estudios y expertos.
    """
    U, B, Z, C, E, P, R = y
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = k

    # Nivel de Urbanizacion: El nivel de urbanizacion aumenta a una tasa proporcional al crecimiento poblacional.
    dU_dt = k1 * P
    # Biodiversidad Local: La biodiversidad es afectada negativamente por la urbanizacion y la contaminacion, pero beneficiada por las zonas verdes y la educacion ambiental.
    dB_dt = -k2 * U + k3 * Z - k4 * C + k5 * E
    # Zonas Verdes: Las zonas verdes aumentan con la educacion ambiental pero disminuyen con la urbanizacion y la contaminacion.
    dZ_dt = k6 * E - k7 * U - k8 * C
    # Contaminacion Ambiental: La contaminacion aumenta con la urbanizacion y disminuye con la educacion ambiental.
    dC_dt = k9 * U - k10 * E
    # Educacion Ambiental: La educacion ambiental es impulsada por la percepcion publica.
    dE_dt = k11 * R
    # Crecimiento Poblacional (asumimos que es constante, por lo que su derivada es 0)
    dP_dt = 0
    # Percepcion Publica: La percepcion publica mejora con una biodiversidad saludable pero disminuye con el aumento de la urbanizacion.
    dR_dt = k12 * B - k12 * U

    dydt = np.array([dU_dt, dB_dt, dZ_dt, dC_dt, dE_dt, dP_dt, dR_dt])

    return dydt


# ****************** Simulacion metodo de Euler ******************
'''
El metodo de Euler es uno de los metodos mas simples para resolver ecuaciones diferenciales numericamente. 
La idea basica es usar la derivada de una funcion para encontrar su valor en un pequeño paso en el futuro.
'''


def metodoEuler(func, y0, t, k):
    """
    Implementacion del metodo de Euler para sistemas de ecuaciones diferenciales.

    Parametros:
    - func: funcion que define el sistema de ecuaciones diferenciales.
    - y0: condiciones iniciales.
    - t: array de tiempos.
    - k: coeficientes de las ecuaciones.

    Retorna:
    - Y: array con los valores de las variables en cada paso de tiempo.
    """

    h = t[1] - t[0]  # Tamaño del paso de tiempo
    Y = np.empty((len(t), len(y0)))
    Y[0] = y0

    for i in range(0, len(t) - 1):
        Y[i + 1] = Y[i] + h * func(t[i], Y[i], k)

    return Y


# Definir condiciones iniciales y coeficientes
# Condiciones iniciales para [U, B, Z, C, E, P, R]
y0 = [0.3, 0.9, 0.4, 0.3, 0.5, 0.2, 0.5]
# Coeficientes para el sistema
k_values = [0.05, 0.3, 0.15, 0.15, 0.05, 0.15, 0.1, 0.1, 0.1, 0.05, 0.15, 0.1]


# Definir el rango de tiempo para la simulacion
# Simularemos desde t=0 hasta t=100 con 1000 pasos
t = np.linspace(0, 100, 1000)

# Realizar la simulacion
Y = metodoEuler(ecuacionesDiferenciales, y0, t, k_values)

# Visualizar los resultados
plt.figure(figsize=(14, 8))
for i, var in enumerate(variables.keys()):
    plt.plot(t, Y[:, i], label=variables[var].split(":")[0])

plt.title("Simulacion de Efectos de la Urbanizacion en la Biodiversidad Local")
plt.xlabel("Tiempo")
plt.ylabel("Valor de las Variables")
plt.legend()
plt.grid(True)
plt.show()
