import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Parámetros
N = 3000           # Número de partículas
V0 = 2             # Velocidad lineal de las partículas
Dt = 0.005         # Coeficiente de difusión translacional
Dr = 0.001         # Coeficiente de difusión rotacional
t = 60             # Tiempo total de la simulación en segundos
dt = 0.01          # Paso de tiempo en segundos
w = 2              # Velocidad angular
R = 3.0            # Radio del círculo donde se mueven las partículas
prob = [0.0, 1.0]  # Probabilidades de no-adsorción y adsorción respectivamente
lambd = 10         # Tasa de llegada de eventos (adsorción)

# Tipo de Animación
detach = False     # Permite que las partículas se despeguen
interact = True    # Indica si hay interacción entre las partículas
save_a = True      # Guardar los datos de adsorción

# Condiciones iniciales de las particulas
X = np.zeros(N)                                      # Arreglo posicion actual de cada particula en x
Y = np.zeros(N)                                      # Arreglo posicion actual de cada particula en y
Phi = np.random.uniform(0, 2*np.pi , N)         # Angulo aleatorio entre 0 y 2pi
Tau = np.random.exponential(scale=1 / lambd, size=N) # Arreglo con los tiempos de relajacion
Con = np.zeros(N, dtype=bool)                        # Arreglo inidica si esta o no adsorbida

# Arrays para registrar los datos de adsorción
time_steps = []
adsorbed_particles = []

# Calcula el punto de contacto
def interseccion(x1, y1, x2, y2, r):
    dx = x2 - x1
    dy = y2 - y1

    # Coeficientes de la ecuación cuadrática
    a = dx ** 2 + dy ** 2
    b = 2 * (x1 * dx + y1 * dy)
    c = x1 ** 2 + y1 ** 2 - r ** 2

    discriminante = b ** 2 - 4 * a * c

    # Si el discriminante es negativo, no hay intersección
    if discriminante < 0:
        return None, None

    # Calcular las soluciones de la ecuación cuadrática
    t1 = (-b + np.sqrt(discriminante)) / (2 * a)
    t2 = (-b - np.sqrt(discriminante)) / (2 * a)

    # Elegir la solución que está en el rango [0, 1]
    if 0 <= t1 <= 1:
        intersection_x = x1 + t1 * dx
        intersection_y = y1 + t1 * dy
        return intersection_x, intersection_y
    elif 0 <= t2 <= 1:
        intersection_x = x1 + t2 * dx
        intersection_y = y1 + t2 * dy
        return intersection_x, intersection_y
    else:
        return None, None


# Funcion para actualizar la posicion y orientacion de las particulas
def posicion_y_orientacion():
    for i in range(N):
        x = X[i]
        y = Y[i]
        phi = Phi[i]

        if interact == True:
            phi_no = phi[phi != 0]
            p_phi = np.mean(phi_no)  # Interaccion entre las particulas
        else:
            p_phi = phi

        if not Con[i]:
            # Generamos los incrementos
            dx = V0 * np.cos(phi) * dt + np.sqrt(2 * Dt * dt) * np.random.normal(0, 1)
            dy = V0 * np.sin(phi) * dt + np.sqrt(2 * Dt * dt) * np.random.normal(0, 1)
            dphi = w * dt + np.sqrt(2 * Dr * dt) * np.random.normal(0, 1)

            #print(f"El valor de dx es: {dx}")

            x_s = x + dx
            y_s = y + dy

            # Calcular la distancia entre el centro del círculo y la posición actual de la partícula
            distance = np.sqrt(x_s ** 2 + y_s ** 2)

            if distance >= R:  # Si la particula esta por salir del circulo

                impact_x, impact_y = interseccion(x, y, x_s, y_s, R)
                s = np.random.choice([0, 1], p=prob)

                if s == 1:  # Si la particula puede ser adsorbida
                    Con[i] = True
                    X[i] = impact_x
                    Y[i] = impact_y

                else:  # La particula rebotara
                    v_incidence = np.array([impact_x, impact_y])  # Vector de incidencia real
                    n_vector = np.array([x, y])  # Vector normal
                    reflection_direction = v_incidence - 2 * np.dot(v_incidence,
                                                                    n_vector) * n_vector  # Vector reflejado

                    # Reflejar la dirección de la partícula
                    theta = np.arctan2(reflection_direction[1], reflection_direction[0])
                    dx = V0 * np.cos(theta) * dt
                    dy = V0 * np.sin(theta) * dt
                    X[i] += dx
                    Y[i] += dy
                    Phi[i] = p_phi + dphi
            else:
                # Actualizar las posiciones y orientación de la partícula
                X[i] += dx
                Y[i] += dy
                Phi[i] = p_phi + dphi

        else:  # Si Con[i] es True no actualizamos la posicion de la particula
            if detach == True:
                if Tau[i] <= 0:
                    while True:
                        # Generamos los incrementos
                        dx = V0 * np.cos(phi) * dt + np.sqrt(2 * Dt * dt) * np.random.normal(0, 1)
                        dy = V0 * np.sin(phi) * dt + np.sqrt(2 * Dt * dt) * np.random.normal(0, 1)
                        dphi = w * dt + np.sqrt(2 * Dr * dt) * np.random.normal(0, 1)

                        # Calcular la distancia entre el centro del círculo y la nueva posición de la partícula
                        distance = np.sqrt(np.power((x + dx), 2) + np.power((y + dy), 2))

                        # Si la distancia es menor que el radio R, salimos del bucle
                        if distance <= R:
                            x += dx
                            y += dy
                            phi = p_phi + dphi

                            X[i] = x
                            Y[i] = y
                            Phi[i] = phi
                            # Si se cumple la condición, actualizamos la posición
                            break
                        else:
                            continue
                    Tau[i] = np.random.exponential(scale=1 / lambd)
                    Con[i] = False
                else:
                    Tau[i] -= dt
                    continue
            else:
                continue
    return X, Y, Con


# Crear la figura y los ejes
fig, ax = plt.subplots()


# Crear la animación
def update(frame):
    x,y,con = posicion_y_orientacion()

    # Guardar datos en arrays
    if save_a is True:
        current_time = frame * dt
        time_steps.append(current_time)
        adsorbed_particles.append(np.sum(Con))

    ax.clear()
    ax.set_xlabel("Posición en x")
    ax.set_ylabel("Posición en y")
    ax.set_title("Partículas activas ")
    scatter = ax.scatter(x, y, c='blue', marker='o', s=2)
    circle_patch = Circle((0, 0), R, edgecolor='black', facecolor='none')
    ax.add_patch(circle_patch)

    return scatter, circle_patch  # Devolver una lista de objetos Artist

#Itera la animación frame por frame
animation = FuncAnimation(fig, update, frames=int(t / dt), interval=9, repeat=False)

# Mostrar la animación
plt.show()


# Guardar datos en un archivo para análisis posterior
if save_a is True:
    import pandas as pd
    data = {
        "Tiempo": time_steps,
        "Partículas Adsorbidas": adsorbed_particles
    }
    df = pd.DataFrame(data)
    name = input("Nombre de los datos: ")
    df.to_csv(name + ".csv", index=False)

    print("Simulación completada y datos guardados en 'resultados_simulacion.csv'")

