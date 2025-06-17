# %% [markdown]
#  # Laboratorium 11 - Spadek wzdłuż gradientu

# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.optimize import golden
from itertools import cycle


# %% [markdown]
# ## Zadanie 2.
#
# Należy wyznaczyć najkrótszą ścieżkę robota pomiędzy dwoma punktami $x^{(0)}$ i $x^{(n)}$. Problemem są przeszkody usytuowane na trasie robota, których należy unikać. Zadanie polega na minimalizacji funkcji kosztu, która sprowadza problem nieliniowej optymalizacji z ograniczeniami do problemu nieograniczonego optymalizacji.
#
# Macierz $X \in \mathbb{R}^{(n+1) \times 2}$ opisuje ścieżkę złożoną z $n+1$ punktów $x^{(0)}, x^{(1)}, x^{(2)}, \ldots, x^{(n)}$. Każdy punkt $x^{(i)} \in \mathbb{R}^2$. Punkty początkowy i końcowy ścieżki, $x^{(0)}$ i $x^{(n)}$, są ustalone.
#
# Punkty z przeszkodami (punkty o 2 współrzędnych), $r^{(j)}$ dane są w macierzy przeszkód $R \in \mathbb{R}^{k \times 2}$.
#
# W celu optymalizacji ścieżki robota należy użyć metody największego spadku. Funkcja celu użyta do optymalizacji $F(x^{(0)}, x^{(1)}, \ldots, x^{(n)})$ zdefiniowana jest jako:
#
# $$F(x^{(0)}, x^{(1)}, \ldots, x^{(n)}) = \lambda_1 \sum_{i=0}^{n} \sum_{j=1}^{k} \frac{1}{\epsilon + \|x^{(i)} - r^{(j)}\|^2} + \lambda_2 \sum_{i=0}^{n-1} \|x^{(i+1)} - x^{(i)}\|^2$$
#
# Symbole użyte we wzorze mają następujące znaczenie:
#
# * Stałe $\lambda_1$ i $\lambda_2$ określają wpływ każdego członu wyrażenia na wartość $F(X)$.
#     * $\lambda_1$ określa wagę składnika zapobiegającego zbytniemu zbliżaniu się do przeszkody
#     * $\lambda_2$ określa wagę składnika zapobiegającego tworzeniu bardzo długich ścieżek
#
# * $n$ jest liczbą odcinków, a $n+1$ liczbą punktów na trasie robota.
# * $k$ jest liczbą przeszkód, których robot musi unikać.
# * Dodanie $\epsilon$ w mianowniku zapobiega dzieleniu przez zero.
#
# 1.  Wyprowadź wyrażenie na gradient $\nabla F$ funkcji celu $F$ względem $x^{(i)}$:
#     $\nabla F = \left[ \frac{\partial F}{\partial x^{(0)}}, \ldots, \frac{\partial F}{\partial x^{(n)}} \right]$.
#     Wzór wyraź poprzez wektory $x^{(i)}$ i ich składowe, wektory $r^{(j)}$ i ich składowe, $\epsilon, \lambda_1, \lambda_2, n$ i $k$ (niekoniecznie wszystkie).
#     Wskazówka. $\frac{\partial \|z\|^2}{\partial z} = 2z$.
#
# 2.  Opisz matematycznie i zaimplementuj kroki algorytmu największego spadku z przeszukiwaniem liniowym, który służy do minimalizacji funkcji celu $F$. Do przeszukiwania liniowego (ang. *line search*) użyj metody złotego podziału (ang. *golden section search*). W tym celu załóż, że $F$ jest unimodalna (w rzeczywistości tak nie jest) i że można ustalić początkowy przedział, w którym znajduje się minimum.
#
# 3.  Znajdź najkrótszą ścieżkę robota przy użyciu algorytmu zaimplementowanego w w poprzednim punkcie. Przyjmij następujące wartości parametrów:
#     * $n = 20, k = 50$
#     * $x^{(0)} = [0, 0]$, $x^{(n)} = [20, 20]$
#     * $r^{(j)} \sim \mathcal{U}(0, 20) \times \mathcal{U}(0, 20)$
#     * $\lambda_1 = \lambda_2 = 1$
#     * $\epsilon = 10^{-13}$
#     * liczba iteracji = 400
#
# Ponieważ nie chcemy zmieniać położenia punktu początkowego i końcowego, $x^{(0)}, x^{(n)}$, wyzeruj gradient funkcji $F$ względem tych punktów.
#
# Obliczenia przeprowadź dla 5 różnych losowych inicjalizacji punktów wewnątrz ścieżki $x^{(1)}, \ldots, x^{(n-1)}$.
#
# Narysuj przykładowy wykres wartości funkcji $F$ w zależności od iteracji.
#
# Zapewnij powtarzalność wyników, ustawiając wartość odpowiedniego ziarna.

# %% [markdown]
# ![signal-2025-06-17-152231_002.jpeg](attachment:signal-2025-06-17-152231_002.jpeg)

# %% [markdown]
# Po znalezieniu gradientu funkcji kosztu zaimplementujemy algorytm największego spadku:
#  - inicjalizujemy punkty ścieżki i przeszkody
#  - Powatarzamy poniższe kroki ustaloną liczbę razy (w naszym wypadku 400):
#    - Obliczamy gradient $\nabla F$ w punkcie
#    - Wyznaczamy optymalną wartość kroku za pomocą metody złotego podziału
#    - Aktualizujemy punkty ścieżki
#
#
# Metoda złotego podziału opiera się na wykorzystaniu szczególnych własności funkcji unimodalnej, to jest takiej, która na danym przedziale posiada dokładnie jedno minimum.
#
# Ustalamy współczynnik $t$, $0<t<1$, a następnie powtarzamy poniższe operacje aż do uzyskania żądanej zbieżności:
#  - Obliczamy długość przedziałów $d =t(b-a)$,
#  - Przyjmujemy punkty $l = b-d$, $r=a+d$,
#  - Porównujemy wartości funkcji i decydujemy, czy minimum znajduje się w przedziale lewym, czy prawym i ustawiamy odpowiednio $a$ i $b$.

# %%
np.random.seed(1234)

n = 20
k = 50
x_0 = [0, 0]
x_n = [20, 20]
R = np.random.uniform(0, 20, (k, 2))
l_1 = l_2 = 1.0
eps = 10e-13
iterations = 400


# %%
def cost(X):
    A = 0
    for i in range(n + 1):
        for j in range(k):
            A += 1 / (eps + np.linalg.norm(X[i] - R[j]) ** 2)

    B = 0
    for i in range(n):
        B += np.linalg.norm(X[i + 1] - X[i]) ** 2

    return l_1 * A + l_2 * B


# %%
def grad_cost(X):
    gradient = np.zeros_like(X)

    for i in range(1, n):
        for j in range(k):
            diff = X[i] - R[j]
            norm_sq = np.linalg.norm(diff) ** 2
            gradient[i] += -2 * l_1 * diff / (eps + norm_sq) ** 2

        gradient[i] += -2 * l_2 * (X[i + 1] - X[i])
        gradient[i] += 2 * l_2 * (X[i] - X[i - 1])

    gradient[0] = 0
    gradient[-1] = 0

    return gradient


# %%
def line_search(X):
    costs = []
    copyX = copy.deepcopy(X)

    def helper(xs, grad):
        return lambda alpha: cost(xs + alpha * grad)

    for i in range(iterations):
        costs.append(cost(copyX))
        grad = grad_cost(copyX)

        alpha = golden(helper(copyX, -grad))

        copyX += alpha * (-grad)

        if i > 0 and abs(costs[-1] - costs[-2]) < eps:
            break

    return (copyX, costs)


# %%
XS = []
costs = []
results = []

for i in range(5):
    X = np.vstack((x_0, np.random.uniform(0, 20, (n - 1, 2)), x_n))

    XS.append(X)

    (result, res_cost) = line_search(X)
    results.append(result)
    costs.append(res_cost)

    next_seed = np.random.randint(0, 1000)
    np.random.seed(next_seed)


# %%
print("Koszty końcowe dla każdej próby:")
for i, cost_history in enumerate(costs):
    print(f"Próba {i + 1}: {cost_history[-1]:.6f}")


# %% [markdown]
# Znalezione przez algorytm optymalne ścieżki ilustrują poniższe wykresy:


# %%
def plot_robot_paths():
    """Rysuje ścieżki robota z wszystkich 5 prób"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes_flat = axes.flatten()

    for i in range(5):
        ax = axes_flat[i]

        ax.scatter(
            R[:, 0], R[:, 1], c="red", s=30, alpha=0.7, marker="x", label="Przeszkody"
        )

        initial_path = XS[i]
        ax.plot(
            initial_path[:, 0],
            initial_path[:, 1],
            "b--",
            alpha=0.5,
            linewidth=2,
            label="Ścieżka początkowa",
        )
        ax.scatter(initial_path[:, 0], initial_path[:, 1], c="blue", s=20, alpha=0.5)

        optimized_path = results[i]
        ax.plot(
            optimized_path[:, 0],
            optimized_path[:, 1],
            "g-",
            linewidth=3,
            label="Ścieżka zoptymalizowana",
        )
        ax.scatter(optimized_path[:, 0], optimized_path[:, 1], c="green", s=30)

        ax.scatter(
            *x_0,
            c="black",
            s=150,
            marker="s",
            label="Start",
            edgecolors="white",
            linewidth=2,
        )
        ax.scatter(
            *x_n,
            c="black",
            s=150,
            marker="^",
            label="Koniec",
            edgecolors="white",
            linewidth=2,
        )

        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"Próba {i + 1} - Koszt końcowy: {costs[i][-1]:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    ax = axes_flat[5]
    for i, cost_history in enumerate(costs):
        ax.plot(cost_history, label=f"Próba {i + 1}", linewidth=2)

    ax.set_xlabel("Iteracja")
    ax.set_ylabel("Wartość funkcji kosztu")
    ax.set_title("Historia zbieżności")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")

    plt.tight_layout()
    plt.show()


plot_robot_paths()


# %%
def plot_best_result():
    """Rysuje najlepszy wynik z większymi szczegółami"""

    final_costs = [cost_history[-1] for cost_history in costs]
    best_idx = np.argmin(final_costs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(
        R[:, 0], R[:, 1], c="red", s=50, alpha=0.7, marker="x", label="Przeszkody"
    )

    initial_path = XS[best_idx]
    ax1.plot(
        initial_path[:, 0],
        initial_path[:, 1],
        "b--",
        alpha=0.5,
        linewidth=2,
        label="Ścieżka początkowa",
    )

    best_path = results[best_idx]
    ax1.plot(
        best_path[:, 0], best_path[:, 1], "g-", linewidth=4, label="Najlepsza ścieżka"
    )
    ax1.scatter(
        best_path[:, 0],
        best_path[:, 1],
        c="green",
        s=50,
        edgecolors="black",
        linewidth=1,
    )

    for i, point in enumerate(best_path):
        ax1.annotate(
            str(i),
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax1.scatter(
        *x_0,
        c="blue",
        s=200,
        marker="s",
        label="Start",
        edgecolors="white",
        linewidth=3,
    )
    ax1.scatter(
        *x_n,
        c="red",
        s=200,
        marker="^",
        label="Koniec",
        edgecolors="white",
        linewidth=3,
    )

    ax1.set_xlim(-1, 21)
    ax1.set_ylim(-1, 21)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(
        f"Najlepsza ścieżka (Próba {best_idx + 1})\nKoszt końcowy: {final_costs[best_idx]:.4f}"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2.loglog(costs[best_idx], "g-", linewidth=3)
    ax2.set_xlabel("Iteracja")
    ax2.set_ylabel("Wartość funkcji kosztu")
    ax2.set_title(f"Zbieżność najlepszego wyniku")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()

    print(f"Najlepszy wynik z próby {best_idx + 1}")
    print(f"Koszt końcowy: {final_costs[best_idx]:.6f}")
    print(f"Liczba iteracji: {len(costs[best_idx])}")


plot_best_result()
