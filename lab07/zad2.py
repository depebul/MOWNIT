# %% [markdown]
# # Laboratorium 7 - Kwadratury adaptacyjne

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import seaborn as sns
from IPython.display import display

sns.set_style("darkgrid")


# %% [markdown]
# ## Zadanie 2.
# Powtórz obliczenia z poprzedniego oraz dzisiejszego laboratorium dla całek:
#
# (a)   $\int_0^1 \sqrt{x}\log xdx = - \frac{4}{9}$
#
#
# (b)   $\int_0^1 \Big(\frac{1}{(x-0.3)^2 + a} + \frac{1}{(x-0.9)^2 + b} - 6 \Big) dx$


# %%
def plot_error_func(f, real_value):
    a = 0 + np.finfo(np.double).eps
    b = 1
    m_vec = np.arange(1, 26)

    def quad_int(xs, f):
        s = 0
        for i in range(len(xs) - 1):
            dx = xs[i + 1] - xs[i]
            x = (xs[i] + xs[i + 1]) / 2
            s += f(x) * dx
        return s

    def trap(xs, f):
        return integrate.trapezoid([f(xs)], x=xs)

    def simpson(xs, f):
        return integrate.simpson(f(xs), x=xs)

    def trap_adaptive(epsrel):
        return integrate.quad_vec(
            f,
            a + np.finfo(np.double).eps,
            b,
            epsrel=epsrel,
            quadrature="trapezoid",
            full_output=True,
        )

    def gauss_kronrod(epsrel):
        return integrate.quad_vec(
            f, a, b, epsrel=epsrel, quadrature="gk21", full_output=True
        )

    values_list = []
    formula_list = [quad_int, trap, simpson]
    formula_list_2 = [trap_adaptive, gauss_kronrod]

    for formula in formula_list:
        values = []
        n_nodes = [2**m + 1 for m in m_vec]
        x_nodes = [np.linspace(a, b, n) for n in n_nodes]
        for nodes in x_nodes:
            value = formula(nodes, f)
            values.append(value)
        values_list.append(values)

    m_vec_gauss = np.arange(1, 15)
    n_vec_gauss = 2**m_vec_gauss + 1
    leggaus_values = [np.polynomial.legendre.leggauss(n) for n in n_vec_gauss]
    x_vec_gauss = [x_vec * 0.5 + 0.5 for x_vec, _ in leggaus_values]
    y_vec_gauss = [w_vec for _, w_vec in leggaus_values]
    gauss_values = [np.sum(f(x) * 0.5 * w) for x, w in zip(x_vec_gauss, y_vec_gauss)]
    gauss_errors = [np.abs((value - real_value) / real_value) for value in gauss_values]
    errors_list = []

    for i, values in enumerate(values_list):
        errors = []
        for j, value in enumerate(values):
            error = np.abs((value - real_value) / real_value)
            errors.append(error)
        errors_list.append(errors)
    x_arr = []
    for formula in formula_list_2:
        errors = []
        x_arr_1 = []
        for eps in [x for x in np.logspace(0, -14, 14)]:
            result = formula(eps)
            err = result[1]
            eval_count = result[-1].neval
            errors.append(err)
            x_arr_1.append(eval_count)
        errors_list.append(errors)
        x_arr.append(x_arr_1)
    plt.figure(figsize=(10, 6))

    plt.plot(2**m_vec + 1, errors_list[0], label="Rectangular")
    plt.plot(2**m_vec + 1, errors_list[1], label="Trapezoidal")
    plt.plot(2**m_vec + 1, errors_list[2], label="Simpson")
    plt.plot(2**m_vec_gauss + 1, gauss_errors, label="Gauss-Legendre")
    plt.plot(x_arr[0], errors_list[3], label="Trap Adaptive")
    plt.plot(x_arr[1], errors_list[4], label="Gauss Kronrod")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Relative error")
    plt.title("Integration error for different methods")
    plt.legend()
    plt.show()


# %% [markdown]
# ### Całka (a)


# %%
def fa(x):
    return np.sqrt(x) * np.log(x)


fa_real = -4 / 9

a = 0.001
b = 0.004


def fb(x):
    return 1 / ((x - 0.3) ** 2 + a) + 1 / ((x - 0.9) ** 2 + b) - 6


fb_real = (
    (1 / np.sqrt(a)) * (np.arctan((1 - 0.3) / np.sqrt(a)) + np.arctan(0.3 / np.sqrt(a)))
    + (1 / np.sqrt(b))
    * (np.arctan((1 - 0.9) / np.sqrt(b)) + np.arctan(0.9 / np.sqrt(b)))
    - 6
)

# plot_error_func(f, np.pi)
plot_error_func(fa, fa_real)


# %% [markdown]
# ### Całka (b)

# %%
plot_error_func(fb, fb_real)

# %% [markdown]
# ## Wnioski
# Ponownie, jak w poprzednim laboratorium, wykresy wyników są wyskalowane logarytmiczne, dlatego przedziały nieciągłości wykresów wskazują, że dla danej liczby węzłów błąd względny jest mniejszy niż precyzja użytych liczb w obliczeniach (w wynikach równa zeru).
#
# ::: {layout-ncol=3}
# ![Funkcja podcałkowa 1](fbdaae2dc966b14e9c2e42de63dae0bedcc6418f0b33e697979caf8bb63f9dff.png){width=250px}
#
# ![Funkcja podcałkowa 2A](7f7f998cefe3656f3e71c80a7e60fafc7422dcb8da9ac00c2d02c7a25bfb9e88.png){width=250px}
#
# ![Funkcja podcałkowa 2B](f39fdbb5c6607ef72040263e69aeccfa26e63603721c069550f073e3989f9120.png){width=250px}
# :::
#

# %% [markdown]
#
# Dla całki z zadania 1. metoda kwadratur adaptacyjnych trapezów miała większy błąd względny od pozostałych metod, podobnie było w przypadku całki 2 (b), natomiast całkując tą metodą funkcję 2 (a) otrzymamy dokładniejsze wyniki niż przy użyciu metod nieadaptacyjnych.
#
# Przy całkowaniu funkcji 1. metoda Gaussa-Kronroda osiągała minimalny błąd, pomijalny w porównaniu z precyzją obliczeń. W zadaniu 2 osiągała większą dokładność od większości metod nieadaptacyjnych, chociaż należy zauważyć, że w niektórych przypadkach metoda Simpsona, czy Gaussa-Legendre'a okazywała się dokładniejsza. Może to też wynikać z przewagi błędu numerycznego nad błędem metody.
#
# Porównując wykresy funkcji oraz skuteczność algorytmów można zauważyć, że metody adaptacyjne mają tym większą przewagę, im bardziej "skomplikowany" jest wykres funkcji.
#
# Podsumowując, metody adaptacyjne, a w szczególności metoda Gaussa-Kronroda dają dokładniejsze wyniki od metod nieadaptacyjnych, ale ich użycie w przypadku prostych funkcji może nie być opłacalne.
