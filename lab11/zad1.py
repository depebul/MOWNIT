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
# ## Zadanie 1.
#
# Rozwiąż ponownie problem predykcji typu nowotworu (laboratorium 2), używając metody spadku wzdłuż gradientu (ang. *gradient descent*). Stałą uczącą możesz wyznaczyć na podstawie najmniejszej i największej wartości własnej macierzy $A^T A$. Porównaj uzyskane rozwiązanie z metodą najmniejszych kwadratów, biorąc pod uwagę następujące kryteria:
#
# * Dokładność predykcji na zbiorze testowym
# * Teoretyczną złożoność obliczeniową
# * Czas obliczeń.

# %%
labels = pd.read_csv("breast-cancer.labels", header=None, names=["name"])
column_names = labels["name"].tolist()


train_data = pd.read_csv("breast-cancer-train.dat", header=None, names=column_names)
validate_data = pd.read_csv(
    "breast-cancer-validate.dat", header=None, names=column_names
)


# %%
# Reprezentacja liniowa
A_train_linear = train_data.drop(["patient ID", "Malignant/Benign"], axis=1).values
A_validate_linear = validate_data.drop(
    ["patient ID", "Malignant/Benign"], axis=1
).values

# Reprezentacja kwadratowa
selected_features = [
    "radius (mean)",
    "perimeter (mean)",
    "area (mean)",
    "symmetry (mean)",
]


def create_quadratic_features(data):
    quadratic_features = data[selected_features].copy()
    for feature in selected_features:
        quadratic_features[f"{feature}^2"] = data[feature] ** 2
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            feature1 = selected_features[i]
            feature2 = selected_features[j]
            quadratic_features[f"{feature1}*{feature2}"] = (
                data[feature1] * data[feature2]
            )
    return quadratic_features.values


A_train_quadratic = create_quadratic_features(train_data)
A_validate_quadratic = create_quadratic_features(validate_data)


# %%
# Wektor b dla zbioru treningowego
b_train = np.array(
    [[1, 0] if row == "M" else [0, 1] for row in train_data["Malignant/Benign"]]
)

# Wektor b dla zbioru walidacyjnego
b_validate = np.array(
    [[1, 0] if row == "M" else [0, 1] for row in validate_data["Malignant/Benign"]]
)


# %%
def classify(W, X):
    S = X @ W
    return S == np.max(S, axis=1, keepdims=True)


# %%
def calc_acc(P, T):
    accuracy = np.sum(P * T) / P.shape[0]
    return 100.0 * accuracy


def print_log(step, cost, train_acc, val_acc):
    log = (
        "Step {:3d}\tcost value: {:5.2f},\ttrain accuracy: {:5.2f},\t"
        "validation accuracy: {:5.2f}"
    )
    log = log.format(step, cost.item(), train_acc.item(), val_acc.item())

    print(log)


# %%
def mse(S, T):
    return 0.5 * np.mean((S - T) ** 2)


def grad_mse(X, S, T):
    n = X.shape[0]
    return (1.0 / n) * X.T @ (S - T)


# %%
def gd_fit(W0, X, T, X_val, T_val, lr=1.0, steps=100, log_every=5):
    n = X.shape[0]
    W = np.copy(W0)
    M = 0
    mu = 0.9

    stats = []

    for step in range(steps):
        S = X @ W
        cost_val = mse(S, T)

        cost_grad = grad_mse(X, S, T)
        M = mu * M - lr * cost_grad
        W = W + M

        P_train = classify(W, X)
        train_acc = calc_acc(P_train, T)

        P_val = classify(W, X_val)
        val_acc = calc_acc(P_val, T_val)

        stats.append((cost_val, train_acc, val_acc))
        if step == 0 or (step + 1) % log_every == 0:
            print_log(step + 1, cost_val, train_acc, val_acc)

    return W, stats


# %%
ATA_eigenvalues, _ = np.linalg.eig(A_train_linear.T @ A_train_linear)
lambda_min = np.min(ATA_eigenvalues)
lambda_max = np.max(ATA_eigenvalues)
condition_no = lambda_max / lambda_min


# %%
X = np.column_stack([A_train_linear, np.full(A_train_linear.shape[0], 1)])
T = b_train
X_val = np.column_stack([A_validate_linear, np.full(A_validate_linear.shape[0], 1)])
T_val = b_validate

print(X.shape, T.shape, X_val.shape, T_val.shape)


# %%
W0 = np.zeros((31, 2))

# %%
lr = 20 / (lambda_max)

print("Learning rate: ", lr)
W, stats = gd_fit(W0, X, T, X_val, T_val, lr=lr, steps=1000, log_every=50)


cost = [t[0] for t in stats]
t_acc = [t[1] for t in stats]
v_acc = [t[2] for t in stats]

plt.figure(figsize=(10, 6))

plt.plot(t_acc, label="Train accuracy")
plt.plot(v_acc, label="Validation accuracy")

plt.xlabel("Step")
plt.ylabel("%")
plt.title("Gradient descent accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

plt.plot(cost, label="Cost")

plt.xlabel("Step")
plt.title("Gradient descent cost")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Metodą spadku wzdłuż gradientu udało się uzyskać dokładność predykcji na zbiorze walidacyjnym na poziomie 93%. Jest to nieco mniejszy wynik o dokładności predykcji metodą najmniejszych kwadratów, która wynosiła 97%.
#
# Porównując czasy wykonania, należy zwrócić uwagę na znaczną przewagę metody najmniejszych kwadratów, przy której czas rozwiązywania układu równań funkcją biblioteczną numpy.linalg.solve był krótszy niż 0.1 s, natomiast znajdywanie rozwiązania metodą gradient descent trwało 1.9 s.
#
# Wynika to bezpośrednio ze złożoności obliczeniowej, która dla metody najmniejszych kwadratów wynosi $O(n^{3})$, gdzie $n$ to liczba parametrów, natomiast dla gradient descent - $O(ndk)$, gdzie $n$ to liczba iteracji, $d$ to liczba parametrów, a $k$ to liczba punktów danych. Stąd gradient descent może sprawdzać się tylko dla dużych liczb parametrów.
