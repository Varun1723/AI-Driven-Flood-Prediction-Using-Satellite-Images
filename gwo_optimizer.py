import numpy as np
import random
from hyperparam_train import train_with_hyperparams

# Parameter ranges
param_ranges = {
    "lr": [0.0001, 0.01],
    "batch_size": [4, 16],
    "num_filters": [8, 64],
    "dropout": [0.0, 0.5],
}

# Fixed options for discrete values
batch_size_options = [4, 8, 16]
num_filters_options = [8, 16, 32, 64]

# GWO Settings
NUM_WOLVES = 6
NUM_ITERATIONS = 3


def initialize_wolves():
    wolves = []
    for _ in range(NUM_WOLVES):
        wolf = [
            random.uniform(*param_ranges["lr"]),
            random.choice(batch_size_options),
            random.choice(num_filters_options),
            random.uniform(*param_ranges["dropout"])
        ]
        wolves.append(wolf)
    return np.array(wolves)


def clip_params(params):
    lr = np.clip(params[0], *param_ranges["lr"])
    batch_size = min(batch_size_options, key=lambda x: abs(x - params[1]))
    num_filters = min(num_filters_options, key=lambda x: abs(x - params[2]))
    dropout = np.clip(params[3], *param_ranges["dropout"])
    return [lr, batch_size, num_filters, dropout]


def evaluate_wolf(wolf):
    param_dict = {
        "lr": round(wolf[0], 5),
        "batch_size": int(wolf[1]),
        "num_filters": int(wolf[2]),
        "dropout": round(wolf[3], 2),
        "epochs": 2  # fixed for faster evaluation
    }
    f1 = train_with_hyperparams(param_dict)
    return f1


def gwo():
    wolves = initialize_wolves()
    fitness = [evaluate_wolf(wolf) for wolf in wolves]

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Sort wolves by fitness
        sorted_indices = np.argsort(fitness)[::-1]
        wolves = wolves[sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]

        alpha, beta, delta = wolves[0], wolves[1], wolves[2]

        a = 2 - iteration * (2 / NUM_ITERATIONS)  # a decreases linearly from 2 to 0

        for i in range(NUM_WOLVES):
            for j in range(4):  # for each hyperparam
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i][j])
                X3 = delta[j] - A3 * D_delta

                wolves[i][j] = (X1 + X2 + X3) / 3

            wolves[i] = clip_params(wolves[i])

        fitness = [evaluate_wolf(wolf) for wolf in wolves]

    best_index = np.argmax(fitness)
    best_wolf = wolves[best_index]
    best_f1 = fitness[best_index]

    best_params = {
        "lr": round(best_wolf[0], 5),
        "batch_size": int(best_wolf[1]),
        "num_filters": int(best_wolf[2]),
        "dropout": round(best_wolf[3], 2),
        "epochs": 2
    }

    print("\nBest Parameters (GWO):", best_params)
    print("Best F1 Score:", best_f1)
    return best_params


if __name__ == "__main__":
    gwo()
