import random
import copy
from hyperparam_train import train_with_hyperparams

# Define search space
HYPERPARAM_SPACE = {
    "lr": [0.0005, 0.001, 0.005],
    "batch_size": [4, 8, 16],
    "num_filters": [8, 16, 32],
    "dropout": [0.2, 0.3, 0.4]
}

POPULATION_SIZE = 6
GENERATIONS = 3
MUTATION_PROB = 0.2


def random_individual():
    return {
        "lr": random.choice(HYPERPARAM_SPACE["lr"]),
        "batch_size": random.choice(HYPERPARAM_SPACE["batch_size"]),
        "num_filters": random.choice(HYPERPARAM_SPACE["num_filters"]),
        "dropout": random.choice(HYPERPARAM_SPACE["dropout"]),
        "epochs": 2  # Keep it small for speed
    }


def mutate(individual):
    param = random.choice(list(HYPERPARAM_SPACE.keys()))
    individual[param] = random.choice(HYPERPARAM_SPACE[param])
    return individual


def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def genetic_algorithm():
    population = [random_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        print(f"\n=== Generation {generation + 1} ===")
        scored = []
        for indiv in population:
            fitness = train_with_hyperparams(indiv)
            scored.append((fitness, indiv))

        scored.sort(reverse=True, key=lambda x: x[0])
        print(f"Best F1: {scored[0][0]:.4f}, Params: {scored[0][1]}")

        top = scored[:POPULATION_SIZE // 2]
        new_population = [copy.deepcopy(indiv) for _, indiv in top]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(top, 2)
            child = crossover(parent1[1], parent2[1])
            if random.random() < MUTATION_PROB:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    print("\n✅ Genetic Algorithm Optimization Complete")
    print("Best config:", scored[0][1])


if __name__ == "__main__":
    genetic_algorithm()
