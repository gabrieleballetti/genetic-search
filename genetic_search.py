import random
import datetime
import os
import itertools
from sympy import floor
from genetic_search_utils import (
    h_star_vector_of_cartesian_product_from_h_star_vectors,
    _is_unimodal,
)


# HELPER FUNCTIONS FOR A SINGLE VECTOR (h, h)


def _generate_all_mutations_single(h):
    """Generate all possible mutations of h."""

    mutations = set()

    for i, _ in enumerate(h):
        # remove h[i]
        h_removed = [h[j] for j in range(len(h)) if j != i]
        mutations.add(tuple(h_removed))

        # # add one entry after h[i]
        h_next = h[i + 1] if i < len(h) - 1 else 0
        if h[i] > h_next:
            for extra in range(h_next, h[i] + 1):
                h_extra = (
                    [h[j] for j in range(i + 1)]
                    + [extra]
                    + [h[j] for j in range(i + 1, len(h))]
                )
                mutations.add(tuple(h_extra))
        else:
            for extra in range(h[i], h_next + 1):
                h_extra = (
                    [h[j] for j in range(i + 1)]
                    + [extra]
                    + [h[j] for j in range(i + 1, len(h))]
                )
                mutations.add(tuple(h_extra))

        if i == 0:
            # do not change h[0]
            continue

        # increase h[i] by 1
        h_plus = [hi for hi in h]
        h_plus[i] += 1
        mutations.add(tuple(h_plus))

        # decrease h[i] by 1
        if h[i] > 0:
            h_minus = [hi for hi in h]
            h_minus[i] -= 1
            mutations.add(tuple(h_minus))

    return mutations


def _generate_all_crossovers_single(h1, h2):
    cs = set()

    # in case the lengths do not match, add both and return
    if len(h1) <= len(h2):
        g1 = h1
        g2 = h2
    elif len(h1) > len(h2):
        g1 = h2
        g2 = h1

    while len(g1) < len(g2):
        g1 = g1 + (0,)

    # in case the lengths match generate all intermediate vectors
    ranges = [range(min(g1[i], g2[i]), max(g1[i], g2[i]) + 1) for i in range(len(g1))]

    # generate all possible combinations of the diffs
    for h in itertools.product(*ranges):
        cs.add(h)

    return cs


def _remove_unfit_single(mutants):
    to_remove = set()
    for h in mutants:
        if not _is_unimodal(h):
            to_remove.add(h)

    for h in to_remove:
        mutants.remove(h)

    return mutants


def _calculate_fitness_function_single(h):
    # lower is better
    score = 0
    ineqs = 0

    # d and s are shorthand for dimension and degree respectively
    d = len(h) - 1
    for s in range(d, 0, -1):
        if h[s] != 0:
            break

    score += max(0, h[d] - h[1])
    ineqs += 1

    for i in range(2, floor(d / 2) + 1):
        score += max(
            0,
            sum(h[k] for k in range(d - i + 1, d)) - sum(h[k] for k in range(2, i + 1)),
        )
        ineqs += 1

    for i in range(0, floor(s / 2) + 1):
        score += max(
            0,
            sum(h[k] for k in range(0, i + 1)) - sum(h[k] for k in range(s - i, s + 1)),
        )
        ineqs += 1

    if s == d:
        for i in range(1, d - 1):
            score += max(0, h[1], h[i])
            ineqs += 1
    else:
        for i in range(1, d):
            score += max(0, h[0] + h[1] - sum(h[k] for k in range(i - d + s, i + 1)))
            ineqs += 1

    score = float(score) / ineqs

    return score


def _h_cartesian_prod_from_single_h(individual):
    return h_star_vector_of_cartesian_product_from_h_star_vectors(
        individual, individual
    )


# HELPER FUNCTIONS FOR TWO VECTORS (h1, h2)


def _generate_all_mutations_double(couple):
    mutations = set()

    m1 = _generate_all_mutations_single(couple[0])
    m1 = _remove_unfit_single(m1)
    m2 = _generate_all_mutations_single(couple[1])
    m2 = _remove_unfit_single(m2)
    for h1 in m1:
        for h2 in m2:
            mutations.add((h1, h2))

    return mutations


def _generate_all_crossovers_double(couple1, couple2):
    crossovers = set()

    c1 = _generate_all_crossovers_single(couple1[0], couple2[0])
    c1 = _remove_unfit_single(c1)
    c2 = _generate_all_crossovers_single(couple1[1], couple2[1])
    c2 = _remove_unfit_single(c2)

    for h1 in c1:
        for h2 in c2:
            crossovers.add((h1, h2))

    return crossovers


def _remove_unfit_double(mutants):
    to_remove = set()
    for couple in mutants:
        if not _is_unimodal(couple[0]) or not _is_unimodal(couple[1]):
            to_remove.add(couple)

    for h in to_remove:
        mutants.remove(h)

    return mutants


def _calculate_fitness_function_double(couple):
    ff1 = _calculate_fitness_function_single(couple[0])
    ff2 = _calculate_fitness_function_single(couple[1])

    return ff1 + ff2


def _h_cartesian_prod_from_two_hs(individual):
    return h_star_vector_of_cartesian_product_from_h_star_vectors(
        individual[0], individual[1]
    )


def genetic_search(
    parameters,
):
    """ """
    # read parameters
    population = parameters["starting_population"]
    generate_all_mutations = parameters["generate_all_mutations"]
    generate_all_crossovers = parameters["generate_all_crossovers"]
    remove_unfit = parameters["remove_unfit"]
    fitness_function = parameters["fitness_function"]
    h_cartesian_product = parameters["h_cartesian_product"]
    p_mutation = parameters["p_mutation"]
    max_pop = parameters["max_pop"]
    min_pop = parameters["min_pop"]
    penalty_factor = parameters["penalty_factor"]

    # create logging folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = ".logs"
    os.makedirs(folder_name, exist_ok=True)
    filename = f"{folder_name}/{timestamp}_search.txt"

    # init the score dictionary, will be updated at each epoch with penalties
    scores = {}
    for h in population:
        scores[h] = fitness_function(h)

    generation = 0

    # log the first values
    with open(filename, "a") as f:
        f.write(f"{generation}\t{h}\t{scores[h]}\t{scores[h]}\n")

    while True:
        # start a new generation
        generation += 1

        # generate new mutations
        n_mutation_attempts = 0
        while len(population) < max_pop:
            n_mutation_attempts += 1

            # pick between mutation and crossover depending on probabilities
            if random.random() < p_mutation:
                # attempt mutation

                # pick an h randomly, but weighted on the (reciprocal of) fitness score
                total_score = sum([1 / scores[h] for h in population])
                probs = [1 / (scores[h] * total_score) for h in population]
                h = random.choices(population, weights=probs, k=1)[0]

                # generate all possible mutations for h
                possible_mutations = set(generate_all_mutations(h))

                # remove all already existing mutations and unfit mutations
                possible_mutations = possible_mutations.difference(set(population))
                possible_mutations = remove_unfit(possible_mutations)

                # if there are no possible mutations, penalize h and continue
                if len(possible_mutations) == 0:
                    # scores[h] *= penalty_factor
                    continue

                # pick a mutation randomly and add it to the population
                h_new = random.choice(tuple(possible_mutations))
            else:
                # attempt crossover

                # pick two h's randomly, but weighted on the (reciprocal of) fitness score
                total_score = sum([1 / scores[h] for h in population])
                probs = [1 / (scores[h] * total_score) for h in population]
                h1, h2 = random.choices(population, weights=probs, k=2)

                # generate all possible crossovers for h1 and h2
                possible_crossovers = generate_all_crossovers(h1, h2)

                # remove all already existing crossovers and unfit crossovers
                possible_crossovers = possible_crossovers.difference(set(population))
                possible_crossovers = remove_unfit(possible_crossovers)

                # if there are no possible crossovers, penalize h1 and h2 and continue
                # (only do this if len(h1) == len(h2))
                if len(possible_crossovers) == 0:
                    # if len(h1) == len(h2):
                    # scores[h1] *= penalty_factor
                    # scores[h2] *= penalty_factor
                    continue

                # pick a crossover randomly and add it to the population
                h_new = random.choice(tuple(possible_crossovers))

            # if Pi(h,h) is unimodal, penalize h and continue
            h_product = h_cartesian_product(h_new)

            if _is_unimodal(h_product):
                # scores[h] *= penalty_factor
                continue

            # if we get here, h is a valid mutation
            score_h_new = fitness_function(h_new)

            # check if h is a solution, if so, write to the logs and return it
            if score_h_new == 0:
                print(f"Found eligible mutation: {h_new}")
                avg_score = sum(sorted(scores.values())[:min_pop]) / min_pop
                with open(filename, "a") as f:
                    f.write(f"{generation}\t{h_new}\t{score_h_new}\t{avg_score}\n")
                return h_new

            # add h_new to the population
            population.append(h_new)
            if h_new not in scores:
                scores[h_new] = score_h_new

        # now start the selection process, pick the best min_pop individuals based on
        # their fitness function score (lower is better)
        selected = sorted(population, key=lambda h: scores[h])[:min_pop]

        # find the best of the elected for logging purposes
        best_h_score = min([scores[h] for h in selected])
        best_h = [h for h in selected if scores[h] == best_h_score][0]

        # find the average score
        avg_score = sum([scores[h] for h in selected]) / len(selected)

        # log generation summary to console
        print(f"Generation {generation} summary:")
        print(f"Mutation attempt: {n_mutation_attempts}")
        print(f"Average score: {avg_score}")
        print(f"Best score: {best_h_score}")
        print(f"Best h: {best_h}")
        print()

        # log generation summary to file
        with open(filename, "a") as f:
            f.write(f"{generation}\t{best_h}\t{best_h_score}\t{avg_score}\n")

        # check if h is a solution, if so, write to the logs and return it
        if best_h_score == 0:
            print(f"Found eligible mutation: {h_new}")
            return best_h

        # multiply all scores by the penality factor (aging process)
        for h in population:
            scores[h] *= penalty_factor

        population = list(selected)


def expand_solutions(solutions, parameters, n_desired_solutions=10):
    """
    Expand - via mutations - a set of solutions until a desired number is reached.
    """
    generate_all_mutations = parameters["generate_all_mutations"]
    remove_unfit = parameters["remove_unfit"]
    fitness_function = parameters["fitness_function"]

    previous_hs = solutions.copy()

    while len(previous_hs) > 0 and len(solutions) < n_desired_solutions:
        new_hs = set()
        all_mutations = set()
        for h in tuple(previous_hs):
            if len(h) > 29:
                continue
            all_mutations = all_mutations.union(set(generate_all_mutations(h)))
        all_mutations -= solutions
        all_mutations -= previous_hs
        all_mutations = remove_unfit(all_mutations)
        all_mutations = [a for a in all_mutations if fitness_function(a) == 0]
        i = 0
        for h_mutant in all_mutations:
            i += 1
            h_product = h_star_vector_of_cartesian_product_from_h_star_vectors(
                h_mutant, h_mutant
            )

            if _is_unimodal(h_product):
                continue

            new_hs.add(h_mutant)

        print(f"Found {len(new_hs)} new h's")
        solutions.update(previous_hs)
        previous_hs = new_hs

    return solutions


if __name__ == "__main__":
    # parameters for solutions of the form (h, h)
    parameters_single = {
        "starting_population": [(1, 1, 1, 1, 1, 6)],
        "generate_all_mutations": _generate_all_mutations_single,
        "generate_all_crossovers": _generate_all_crossovers_single,
        "remove_unfit": _remove_unfit_single,
        "fitness_function": _calculate_fitness_function_single,
        "h_cartesian_product": _h_cartesian_prod_from_single_h,
        "p_mutation": 0.5,
        "max_pop": 30,
        "min_pop": 5,
        "penalty_factor": 1.05,
    }

    # parameters for solutions of the form (h1, h2)
    parameters_double = {
        "starting_population": [((1, 1, 1, 1, 1, 6), (1, 1, 1, 1, 2, 5))],
        "generate_all_mutations": _generate_all_mutations_double,
        "generate_all_crossovers": _generate_all_crossovers_double,
        "remove_unfit": _remove_unfit_double,
        "fitness_function": _calculate_fitness_function_double,
        "h_cartesian_product": _h_cartesian_prod_from_two_hs,
        "p_mutation": 0.5,
        "max_pop": 30,
        "min_pop": 5,
        "penalty_factor": 1.05,
    }

    solutions = genetic_search(parameters=parameters_single)

    # Uncomment to generate extra solutions from the one found
    # solutions = expand_solutions(set((solutions,)), parameters_single, 10)
    print(solutions)
