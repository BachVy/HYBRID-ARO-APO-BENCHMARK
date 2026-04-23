import numpy as np
import pandas as pd
import csv
import os
import random
from scipy.special import gamma
from benchmark import benchmark_functions, set_bounds, N, T, dim, benchmark_names, SpaceBound, Levy

# Initialize populations for each benchmark function
PopPos_init = {}
PopPos_init1 = {}  # For APO
PopPos_init2 = {}  # For PA1 (AOA)
PopPos_init3 = {}  # For PA2 (COA)
PopPos_init4 = {}  # For PA3 (GA)
PopPos_init5 = {}  # For PA4 (ARO_APO)
PopPos_init6 = {}  # For PA5 (PSO)

for idx in range(9):  # Chạy trên 9 hàm
    lb, ub = set_bounds(idx, dim)
    PopPos_init[idx] = np.random.rand(N, dim) * (ub - lb) + lb
    PopPos_init1[idx] = PopPos_init[idx].copy()  # For APO
    PopPos_init2[idx] = PopPos_init[idx].copy()  # For PA1 (AOA)
    PopPos_init3[idx] = PopPos_init[idx].copy()  # For PA2 (COA)
    PopPos_init4[idx] = PopPos_init[idx].copy()  # For PA3 (GA)
    PopPos_init5[idx] = PopPos_init[idx].copy()  # For PA4 (ARO_APO)
    PopPos_init6[idx] = PopPos_init[idx].copy()  # For PA5 (PSO)

# ARO Algorithm
def ARO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='aro_results.csv'):
    PopPos2 = PopPos_init.copy()
    fitness_drawARO = []
    pop_fit = np.array([fobj(PopPos2[i, :]) for i in range(N)])

    best_f = float('inf')
    best_x = None
    for i in range(N):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = PopPos2[i, :]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for it in range(T):
        direct1 = np.zeros((N, dim))
        direct2 = np.zeros((N, dim))
        theta = 2 * (1 - (it + 1) / T)
        for i in range(N):
            L = (np.e - np.exp((((it + 1) - 1) / T) ** 2)) * np.sin(2 * np.pi * np.random.rand())
            rd = np.floor(np.random.rand() * dim)
            rand_dim = np.random.permutation(dim)
            direct1[i, rand_dim[:int(rd)]] = 1
            c = direct1[i, :]
            R = L * c
            A = 2 * np.log(1 / np.random.rand()) * theta
            if A > 1:
                K = np.r_[0:i, i + 1:N]
                RandInd = K[np.random.randint(0, N - 1)]
                newPopPos = PopPos2[RandInd, :] + R * (PopPos2[i, :] - PopPos2[RandInd, :]) + \
                            0.5 * (0.05 + np.random.rand()) * np.random.randn()
            else:
                ttt = int(np.floor(np.random.rand() * dim))
                direct2[i, ttt] = 1
                gr = direct2[i, :]
                H = ((T - (it + 1) + 1) / T) * np.random.randn()
                b = PopPos2[i, :] + H * gr * PopPos2[i, :]
                newPopPos = PopPos2[i, :] + R * (np.random.rand() * b - PopPos2[i, :])

            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            if newPopFit < pop_fit[i]:
                pop_fit[i] = newPopFit
                PopPos2[i, :] = newPopPos

            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = PopPos2[i, :]

        fitness_drawARO.append(best_f)

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': best_f
            })

    return fitness_drawARO, PopPos2, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(fitness_drawARO)]

# APO Algorithm
def APO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='apo_results.csv'):
    if PopPos_init is None:
        PopPos = np.random.rand(N, dim) * (ub - lb) + lb
    else:
        PopPos = PopPos_init.copy()
    PopPos1 = PopPos.copy()
    fitness_drawAPO = []
    PopFit = np.array([fobj(PopPos1[i, :]) for i in range(N)])

    BestF = float('inf')
    BestX = None
    for i in range(N):
        if PopFit[i] <= BestF:
            BestF = PopFit[i]
            BestX = PopPos1[i, :]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for It in range(T):
        rand = np.random.rand()
        for i in range(N):
            theta1 = (1 - It / T)
            B = 2 * np.log(1 / rand) * theta1

            if B > 0.5:
                for _ in range(T):
                    K = [j for j in range(N) if j != i]
                    if not K:
                        break
                    RandInd = np.random.choice(K)
                    step1 = PopPos[i] - PopPos[RandInd]
                    if np.linalg.norm(step1) != 0:
                        break
                else:
                    continue
                R = 0.5 * (0.05 + rand) * np.random.normal(0, 1)
                Y = PopPos1[i, :] + 0.01 * Levy(dim) * step1 + R
                step2 = (rand - 0.5) * np.pi
                S = np.tan(step2)
                Z = Y * S

                Y = SpaceBound(Y, ub, lb)
                Z = SpaceBound(Z, ub, lb)
                NewPop = np.array([Y, Z])
                NewPopFit = np.array([fobj(Y), fobj(Z)])
                sorted_indexes = np.argsort(NewPopFit)
                newPopPos = NewPop[sorted_indexes[0], :]

            else:
                F = 0.5
                K = [j for j in range(N) if j != i]
                for _ in range(T):
                    available_indices = [j for j in range(N) if j != i]
                    if len(available_indices) < 3:
                        continue
                    RandInd = np.random.choice(available_indices, 3, replace=False)
                    step1 = PopPos[RandInd[1]] - PopPos[RandInd[2]]
                    if np.linalg.norm(step1) != 0:
                        break
                    else:
                        continue

                if rand < 0.5:
                    W = PopPos1[RandInd[0], :] + F * step1
                else:
                    W = PopPos1[RandInd[0], :] + F * 0.01 * Levy(dim) * step1
                f = 0.1 * (rand - 1) * ((T - It) / T)
                Y = (1 + f) * W
                for _ in range(T):
                    rand_leader_index1 = np.random.randint(0, N)
                    rand_leader_index2 = np.random.randint(0, N)
                    X_rand1 = PopPos1[rand_leader_index1, :]
                    X_rand2 = PopPos1[rand_leader_index2, :]
                    step2 = X_rand1 - X_rand2
                    if np.linalg.norm(step2) != 0 and not np.array_equal(X_rand1, X_rand2):
                        break
                    else:
                        continue
                Epsilon = np.random.uniform(0, 1)
                if rand < 0.5:
                    Z = PopPos1[i, :] + Epsilon * step2
                else:
                    Z = PopPos1[i, :] + F * 0.01 * Levy(dim) * step2

                W = SpaceBound(W, ub, lb)
                Y = SpaceBound(Y, ub, lb)
                Z = SpaceBound(Z, ub, lb)
                NewPop = np.array([W, Y, Z])
                NewPopFit = np.array([fobj(W), fobj(Y), fobj(Z)])
                sorted_indexes = np.argsort(NewPopFit)
                newPopPos = NewPop[sorted_indexes[0], :]

            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos1[i, :] = newPopPos

        for i in range(N):
            if PopFit[i] < BestF:
                BestF = PopFit[i]
                BestX = PopPos1[i, :]

        fitness_drawAPO.append(BestF)

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': It + 1,
                'Best_Fitness': BestF
            })

    return fitness_drawAPO, PopPos1, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(fitness_drawAPO)]

# AOA Algorithm (PA1)
def initial_variables(size, min_values, max_values, target_function, dim, start_init=None):
    min_values = np.array([min_values] * dim) if np.isscalar(min_values) else np.array(min_values)
    max_values = np.array([max_values] * dim) if np.isscalar(max_values) else np.array(max_values)
    
    if start_init is not None:
        start_init = np.atleast_2d(start_init)
        n_rows = size - start_init.shape[0]
        if n_rows > 0:
            rows = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = np.array([target_function(ind) for ind in start_init])
        population = np.hstack((start_init, fitness_values[:, np.newaxis]))
    else:
        population = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = np.array([target_function(ind) for ind in population])
        population = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

def update_population(population, elite, mu, moa, mop, min_values, max_values, target_function):
    e = 2.2204e-16
    dim = len(min_values)
    p = np.copy(population)
    r1 = np.random.rand(population.shape[0], dim)
    r2 = np.random.rand(population.shape[0], dim)
    r3 = np.random.rand(population.shape[0], dim)
    update_1 = np.where(r1 > moa, elite[:-1] / (mop + e) * ((max_values - min_values) * mu + min_values), elite[:-1])
    update_2 = np.where(r2 <= 0.5, update_1 * mop, update_1 - mop)
    update_3 = np.where(r3 > 0.5, update_2 - ((max_values - min_values) * mu + min_values), update_2 + ((max_values - min_values) * mu + min_values))
    up_pos = np.clip(update_3, min_values, max_values)
    for i in range(population.shape[0]):
        new_fitness = target_function(up_pos[i, :])
        if new_fitness < population[i, -1]:
            p[i, :-1] = up_pos[i, :]
            p[i, -1] = new_fitness
    return p

def AOA(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA1_aoa.csv'):
    population = initial_variables(N, lb, ub, fobj, dim, PopPos_init)
    best_f = float('inf')
    best_x = None
    elite = np.copy(population[population[:, -1].argsort()][0, :])
    if elite[-1] < best_f:
        best_f = elite[-1]
        best_x = elite[:-1].copy()

    curve = np.zeros(T)
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for t in range(T):
        alpha = (1.1 - t / T) * 0.5
        mu = 0.5
        moa = 0.2 + (t / T) * (0.9 - 0.2)
        mop = 1 - (t ** (1/6)) / (T ** (1/6))
        population = update_population(population, elite, mu, moa, mop, lb, ub, fobj)
        elite = np.copy(population[population[:, -1].argsort()][0, :])
        if elite[-1] < best_f:
            best_f = elite[-1]
            best_x = elite[:-1].copy()
        curve[t] = best_f
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': t + 1,
                'Best_Fitness': best_f
            })

    return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# COA Algorithm (PA2)
def COA(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA2_coa.csv'):
    n_groups = 5
    n_coy = int(N / n_groups)
    lb = np.array([lb] * dim) if np.isscalar(lb) else lb
    ub = np.array([ub] * dim) if np.isscalar(ub) else ub
    pop = PopPos_init if PopPos_init is not None else np.random.uniform(lb, ub, (N, dim))
    pop = np.clip(pop, lb, ub)
    pop_fit = np.array([fobj(ind) for ind in pop])
    population = np.hstack((pop, pop_fit[:, np.newaxis]))
    best_f = float('inf')
    best_x = None
    idx_best = np.argmin(population[:, -1])
    if population[idx_best, -1] < best_f:
        best_f = population[idx_best, -1]
        best_x = population[idx_best, :-1].copy()

    curve = np.zeros(T)
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for t in range(T):
        for g in range(n_groups):
            group = population[g * n_coy:(g + 1) * n_coy, :]
            alpha_idx = np.argmin(group[:, -1])
            alpha = group[alpha_idx, :-1]
            prob_leave = 0.1
            p_s = 0.1
            for i in range(n_coy):
                r1, r2 = np.random.rand(2)
                a = np.random.uniform(-1, 1, dim)
                b = np.random.uniform(-1, 1, dim)
                if np.random.rand() < 0.5:
                    new_coy = alpha + a * (group[np.random.randint(0, n_coy), :-1] - np.random.uniform(lb, ub))
                else:
                    new_coy = alpha + b * (group[np.random.randint(0, n_coy), :-1] - np.random.uniform(lb, ub))
                if np.random.rand() < prob_leave:
                    new_coy = lb + np.random.rand(dim) * (ub - lb)
                new_coy = np.clip(new_coy, lb, ub)
                new_fitness = fobj(new_coy)
                if new_fitness < group[i, -1]:
                    group[i, :-1] = new_coy
                    group[i, -1] = new_fitness
            if np.random.rand() < p_s:
                worst_idx = np.argmax(group[:, -1])
                group[worst_idx, :-1] = lb + np.random.rand(dim) * (ub - lb)
                group[worst_idx, -1] = fobj(group[worst_idx, :-1])
            population[g * n_coy:(g + 1) * n_coy, :] = group
        idx_best = np.argmin(population[:, -1])
        if population[idx_best, -1] < best_f:
            best_f = population[idx_best, -1]
            best_x = population[idx_best, :-1].copy()
        curve[t] = best_f
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': t + 1,
                'Best_Fitness': best_f
            })

    return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# GA Algorithm (PA3)
# def GA(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA3_ga.csv'):
#     mutation_rate = 0.1
#     population = PopPos_init if PopPos_init is not None else np.random.uniform(lb, ub, (N, dim))
#     pop_fit = np.array([fobj(ind) for ind in population])

#     best_f = float('inf')
#     best_x = None
#     for i in range(N):
#         if pop_fit[i] <= best_f:
#             best_f = pop_fit[i]
#             best_x = population[i, :].copy()

#     curve = np.zeros(T)
#     file_exists = os.path.isfile(csv_filename)
#     with open(csv_filename, 'a', newline='') as csvfile:
#         fieldnames = ['Function', 'Iteration', 'Best_Fitness']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()

#     for t in range(T):
#         # Select parents using tournament selection
#         tournament_size = 3
#         parents = []
#         for _ in range(N):
#             selected = np.random.choice(N, tournament_size, replace=False)
#             selected_fitness = [pop_fit[i] for i in selected]
#             best_index = selected[np.argmin(selected_fitness)]
#             parents.append(population[best_index].copy())

#         # Create next generation through crossover and mutation
#         next_population = []
#         for i in range(0, N, 2):  # Loop over N/2 pairs
#             parent1 = parents[i]
#             parent2 = parents[(i + 1) % N]  # Handle odd population size
#             crossover_point = np.random.randint(1, dim)
#             child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#             child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            
#             # Mutate children
#             for j in range(dim):
#                 if np.random.rand() < mutation_rate:
#                     child1[j] = np.random.uniform(lb[j] if isinstance(lb, np.ndarray) else lb,
#                                                   ub[j] if isinstance(ub, np.ndarray) else ub)
#                 if np.random.rand() < mutation_rate:
#                     child2[j] = np.random.uniform(lb[j] if isinstance(lb, np.ndarray) else lb,
#                                                   ub[j] if isinstance(ub, np.ndarray) else ub)
            
#             child1 = SpaceBound(child1, ub, lb)
#             child2 = SpaceBound(child2, ub, lb)
#             next_population.append(child1)
#             next_population.append(child2)

#         # Update population
#         population = np.array(next_population[:N])  # Ensure exactly N individuals
#         pop_fit = np.array([fobj(ind) for ind in population])

#         # Update best fitness and position
#         for i in range(N):
#             if pop_fit[i] < best_f:
#                 best_f = pop_fit[i]
#                 best_x = population[i, :].copy()

#         curve[t] = best_f
#         with open(csv_filename, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow({
#                 'Function': function_name,
#                 'Iteration': t + 1,
#                 'Best_Fitness': best_f
#             })

#     return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]
# GA Algorithm
def GA(N, T, lb, ub, dim, fobj, function_name, PopPos_init=None, csv_filename='PA3_ga.csv'):
    # Initialize population
    PopPos = PopPos_init if PopPos_init is not None and PopPos_init.shape == (N, dim) else \
             np.random.rand(N, dim) * (np.array([ub] * dim) - np.array([lb] * dim)) + np.array([lb] * dim)
    PopFit = np.array([fobj(ind) for ind in PopPos])

    # Initialize best solution
    GBestFit = float('inf')
    GBestPos = None
    for i in range(N):
        if PopFit[i] <= GBestFit:
            GBestFit = PopFit[i]
            GBestPos = PopPos[i].copy()

    # GA parameters
    mutation_rate = 0.1
    tournament_size = 3
    curve = np.zeros(T)

    # Initialize CSV file
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    # Main loop
    for it in range(T):
        # Tournament selection
        parents = []
        for _ in range(N):
            selected = random.sample(range(N), tournament_size)
            selected_fitness = [PopFit[i] for i in selected]
            best_index = selected[selected_fitness.index(min(selected_fitness))]
            parents.append(PopPos[best_index].copy())

        # Crossover and mutation
        next_population = []
        for i in range(0, N, 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % N]
            crossover_point = random.randint(1, dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            # Mutate children
            for j in range(dim):
                if random.random() < mutation_rate:
                    child1[j] = np.random.uniform(lb if np.isscalar(lb) else lb[j], ub if np.isscalar(ub) else ub[j])
                if random.random() < mutation_rate:
                    child2[j] = np.random.uniform(lb if np.isscalar(lb) else lb[j], ub if np.isscalar(ub) else ub[j])

            # Apply boundary constraints
            child1 = SpaceBound(child1, np.array([ub] * dim) if np.isscalar(ub) else ub, np.array([lb] * dim) if np.isscalar(lb) else lb)
            child2 = SpaceBound(child2, np.array([ub] * dim) if np.isscalar(ub) else ub, np.array([lb] * dim) if np.isscalar(lb) else lb)

            next_population.extend([child1, child2])

        # Update population
        PopPos = np.array(next_population)
        PopFit = np.array([fobj(ind) for ind in PopPos])

        # Update global best
        for i in range(N):
            if PopFit[i] < GBestFit:
                GBestFit = PopFit[i]
                GBestPos = PopPos[i].copy()

        curve[it] = GBestFit

        # Log to CSV
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': GBestFit
            })

    # Create fitness log (every 5 iterations to match original PA5 structure)
    fitness_log = [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

    return GBestFit, GBestPos, fitness_log

# ARO_APO Algorithm (PA4)
# def ARO_APO(N, T, lb, ub, dim, fobj, function_name, PopPos_ARO=None, csv_filename='PA4_aro_apo.csv'):
#     if PopPos_ARO is None:
#         PopPos_ARO = np.random.rand(N, dim) * (ub - lb) + lb
#     PopFit_ARO = np.array([fobj(ind) for ind in PopPos_ARO])
#     global_best_fitness = float('inf')
#     global_best_position = None
#     for i in range(N):
#         if PopFit_ARO[i] <= global_best_fitness:
#             global_best_fitness = PopFit_ARO[i]
#             global_best_position = PopPos_ARO[i].copy()

#     fitness_log = []
#     file_exists = os.path.isfile(csv_filename)
#     with open(csv_filename, 'a', newline='') as csvfile:
#         # fieldnames = ['Function', 'Iteration', 'Best_Fitness', 'Algorithm']
#         fieldnames = ['Function', 'Iteration', 'Best_Fitness']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()

#     for i in range(T):
#         if i % 2 == 0:
#             _, PopPos_ARO, curve = APO(N, 1, lb, ub, dim, fobj, function_name, PopPos_ARO, csv_filename)
#             # algorithm = 'APO'
#         else:
#             _, PopPos_ARO, curve = ARO(N, 1, lb, ub, dim, fobj, function_name, PopPos_ARO, csv_filename)
#             # algorithm = 'ARO'

#         PopFit_ARO = np.array([fobj(ind) for ind in PopPos_ARO])
#         best_idx = np.argmin(PopFit_ARO)
#         best_fitness = PopFit_ARO[best_idx]
#         if best_fitness < global_best_fitness:
#             global_best_fitness = best_fitness
#             global_best_position = PopPos_ARO[best_idx].copy()

#         fitness_log.append({
#             'Iteration': i + 1,
#             'Best_Fitness': global_best_fitness
#         })

#         with open(csv_filename, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow({
#                 'Function': function_name,
#                 'Iteration': i + 1,
#                 'Best_Fitness': global_best_fitness
#             })

#     return global_best_fitness, global_best_position, fitness_log
def ARO_APO(N, T, lb, ub, dim, fobj, function_name, PopPos_ARO=None, csv_filename='PA4_aro_apo.csv'):
    if PopPos_ARO is None:
        PopPos_ARO = np.random.rand(N, dim) * (ub - lb) + lb
    PopPos_APO = PopPos_ARO.copy()  # Khởi tạo PopPos_APO từ PopPos_ARO
    fitness_log = []
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    current_iteration = 0
    k = 10  # Số lần lặp cho mỗi thuật toán, tương tự như APO_ARO_share

    while current_iteration < T:
        # Chạy APO
        _, PopPos_APO, fitness_log_APO = APO(N, k, lb, ub, dim, fobj, function_name, PopPos_APO, csv_filename)
        # Chạy ARO
        _, PopPos_ARO, fitness_log_ARO = ARO(N, k, lb, ub, dim, fobj, function_name, PopPos_ARO, csv_filename)

        # Cập nhật số lần lặp cho fitness log
        for entry in fitness_log_APO:
            entry['Iteration'] += current_iteration
        for entry in fitness_log_ARO:
            entry['Iteration'] += current_iteration

        # Kết hợp fitness log
        fitness_log.extend(fitness_log_APO)
        fitness_log.extend(fitness_log_ARO)

        # Tìm chỉ số tốt nhất từ mỗi quần thể
        best_idx_APO = np.argmin([fobj(ind) for ind in PopPos_APO])
        best_idx_ARO = np.argmin([fobj(ind) for ind in PopPos_ARO])
        best_APO = PopPos_APO[best_idx_APO]
        best_ARO = PopPos_ARO[best_idx_ARO]

        # Chia sẻ giải pháp tốt nhất giữa hai quần thể
        PopPos_APO[np.random.randint(0, N)] = best_ARO
        PopPos_ARO[np.random.randint(0, N)] = best_APO

        current_iteration += k

    # Gộp và sắp xếp fitness log
    merged_log = {}
    for entry in fitness_log:
        it = entry['Iteration']
        if it not in merged_log or entry['Best_Fitness'] < merged_log[it]['Best_Fitness']:
            merged_log[it] = entry

    sorted_log = sorted(merged_log.values(), key=lambda x: x['Iteration'])

    # Ghi lại kết quả tốt nhất vào file CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for entry in sorted_log:
            writer.writerow({
                'Function': function_name,
                'Iteration': entry['Iteration'],
                'Best_Fitness': entry['Best_Fitness']
            })

    # Tìm vị trí và giá trị tốt nhất toàn cục
    global_best_fitness = float('inf')
    global_best_position = None
    for i in range(N):
        fitness = fobj(PopPos_ARO[i])
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = PopPos_ARO[i].copy()

    return global_best_fitness, global_best_position, sorted_log

def run_aro_apo():  # PA4
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = ARO_APO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init5[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA4_aro_apo.csv', index=False)

# PSO Algorithm (PA5)
def PSO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA5_pso.csv'):
    PopPos = PopPos_init if PopPos_init is not None else np.random.rand(N, dim) * (ub - lb) + lb
    PopVel = np.random.rand(N, dim) * (ub - lb)  # Initialize velocity
    PopFit = np.array([fobj(ind) for ind in PopPos])

    PBestPos = PopPos.copy()
    PBestFit = PopFit.copy()
    GBestFit = float('inf')
    GBestPos = None
    for i in range(N):
        if PopFit[i] <= GBestFit:
            GBestFit = PopFit[i]
            GBestPos = PopPos[i].copy()

    w = 0.9 - 0.5 * (np.arange(T) / T)  # Linearly decreasing inertia weight
    c1 = 2  # Cognitive parameter
    c2 = 2  # Social parameter
    curve = np.zeros(T)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for it in range(T):
        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            PopVel[i] = (w[it] * PopVel[i] + 
                         c1 * r1 * (PBestPos[i] - PopPos[i]) + 
                         c2 * r2 * (GBestPos - PopPos[i]))
            newPopPos = PopPos[i] + PopVel[i]
            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            
            # Update personal best
            if newPopFit < PBestFit[i]:
                PBestFit[i] = newPopFit
                PBestPos[i] = newPopPos.copy()
            
            # Update global best
            if newPopFit < GBestFit:
                GBestFit = newPopFit
                GBestPos = newPopPos.copy()
            
            PopPos[i] = newPopPos
            PopFit[i] = newPopFit

        curve[it] = GBestFit
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': GBestFit
            })

    return GBestFit, GBestPos, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# Run functions
def run_apo_adjusted():
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = APO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init1[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('apo.csv', index=False)

def run_aro_adjusted():
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = ARO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('aro.csv', index=False)

def run_aoa():  # PA1
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = AOA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init2[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA1_aoa.csv', index=False)

def run_coa():  # PA2
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = COA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init3[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA2_coa.csv', index=False)

# def run_ga():  # PA3
#     all_results = []
#     for idx in range(9):  # Chạy trên 9 hàm
#         lb, ub = set_bounds(idx, dim)
#         fobj = lambda x: benchmark_functions(x, idx)
#         _, _, fitness_log = GA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init4[idx])
#         df = pd.DataFrame(fitness_log)
#         df['Function'] = benchmark_names[idx]
#         all_results.append(df)
#     pd.concat(all_results).to_csv('PA3_ga.csv', index=False)
def run_ga():
    all_results = []
    # Initialize PopPos_init6 for 9 functions
    PopPos_init6 = [np.random.rand(N, dim) * (set_bounds(idx, dim)[1] - set_bounds(idx, dim)[0]) + set_bounds(idx, dim)[0] for idx in range(6)]
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = GA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init4[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA3_ga.csv', index=False)


# def run_aro_apo():  # PA4
#     all_results = []
#     for idx in range(9):  # Chạy trên 9 hàm
#         lb, ub = set_bounds(idx, dim)
#         fobj = lambda x: benchmark_functions(x, idx)
#         _, _, fitness_log = ARO_APO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init5[idx])
#         df = pd.DataFrame(fitness_log)
#         df['Function'] = benchmark_names[idx]
#         all_results.append(df)
#     pd.concat(all_results).to_csv('PA4_aro_apo.csv', index=False)

def run_pso():  # PA5
    all_results = []
    for idx in range(6):  # Chạy trên 9 hàm
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = PSO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init6[idx])
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA5_pso.csv', index=False)

if __name__ == "__main__":
    if os.path.exists('apo.csv'):
        os.remove('apo.csv')
    if os.path.exists('aro.csv'):
        os.remove('aro.csv')
    if os.path.exists('PA1_aoa.csv'):
        os.remove('PA1_aoa.csv')
    if os.path.exists('PA2_coa.csv'):
        os.remove('PA2_coa.csv')
    if os.path.exists('PA3_ga.csv'):
        os.remove('PA3_ga.csv')
    if os.path.exists('PA4_aro_apo.csv'):
        os.remove('PA4_aro_apo.csv')
    if os.path.exists('PA5_pso.csv'):
        os.remove('PA5_pso.csv')

    run_apo_adjusted()
    run_aro_adjusted()
    run_aoa()  # PA1
    run_coa()  # PA2
    run_ga()   # PA3
    run_aro_apo()  # PA4
    run_pso()  # PA5