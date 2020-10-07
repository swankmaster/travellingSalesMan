import pandas as pd
import numpy as np
import random
from scipy.spatial import distance

global df
df = pd.read_csv("Cities.csv")


def make_graph():
    global df
    sz = len(df.NODE)
    x = df.X
    y = df.Y
    cities = list(df.NODE)
    dist = []
    for i in range(sz):
        row = []
        for j in range(sz):
            a = (x[i], y[i])
            b = (x[j], y[j])
            row.append(distance.euclidean(a, b))

        dist.append(row)
    graph = pd.DataFrame(data=dist, index=cities, columns=cities)
    return graph


def init_population(pop_size, nodes):
    population = []
    while pop_size > 0:
        random.shuffle(nodes)
        population.append(nodes)
        pop_size = pop_size - 1
    return population


def fitness(individual, graph):
    dist = []
    cols = list(graph.columns)
    dist.append(graph[individual[0]][individual[-1]])
    for i in range(len(individual) - 1):
        dist.append(graph[individual[i]][individual[i + 1]])
    fit = sum(dist)
    return fit


def inherit_2(child, i_2, split):
    N = len(i_2)
    child_ix = split
    parent_ix = split
    while -1 in child:
        if parent_ix >= N:
            parent_ix = 0
        if child_ix >= N:
            child_ix = 0
        if i_2[parent_ix] not in child:
            child[child_ix] = i_2[parent_ix]
            child_ix = child_ix + 1
        parent_ix = parent_ix + 1
    return child


def crossover(par_1, par_2,population):
    global df
    N = len(df)
    p1 = population[par_1]
    p2 = population[par_2]
    point_1 = random.randint(0, N)
    point_2 = random.randint(0, N)
    child1 = [-1 for i in range(len(p1))]
    child2 = [-1 for i in range(len(p1))]
    if point_1 == point_2:
        crossover(par_1, par_2,population)
    elif point_1 > point_2:
        temp = point_2
        point_2 = point_1
        point_1 = temp
    h_t = random.randint(1, 2)
    if h_t == 1:
        child1[point_1:point_2 + 1] = p1[point_1:point_2 + 1]
        child2[point_1:point_2 + 1] = p2[point_1:point_2 + 1]
        child1 = inherit_2(child1, p2, point_2 + 1)
        child2 = inherit_2(child2, p1, point_2 + 1)
    else:
        child1[point_1:point_2 + 1] = p2[point_1:point_2 + 1]
        child2[point_1:point_2 + 1] = p2[point_1:point_2 + 1]
        child2 = inherit_2(child2, p2, point_2 + 1)
        child1 = inherit_2(child1, p1, point_2 + 1)
    return child1, child2


def mutate(child):
    mutation_p = np.random.uniform(0.05, (1 / 3))
    increment = int(len(child) * mutation_p)
    stop = len(child) - increment
    mutate_point = random.randint(0, stop)
    if mutate_point == stop:
        snip = child[mutate_point:]
        random.shuffle(snip)
        child[mutate_point:] = snip
    else:
        snip = child[mutate_point:increment + mutate_point]
        random.shuffle(snip)
        child[mutate_point:increment + mutate_point] = snip
    return child


def win(df, p):
    val_p = np.random.uniform(0, 1)
    prob = []
    for i in range(len(df)):
        prob_i = (1 - p) ** i
        prob.append(prob_i * p)
    index = 0
    inc = 1 - sum(prob)
    prob[0] = prob[0] + inc
    # print(val_p)

    while index < len(prob):
        # print(index)
        val_p = val_p - prob[index]
        # print(prob[index])
        # print(val_p)
        if val_p < 0:
            break
        if index == len(prob) - 1:
            break
        else:
            index = index + 1
    return index


def tournament(num_ind, population, p, fit,graph):

    if len(fit) < 1:
        fit_list = [fitness(list(i)) for i in population]
    else:
        fit_list = fit
    individuals = [random.randint(0, len(fit_list) - 1) for i in range(num_ind)]

    fit_i = [fit_list[i] for i in individuals]
    tour_df = pd.DataFrame(data=individuals, columns=['individuals'])
    tour_df['fitness'] = fit_i
    tour_df = tour_df.sort_values(by='fitness', ascending=True)
    # prob_i = []
    # for i in range(len(tour_df)):
    #     prob = (1 - p) ** i
    #     prob_i.append(prob * p)
    # print(prob_i)
    # tour_df['probability'] = prob_it
    tour_winner1 = win(tour_df, p)
    ix_1 = list(tour_df.individuals)[tour_winner1]
    tour_df = tour_df.drop(tour_winner1)
    tour_winner2 = win(tour_df, p)
    # print(tour_df)
    ix_2 = list(tour_df.individuals)[tour_winner2]
    # print(ix_1,ix_2)
    (child_1,child_2) = crossover(ix_1, ix_2, population)
    # print(child_1)
    # print(child_2)
    child_1 = mutate(child_1)
    child_2 = mutate(child_2)
    # print(child_1)
    # print(child_2)
    child_fit1 = fitness(child_1,graph)
    child_fit2 = fitness(child_2,graph)
    # child = child_1
    # print(child_fit1)
    # print(child_fit2)
    if child_fit1 < child_fit2:
        child = child_1
        child_fit = child_fit1
    else:
        child = child_2
        child_fit = child_fit2

    ix_replace = list(tour_df.individuals)[-1]

        # print(fit_list)
        # print(child)
        # print("Child", child_fit)
        # replace = pd.DataFrame(data=fit_list, columns=['fitness'])
        # replace = replace.sort_values(by='fitness', ascending=False)
        # print(ix_replace)
        # print("Replace ", list(tour_df.fitness)[-1])
    # fit_tour = list(tour_df.fitness)
    # if fit_tour[-1] > child_fit:
    population[ix_replace] = np.array(child)
    fit_list[ix_replace] = child_fit
    return population, fit_list


def run(size_of_pop, nodes, p,size):
    epochs = 0
    pop = init_population(size_of_pop, nodes)
    graph = make_graph()
    fit = [fitness(i,graph) for i in pop]
    size_of_tourny = size
    gen = 1000000
    sol = ""
    # graph = make_graph()
    while epochs < gen:
        # minfit = fit[min_fit(fit)]
        # factor = num_queens - minfit
        # formula = (size_of_pop * 0.124) - factor
        # size_of_tourny = max(formula, 7)
        (pop, fit) = tournament(size_of_tourny, pop, p, fit,graph)
        epochs += 2
        fitDF = pd.DataFrame(data = fit, columns = ["fitness"])
        bestFit = fitDF.sort_values(by = 'fitness', ascending=True).index[0]
        sol = pop[bestFit]
        if epochs%200 == 0:
            print("Iteration:", epochs)
            print("Solution", sol)
            print("Best Fitness:",min(fit))
        if 0 in fit:
            sol = "solution"
            break

    fit_df = pd.DataFrame(data=fit, columns=['fitness'])
    ix_best = fit_df.sort_values(by='fitness', ascending=True).index[0]

    return

node = list(df.NODE)
run(200,node,0.4,20)
#travellingSalesMan
