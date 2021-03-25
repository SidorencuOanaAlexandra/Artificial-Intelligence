import nx as nx
from math import sqrt

from GA import GA
import matplotlib.pyplot as plt
from numpy import random
import numpy as np


def fitness_function(v,matrix):
    d=0
    for i in range(len(v)-1):
        d+=matrix[v[i]][v[i+1]]

    d+=matrix[v[len(v)-1]][v[0]]

    return d

def scriere(matrix,lista_drum_minim,drum_minim,lista_drum_minim_x_y,drum_minim_2):
    f = open("out.txt", "w")
    f.write("%s\n" % len(matrix))
    f.writelines("%s," % lista_drum_minim[i] for i in range(len(lista_drum_minim) - 1))
    f.write("%s\n" % lista_drum_minim[len(lista_drum_minim) - 1])
    f.write("%s\n" % drum_minim)
    f.write("%s\n" % len(lista_drum_minim_x_y))
    f.writelines("%s," % lista_drum_minim_x_y[i] for i in range(len(lista_drum_minim_x_y) - 1))
    f.write("%s\n" % lista_drum_minim_x_y[len(lista_drum_minim_x_y) - 1])
    f.write("%s\n" % drum_minim_2)
    f.close()
    print(lista_drum_minim)
    print(drum_minim)
    print(lista_drum_minim_x_y)
    print(drum_minim_2)


def citire():
    f = open("date.txt", "r")
    n = int(f.readline())
    print(n)
    matrix = []
    for i in range(n):
        line = f.readline()
        matrix.append([int(x) for x in line.split(',')])

    f.close()
    return n,matrix

def citire2():
    fileName = "berlin52.txt"
    f = open(fileName, "r")

    # se omit primele 3 randuri
    for i in range(3):
        f.readline()

    # se citeste nr nodurilor
    n = int(f.readline().split(" ")[-1])

    # se omit urmatoarele 2 randuri
    for i in range(2):
        f.readline()

    # se initializeaza sirul de coordonate
    coord = {}
    for i in range(1, n+1):
        line = f.readline().split(" ")
        coord[float(line[0])] = (float(line[1]), float(line[2]))

    # se initializeaza matricea distantelor
    mat = []
    for i in range(n):
        mat.append([0] * n)

    # se populeaza matricea distantelor
    for i in range(n):
        for j in range(n):
            if i < j:
                e = sqrt((coord[i+1][0] - coord[j+1][0]) * 2 + (coord[i+1][1] - coord[j+1][1]) * 2)
                mat[i][j] = e
                mat[j][i] = e
    f.close()
    return n,mat

def main():

    n,matrix = citire()

    gaParam={'popSize':400,'nrGen':400,'k':150}

    propParam={'matrix':matrix,'n':n,'function':fitness_function}

    ga = GA(gaParam,propParam)
    ga.initialisation()
    ga.evaluation()

    best = ga.bestChromosome()
    fitnees = best.fitness

    x=[]
    y=[]
    for i in range(gaParam['nrGen']):

        bestSolution = ga.bestChromosome()
        # print(bestSolution)
        x.append(i)
        y.append(bestSolution.fitness)
        if bestSolution.fitness < fitnees:
            best = bestSolution
            fitnees = best.fitness
        # if i%2 == 0:
        #     ga.oneGeneration()
        # else:
        print(bestSolution)
        ga.oneGenerationElitism()

    print(best)
    print(fitnees)

    plt.plot(x,y)
    plt.xlabel('generatie')
    plt.ylabel('fitness')

    plt.show()

main()

l = []
lim2=11
ll = np.random.permutation(lim2)
print(ll)
for i in range(lim2):
    l.append(ll[i])
