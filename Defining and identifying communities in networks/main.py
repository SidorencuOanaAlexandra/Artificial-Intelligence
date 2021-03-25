import networkx as nx
from GA import GA
from utils import generareNouaSolutie


def modularity(communities, param):
    noNodes = param['n']
    mat = param['matrix']
    degrees = param['degrees']
    noEdges = param['edges']
    M = 2 * noEdges
    Q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if (communities[i] == communities[j]):
               Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M


def citire():
    f = open("date.txt", "r")
    n = int(f.readline())
    print(n)
    matrix = []
    for i in range(n):
        line = f.readline()
        matrix.append([int(x) for x in line.split(' ')])
    f.close()

    propParam={}

    propParam['n'] = n
    propParam['matrix'] = matrix

    Edges = 0
    degrees=[]
    for i in range(n):
        k=0
        for j in range(n):
            if matrix[i][j]==1:
                k+=1
            if j>i:
                Edges+=matrix[i][j]
        degrees.append(k)

    propParam['edges']=Edges
    propParam['degrees']=degrees

    return propParam

def citire2(filename):
    propParam = {}
    G = nx.read_gml(filename,label='id')
    propParam['n']=G.number_of_nodes()
    k = -1
    matrix = []
    degrees = []
    noEdges = 0
    for i in G.nodes:
        d = 0
        matrix.append([])
        k = k + 1
        for j in G.nodes:
            if j in [n for n in G.neighbors(i)]:
                matrix[k].append(1)
                if j > i:
                    noEdges = noEdges + 1
                d = d + 1
            else:
                matrix[k].append(0)
        degrees.append(d)

    propParam['matrix'] = matrix
    propParam['edges'] =noEdges
    propParam['degrees'] = degrees

    return propParam




def main():

    MIN=0
    MAX=0

    gaParam={'popSize':500,'nrGen':200}

    propParam = citire2("file.gml")

    for i in range(propParam['n']):
        print(propParam['matrix'][i])


    print(propParam['degrees'])
    print(propParam['edges'])

    propParam['function']=modularity


    ga = GA(gaParam,propParam)
    ga.initialisation()
    ga.evaluation()

    best = ga.bestChromosome()
    fitnees = best.fitness


    for i in range(gaParam['nrGen']):

        bestSolution = ga.bestChromosome()
        #print(bestSolution)
        if bestSolution.fitness > fitnees:
            best = bestSolution
            fitnees = best.fitness
        # if i%2 == 0:
        #     ga.oneGeneration()
        # else:
        print(bestSolution.fitness)
        ga.oneGenerationElitism()


    dict=[[]]
    x=best.repres
    v=[0]*propParam['n']
    k=1
    for i in range(propParam['n']):
        if v[x[i]]==0:
            v[x[i]]=k
            k+=1
            ii=i+1
            dict.append([ii])
        else:
            b=v[x[i]]
            ii=i+1
            dict[b].append(ii)

    print(k-1)

    for i in range(k):
        print(dict[i])



main()

