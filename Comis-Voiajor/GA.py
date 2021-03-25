from random import random
from Chhromosome import Chromosome
from random import randint


class GA:
    def __init__(self, param=None, problParam=None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(0, self.__param['popSize']):
            c = Chromosome(self.__problParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problParam['function'](c.repres,self.__problParam['matrix'])

    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness < best.fitness):
                best = c
        return best

    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness > best.fitness):
                best = c
        return best

    def selection(self):
        # pr =[]
        # pr.append(0)
        # fr_tot=0
        # for i in range(len(self.__population)):
        #     fr_tot+=self.__population[i].fitness
        #
        # sum=0
        # for i in range(len(self.__population)):
        #     sum+=self.__population[i].fitness
        #     p = sum / fr_tot
        #     pr.append(p)
        #
        # pr.append(1)
        #
        # r = random()
        # r2 = random()
        # ok=True
        # i=0
        # c=-1
        # c2=-1
        # while ok:
        #     if r > pr[i] and r < pr[i+1]:
        #         c = i
        #     if r2 > pr[i] and r2 < pr[i+1]:
        #         c2 = i
        #
        #     if c!=-1 and c2!=-1:
        #         ok=False
        #     i+=1
        #
        # return c,c2
        # p1 = randint(0,self.__problParam['n']-1)
        # p2 = randint(0,self.__problParam['n']-1)
        p=[]
        for i in range(self.__param['k']):
            p.append(randint(0,self.__problParam['n']-1))

        min=self.__population[p[0]].fitness
        pp=p[0]
        for i in range(1,self.__param['k']):
            if min > self.__population[p[i]].fitness:
                min = self.__population[p[i]].fitness
                pp=p[i]

        return pp

    def oneGeneration(self):
        newPop = []
        print("................")
        for _ in range(self.__param['popSize']):
            # s1,s2=self.selection()
            # p1 = self.__population[s1]
            # p2 = self.__population[s2]
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
           # p=self.__problParam['function'](off.repres,self.__problParam['matrix'])
            print(off.repres,p)
        self.__population = newPop
        self.evaluation()

    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        print("................")
        for _ in range(self.__param['popSize'] - 1):
            # s1, s2 = self.selection()
            # p1 = self.__population[s1]
            # p2 = self.__population[s2]
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
           # print(off.repres,self.__problParam['function'](off.repres,self.__problParam['matrix']) )
        self.__population = newPop
        self.evaluation()

    def oneGenerationSteadyState(self):
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            off.fitness = self.__problParam['function'](off.repres)
            worst = self.worstChromosome()
            if (off.fitness < worst.fitness):
                worst = off
