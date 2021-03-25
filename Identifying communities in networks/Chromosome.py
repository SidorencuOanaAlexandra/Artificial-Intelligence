from random import randint
from utils import generateNewValue, generareNouaSolutie


class Chromosome:


    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__repres = generareNouaSolutie(self.__problParam)
        self.__fitness = 0.0


    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l=[]):
        self.__repres = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        r = randint(0, len(self.__repres) - 1)
        val = self.__repres[r]
        newrep = []
        for i in range(len(self.__repres)):
            if self.__repres[i] == val:
                newrep.append(val)
            else:
                newrep.append(c.__repres[i])
        offspring = Chromosome(c.__problParam)
        offspring.repres = newrep
        return offspring

    def mutation(self):
        r = randint(0, len(self.__repres) - 1)
        # if self.__repres[r]<=2:
        #     r2 = randint(r,len(self.__repres)-1)
        #     self.__repres[r] = self.__repres[r2]
        # else:
        #     r2 = randint(1,self.__repres[r])
        #     self.__repres[r] = r2
        r2 = randint(0, len(self.__repres) - 1)
        self.__repres[r] = self.__repres[r2]



    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness