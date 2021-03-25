from random import randint
from Utils import generateNewValue


class Chromosome:


    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__repres = generateNewValue(0,problParam['n'])
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
        newrep = [0]*self.__problParam['n']
        for i in range(self.__problParam['n']):
            newrep[c.__repres[i]]=self.__repres[i]

        off = Chromosome(c.__problParam)
        off.repres = newrep

        return off


    def mutation(self):
        r = randint(0, len(self.__repres) - 1)
        r2=randint(0,len(self.__repres)-1)

        # if len(self.__repres)>10:
        #     r3 = randint(0, len(self.__repres) - 1)
        #     r4=randint(0,len(self.__repres)-1)
        #     self.__repres[r3], self.__repres[r4] = self.__repres[r4], self.__repres[r3]

        self.__repres[r],self.__repres[r2] = self.__repres[r2],self.__repres[r]




    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness