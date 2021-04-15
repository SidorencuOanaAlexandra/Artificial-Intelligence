from cmath import sqrt
from random import randint
from random import random

import matplotlib.pyplot as plt



class Furnica:

    def __init__(self,drum,fitt):
        self.__drum = drum
        self.__fitt = fitt

    def get_drum(self):
        return self.__drum

    def get_fitt(self):
        return self.__fitt

    def set_drum(self,l):
        self.__drum = l

    def set_fitt(self,l):
        self.__fitt = l

    def __str__(self):
        return '\nDrum: ' + str(self.__drum) + ' has fit: ' + str(self.__fitt)


def read_data(filename):
    f = open(filename,'r')
    n = int(f.readline())
    matrix = []
    for i in range(n):
        line = f.readline()
        matrix.append([float (x) for x in line.split(',')])

    return n,matrix

def create_matrix_f(N):
    matrix_f =[]
    for i in range(N):
        l=[]
        for j in range(N):
            if i!=j:
                l.append(1)
            else:

                l.append(0)
        matrix_f.append(l)

    return matrix_f

def alege_nod(list_d,list_f,propParam,v):

    form = []
    fr_tot = 0
    for i in range(len(list_f)):
        if v[i]==0:
            l = (list_f[i]**propParam['alfa'])*((1/list_d[i])**propParam['beta'])
            fr_tot +=l
            form.append([l,i])

    r = random()
    suma = 0
    i=-1
    retine = -1
    while r>suma/fr_tot:
        i+=1
        suma+=form[i][0]
        retine = form[i][1]

    return retine

def create_frum(matrix_d,matrix_f,propParam):

    v = [0]*propParam['N']
    x = randint(0,propParam['N']-1)
    d = [x]
    v[x] = 1
    dist = 0
    for i in range(propParam['N']-1):
        y=alege_nod(matrix_d[x],matrix_f[x],propParam,v)
        matrix_f[x][y]+=1
        d.append(y)
        v[y]=1
        dist+=matrix_d[x][y]
        x=y
    dist+=matrix_d[d[propParam['N']-1]][d[0]]

    return dist,d


def create_pop(matrix_d,matrix_f,propParam):
    best_furnica = Furnica([],1000000)


    for i in range(propParam['sizeP']):
        dist,d = create_frum(matrix_d,matrix_f,propParam)
        if dist<best_furnica.get_fitt():
            best_furnica.set_drum(d)
            best_furnica.set_fitt(dist)

    return best_furnica


def modifica(matrix,N):
    x=randint(0,N-1)
    y=randint(x,N-1)
    r=randint(1,100)
    matrix[x][y]=r
    matrix[y][x]=r

def main():
    N,matrix = read_data("date.in")
    matrix_f = create_matrix_f(N)
    param_pr={'nrPop':500,'sizeP':100,'alfa':1,'beta':1.2,'N':N}

    matrix2 = matrix

    best_furnica = []
    best_fir = 0

    x = []
    y = []

    best_furnica = Furnica([], 1000000)
    for i in range(param_pr['nrPop']):
        f = create_pop(matrix,matrix_f,param_pr)
        print(f)
        x.append(i)
        y.append(f.get_fitt())
        if f.get_fitt() < best_furnica.get_fitt():
            best_furnica.set_drum(f.get_drum())
            best_furnica.set_fitt(f.get_fitt())
        modifica(matrix,param_pr['N'])


    print("best")
    print(best_furnica)

    plt.plot(x,y)
    plt.xlabel('generatie')
    plt.ylabel('fitness')

    plt.show()



main()