from random import randint


def generateNewValue(lim1, lim2):
    return randint(lim1, lim2)

def generareNouaSolutie(param):
    list = [0]*param['n']
    k=1
    r = randint(0,param['n']-1)
    # for i in range(param['n']):
    #    if list[i]==0:
    #        list[i]=k
    #        for j in range(param['n']):
    #            if param['matrix'][i][j]==1 and list[j]==0:
    #                list[j]=k
    #        k+=1
    for i in range(r,param['n']):
        if list[i] == 0:
            list[i] = k
            for j in range(param['n']):
                if param['matrix'][i][j] == 1 and list[j] == 0:
                    list[j] = k
            k += 1
    for i in range(0,r):
        if list[i] == 0:
            list[i] = k
            for j in range(param['n']):
                if param['matrix'][i][j] == 1 and list[j] == 0:
                    list[j] = k
            k += 1
    return list