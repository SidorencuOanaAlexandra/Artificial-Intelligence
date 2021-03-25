from numpy import random
import numpy as np

def generateNewValue(lim1,lim2):
    # r= randint(lim1,lim2)
    # list =[]
    # for i in range(r):
    #     list.append(i)
    # for i in range(r,lim2+1):
    #     list.append(i)
    # return list
    l=[]
    ll=np.random.permutation(lim2)
    for i in range(lim2):
        l.append(ll[i])
    return l