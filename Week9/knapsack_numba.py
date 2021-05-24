from concurrent.futures import thread
import cupy as iVal
import cupy as wt
import cupy as val
import random
import numpy
import numba

class ItemValue:
    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val // wt
 
    def __lt__(self, other):
        return self.cost < other.cost


def getMaxValue(iVal, capacity):

    totalValue = 0
    for i in iVal:
        curWt = int(i.wt)
        curVal = int(i.val)
        if capacity - curWt >= 0:
            capacity -= curWt
            totalValue += curVal
        else:
            fraction = capacity / curWt
            totalValue += curVal * fraction
            capacity = int(capacity - (curWt * fraction))
            break
    return totalValue

if __name__ == "__main__":
    random.seed(2222)
    wt = []
    val = []

    for i in range(30000):
        wt.append(random.randint(1, 100))
        val.append(i)

    print("Values initialized")

    capacity = 500000

    iVal = []
    for i in range(len(wt)):
        iVal.append(ItemValue(wt[i], val[i], i))

    # sorting items by value
    iVal.sort(reverse=True)

    print("Values sorted")

    maxValue = getMaxValue(iVal, capacity)
    print("Maximum value in Knapsack =", maxValue)