from concurrent.futures import thread
import cupy as cp
import cupy
import random
import time

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
    tempwt = []
    tempval = []

    for i in range(30):
        tempwt.append(random.randint(1, 100))
        tempval.append(i)

    wt = cp.array(tempwt)
    val = cp.array(tempval)
    print(wt)
    capacity = 500000
    start = time.time()
    iVal = []
    for i in range(len(wt)):
        iVal.append(ItemValue(wt[i], val[i], i))

    # sorting items by value
    cupy.sort(wt, -1)


    maxValue = getMaxValue(iVal, capacity)

    t = time.time() - start
    print(t)
    print("Maximum value in Knapsack =", maxValue)