import cupy as cp
import random
import time

def getMaxValue(iVal, capacity):

    totalValue = 0
    for i in iVal:
        curWt = int(i[0])
        curVal = int(i[1])
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
    capacity = 500000000

    for i in range(300000):
        tempwt.append(random.randint(1, 100))
        tempval.append(i + 1)

    wt = cp.array(tempwt)
    val = cp.array(tempval)
    
    start = time.time()
    cost = cp.divide(val, wt)

    iVal = cp.column_stack((wt, val, cost))

    # sorting items by value
    sorted_iVal = iVal[cp.argsort(-iVal[:, -1])]

    iVal_cpu = cp.asnumpy(sorted_iVal)

    #print(iVal)
    #print(sorted_iVal)
    #print("Finding Max")
    

    maxValue = getMaxValue(iVal_cpu, capacity)

    t = time.time() - start
    print(t)
    print("Maximum value in Knapsack =", maxValue)