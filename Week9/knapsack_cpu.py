import random
import time

class ItemValue:
 
    """Item Value DataClass"""
 
    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val // wt
 
    def __lt__(self, other):
        return self.cost < other.cost
 
# Greedy Approach
 
 
class FractionalKnapSack:
 
    """Time Complexity O(n log n)"""
    @staticmethod
    def getMaxValue(wt, val, capacity):
        """function to get maximum value """
        #start = time.time()
        iVal = []
        for i in range(len(wt)):
            iVal.append(ItemValue(wt[i], val[i], i))
 
        # sorting items by value
        iVal.sort(reverse=True)

        #t = time.time() - start
        #print(t)
 
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
 
 
# Driver Code
if __name__ == "__main__":
    random.seed(2222)
    wt = []
    val = []

    for i in range(3000000):
        wt.append(random.randint(1, 100))
        val.append(i + 1)

    capacity = 5000000
    start = time.time()
    # Function call
    maxValue = FractionalKnapSack.getMaxValue(wt, val, capacity)

    t = time.time() - start
    print(t)
    print("Maximum value in Knapsack =", maxValue)
 