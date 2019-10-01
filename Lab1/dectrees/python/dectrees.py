import dtree as d, monkdata as m, drawtree_qt5 as drawqt, random, matplotlib.pyplot as plt
import math
#print("monk1 entropy: " + str(d.entropy(m.monk1)))
#print("monk2 entropy: " + str(d.entropy(m.monk2)))
#print("monk3 entropy: " + str(d.entropy(m.monk3)))

def calcUniform(amount):
    i = 1
    entropy = 0
    while i <= amount:
        entropy += - (1/amount) * (d.log2(1/amount))
        i += 1
    return entropy

def calcNonUniform(probabilities):
    entropy = 0
    for element in probabilities:
        entropy += - (element) * (d.log2(element))
    return entropy

def calcAverageGain():
    print("Monk1 = ", end=' ')
    for x in m.attributes:
        print(str(x) + ":" + str(d.averageGain(m.monk1, x)), end=' ')
    print()
    print("Monk2 = ", end=' ')
    for x in m.attributes:
        print(str(x) + ":" + str(d.averageGain(m.monk2, x)), end=' ')
    print()
    print("Monk3 = ", end=' ')
    for x in m.attributes:
        print(str(x) + ":" + str(d.averageGain(m.monk3, x)), end=' ')
    print()

def buildTree():
    m1 = d.buildTree(m.monk1, m.attributes)
    print("MONK 1 TEST: " + str(d.check(m1, m.monk1test)))
    
    m2 = d.buildTree(m.monk2, m.attributes)
    print("MONK 2 TEST: " + str(d.check(m2, m.monk2test)))

    m3 = d.buildTree(m.monk3, m.attributes)
    print("MONK 3 TEST: " + str(d.check(m3, m.monk3test)))
    

    #drawqt.drawTree(m1)
    

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune(dataset, fraction):
    def pruning(tree, val):
        prunedTree = d.allPruned(tree)
        treeVal = d.check(tree, val)
        bestTree = tree
        for x in prunedTree:
            current = d.check(x, val)
            if current >= treeVal:
                bestTree = pruning(x, val)
                break
        return bestTree

    train, val = partition(dataset, fraction)
    tree = d.buildTree(train, m.attributes)
    #drawqt.drawTree(pruning(tree, val))
    return pruning(tree, val)
    
def pruneAlot(i, dataset,testset, fraction):
    total = 0 
    for x in range(i):
        total += d.check(prune(dataset, fraction), testset)
    return total/i

def pruneAlotList(i, dataset,testset, fraction):
    total = list()
    for x in range(i):
         total.append(d.check(prune(dataset, fraction), testset))
    return total

def fractionLists(i, dataset, testset):
    X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    fractions = dict()
    for x in X:
        fractions[x] = pruneAlotList(i, dataset, testset, x)
    return fractions

#for x in fractionLists(100, m.monk1,m.monk1test):
#    print(x)
#print(fractionLists(100, m.monk1,m.monk1test))

def plotMonk1(size, dictionary):
    X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    plt.xlabel('Fraction')
    plt.ylabel('Success Rate')

    for l in X: 
        dataList = dictionary.get(l)
        tempList = [l] * size
        plt.scatter(tempList, dataList, label='Monk1', c='red')

    
    M1 = [0.7650787037037016, 0.7869953703703675, 0.7967685185185149, 0.8005185185185161, 0.8001388888888857, 0.779898148148146]
    plt.plot(X, M1, '-', c='blue')
    #M3 = [0.9137962962962999, 0.9394259259259305, 0.9472962962963009, 0.9521759259259308, 0.94831481481482, 0.9344722222222287]
    #plt.scatter(X, M3, label='Monk3', c='blue')
    #plt.legend(loc='upper left')
    plt.title('Average correctness in pruning Monk1 with fraction x')
    plt.show()

def plotMonk3(size, dictionary):
    X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    plt.xlabel('Fraction')
    plt.ylabel('Success Rate')

    for l in X: 
        dataList = dictionary.get(l)
        tempList = [l] * size
        plt.scatter(tempList, dataList, label='Monk3', c='green')

    
    M1 = [0.9137962962962999, 0.9394259259259305, 0.9472962962963009, 0.9521759259259308, 0.94831481481482, 0.9344722222222287]
    plt.plot(X, M1, '-', c='blue')
    #M3 = [0.9137962962962999, 0.9394259259259305, 0.9472962962963009, 0.9521759259259308, 0.94831481481482, 0.9344722222222287]
    #plt.scatter(X, M3, label='Monk3', c='blue')
    #plt.legend(loc='upper left')
    plt.title('Average correctness in pruning Monk3 with fraction x')
    plt.show()


def treees():
    #tree=d.buildTree(m.monk1, m.attributes)
    
    prunedTree = prune(m.monk1, 0.6)

    #drawqt.drawTree(tree)
    
    drawqt.drawTree(prunedTree)

    
def spread(i, avg, dataset,fraction, testset):
    total = 0 
    for x in range(i):
        total += math.pow(d.check(prune(dataset, fraction), testset) - avg,2)
    return math.sqrt(total/i)

def spreadFrac(i,dataset,testset):
    X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    Y = [0.9137962962962999, 0.9394259259259305, 0.9472962962963009, 0.9521759259259308, 0.94831481481482, 0.9344722222222287]
    for x in range(i):
        print(Y[x])
        print(X[x])
        spread(i, Y[x], dataset, testset,X[x])

#print(str(spreadFrac(500,m.monk1,m.monk1test)))
#plotMonk1()
#treees()
#buildTree()
plotMonk1(500, fractionLists(500, m.monk1, m.monk1test))
#plotMonk3(500, fractionLists(500, m.monk3, m.monk3test))

"""
print(pruneAlot(500, m.monk3, m.monk3test, 0.3))
print(pruneAlot(500, m.monk3, m.monk3test, 0.4))
print(pruneAlot(500, m.monk3, m.monk3test, 0.5))
print(pruneAlot(500, m.monk3, m.monk3test, 0.6))
print(pruneAlot(500, m.monk3, m.monk3test, 0.7))
print(pruneAlot(500, m.monk3, m.monk3test, 0.8))
"""
#print(pruneAlot(500, m.monk3, m.monk3test, 0.6))
#print(d.check(d.buildTree(m.monk1,m.attributes), m.monk1test))

#tree = prune(m.monk1, 0.6)
#drawqt.drawTree(tree)
#drawqt.drawTree(d.buildTree(m.monk1, m.attributes))
#print(d.check(tree, m.monk1test))


#print(d.check(d.buildTree(m.monk1, m.attributes), m.monk1test))

#print("Uniform distr.: " + str(calcUniform(16)))
#print("Non Uniform distr.: "+ str(calcNonUniform({1/5,2/5,1/5,1/10,1/10})))
#calcAverageGain()
#buildTree()
