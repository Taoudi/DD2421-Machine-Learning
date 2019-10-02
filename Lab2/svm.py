import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
numpy.random.seed(100)

P  = 0

classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.8 + [1.5, 0.5],
    numpy.random.randn(10, 2) * 0.8 + [-1.5, 0.5]))

classB = numpy.random.randn(20, 2) * 0.8 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
    -numpy.ones(classB.shape[0])))

N = inputs.shape[0]

permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

non_zero_alpha = 0

start = numpy.zeros(N)

def lin_kernel(x,y):
    return numpy.dot(x,y)

def poly_kernel(x, y):
    p = 3
    return (numpy.dot(x,y) + 1)**p

def rbf_kernel(x, y):
    constant = 0.5
    sigma = 2*(constant**2)
    magnitude = (x[0] - y[0])**2 + (x[1]-y[1])**2
    result = math.pow(math.e, -(magnitude/sigma))
    return result

def generateP(T, dataset):
    p = list()
    for i in range(N):
        pi = list()
        for j in range(N):
            pi.insert(j,T[j]*T[i]*rbf_kernel(dataset[i],dataset[j]))
        p.insert(i,pi)
    return p


def objective(a):
    sum = 0
    suma = 0
    for i in range(N):
        suma = suma + a[i]
        for j in range(N):
            sum = sum + a[i]*a[j]*P[i][j]
    return (sum/2) - suma

def zerofun(a):
    sum = 0
    for i in range(N):
        sum = sum + a[i]*targets[i]
    return sum

def extractNonZero(a):
    nonzerolist = list()
    i = 0
    for v in a:
        if v > 1e-5:
            nonzerolist.append([v, targets[i], inputs[i]])
        i += 1
    return nonzerolist

def bias(support_vectors):
    if not support_vectors:
        return 0
        
    sv = support_vectors[0][2]
    sv_t = support_vectors[0][1]
    sum = 0
    for x in support_vectors:
        sum = sum + x[0]*x[1]*rbf_kernel(sv,x[2])
    sum = sum - sv_t
    return sum

def ind(x,y,support_vectors):
    b = bias(support_vectors)
    if not support_vectors:
        return 0 
    sum = 0
    for el in support_vectors:
        sum = sum + el[0]*el[1]*rbf_kernel([x,y],el[2])
    return sum - b

def doPlot():
    plt.plot(
    [p[0] for p in classA],
    [p[1] for p in classA],
    'b. ')

    plt.plot(
    [p[0] for p in classB],
    [p[1] for p in classB],
    'r. ')

    plt.axis('equal')
    
    xgrid = numpy.linspace(-5, 5)
    ygrid = numpy.linspace(-4, 4)

    grid = numpy.array([[ind(x,y,non_zero_alpha) for x in xgrid] for y in ygrid])
 
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

    plt.savefig('printyboi.pdf')
    plt.show()

#print(lin_kernel(t1,t2))
#print(start)
#dataset = generateDataSet()
#print(dataset)
#T = generateTSet()
#print(generateTSet())
P = generateP(targets, inputs)
#print(P)
#print(objective(start))

C = 5
B = [(0, C) for a in range(N)]
#print(zerofun(start))
ret = minimize(objective,start,bounds=B,constraints={'type':'eq', 'fun':zerofun})
alpha = ret['x']
print(ret['success'])
non_zero_alpha = extractNonZero(alpha)
#print(ind(non_zero_alpha))
doPlot()

#print(non_zero_alpha)
#print(bias(non_zero_alpha))

"""
constraint={'type':'eq', 'fun':zerofun}
C = 10
B = [(0, C) for a in range(N)]

ret = minimize(objective,start,bounds=B,constraints={'type':'eq', 'fun':zerofun})
alpha = ret['x']
print(alpha)
"""
