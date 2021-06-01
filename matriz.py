import sys
import numpy as np
import time
import logging
from matplotlib import pyplot as plt
import seaborn as sns

kmax = int(sys.argv[1])
r = int(sys.argv[2])
amin = sys.argv[3]
amax = sys.argv[4]

def plotResults(k, tNaive, tList, tStrassen):
    x = list(range(1, k+1))
    a = sns.lineplot(x, tNaive, label="Naive")
    a = sns.lineplot(x, tList, label="List Comprehesion")
    a = sns.lineplot(x, tStrassen, label="Strassen")
    a.set(xlabel='Tamanho da entrada 2^', ylabel='tempo de execução médio (s)')
    a.legend()
    plt.show()

def MultiplyMatrix(A, B):
    """Multiply two matrices using naive algorithm"""
    C = np.zeros((len(A), len(B)), dtype=int)
    
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

def addMatrix(A,B):
    """Add two matrices"""
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(B))]

def subMatrix(A,B):
    """subtraction two matrices"""
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(B))]

def Strassen(A, B):
    """Strassen Algorithm to squared matrices"""
    if len(A) <= 2:
        return  matrixMultiplicationListComprehesion(A, B)
    else:
        n = int(len(A)/2)

        a11 = [[A[i][j] for j in range(len(A)) if j < n  ] for i in range(len(A)) if i < n]
        a12 = [[A[i][j] for j in range(len(A)) if j >= n ] for i in range(len(A)) if i < n]
        a21 = [[A[i][j] for j in range(len(A)) if j < n  ] for i in range(len(A)) if i >= n]
        a22 = [[A[i][j] for j in range(len(A)) if j >= n ] for i in range(len(A)) if i >= n]

        b11 = [[B[i][j] for j in range(len(B)) if j < n  ] for i in range(len(B)) if i < n]
        b12 = [[B[i][j] for j in range(len(B)) if j >= n ] for i in range(len(B)) if i < n]
        b21 = [[B[i][j] for j in range(len(B)) if j < n  ] for i in range(len(B)) if i >= n]
        b22 = [[B[i][j] for j in range(len(B)) if j >= n ] for i in range(len(B)) if i >= n]

        m1 = Strassen(addMatrix(a11,a22), addMatrix(b11,b22))
        m2 = Strassen(addMatrix(a21,a22), b11)
        m3 = Strassen(a11, subMatrix(b12, b22))
        m4 = Strassen(a22, subMatrix(b21, b11))
        m5 = Strassen(addMatrix(a11, a12), b22)
        m6 = Strassen(subMatrix(a21, a11), addMatrix(b11, b12))
        m7 = Strassen(subMatrix(a12, a22), addMatrix(b21, b22))

        c11 = addMatrix(addMatrix(m1, m4), subMatrix(m7, m5))
        c12 = addMatrix(m3, m5)
        c21 = addMatrix(m2, m4)
        c22 = addMatrix(subMatrix(m1, m2), addMatrix(m3, m6))
        
        C = np.concatenate((np.concatenate((c11, c12), axis=1), np.concatenate((c21, c22), axis=1)), axis = 0 )

        return C


def matrixMultiplicationListComprehesion(A, B):
    """Multiply two squared matrices using list comprehesion"""
    return [[sum([x*y for (x, y) in zip(row, col)]) for col in zip(*B)] for row in A]

if __name__ == "__main__":
    
    tNaive = [0] * kmax 
    tList = [0] * kmax
    tStrassen = [0] * kmax

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    for n in range(1, kmax+1):
        for _ in range(r):
            A = np.random.randint(low = amin, high = amax, size = (2**n,2**n))
            B = np.random.randint(low = amin, high = amax, size = (2**n,2**n))

            start_time = time.time()
            C = MultiplyMatrix(A,B)
            end_time = time.time()
            tNaive[n-1] += (end_time - start_time)

            start_time = time.time()
            D = matrixMultiplicationListComprehesion(A,B)
            end_time = time.time()
            tList[n-1] += (end_time - start_time)

            start_time = time.time()
            E = Strassen(A,B)
            end_time = time.time()
            tStrassen[n-1] += (end_time - start_time)

        logging.debug("2^%d: {Naive: %f}, {List: %f}, {Strassen: %f}", n, tNaive[n-1], tList[n-1], tStrassen[n-1])
        logging.info("2^%d: Average {Naive: %f}, {List: %f}, {Strassen: %f}", n, tNaive[n-1]/r, tList[n-1]/r, tStrassen[n-1]/r)
 
    plotResults(kmax, [x/r for x in tNaive], [x/r for x in tList], [x/r for x in tStrassen])