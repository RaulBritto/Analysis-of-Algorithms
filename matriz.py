import sys
import numpy as np
import time

kmax = int(sys.argv[1])
r = int(sys.argv[2])
amin = sys.argv[3]
amax = sys.argv[4]

tempoNormal = 0
tempoLista = 0

def printMatrix(matrix):
    for line in matrix:
        print("\t".join(map(str, line)))

def MultiplyMatrix(A, B):
    C = np.zeros((len(A), len(B)), dtype=int)
    
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

def addMatrix(A,B):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(B))]

def subMatrix(A,B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(B))]

def Strassen(A, B):
    """Strassen Algorithm"""
    if len(A) <= 2:
        return  list_comp_matrix_multiplication(A, B)
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


def list_comp_matrix_multiplication(A, B):
    """Third and final version of the list comprehension matrix multiplication."""
    return [[sum([x*y for (x, y) in zip(row, col)]) for col in zip(*B)] for row in A]

if __name__ == "__main__":
    for n in range(1, kmax+1):
        tempoNormal = 0 
        tempoLista = 0
        tempoStrassen = 0
        for _ in range(r):
            A = np.random.randint(low = amin, high = amax, size = (2**n,2**n))
            B = np.random.randint(low = amin, high = amax, size = (2**n,2**n))

            #print("A: ", A)
            #print("B: ", B)

            start_time = time.time()
            C = MultiplyMatrix(A,B)
            end_time = time.time()
            tempoNormal += (end_time - start_time)
            #print("C: ", C)

            start_time = time.time()
            D = list_comp_matrix_multiplication(A,B)
            end_time = time.time()
            tempoLista += (end_time - start_time)
            #print("D: ", D)

            start_time = time.time()
            E = Strassen(A,B)
            end_time = time.time()
            tempoStrassen += (end_time - start_time)
            #print("E: ", E)


        print("2^",n , ": ",tempoNormal, tempoLista, tempoStrassen)