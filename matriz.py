import sys
import numpy as np
import time

kmax = int(sys.argv[1])
r = int(sys.argv[2])
amin = sys.argv[3]
amax = sys.argv[4]

tempoNormal = 0
tempoLista = 0

def MultiplyMatrix(A, B):
    C = np.zeros((len(A), len(B)), dtype=int)
    
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

def list_comp_matrix_multiplication(A, B):
    """Third and final version of the list comprehension matrix multiplication."""
    return [[sum([x*y for (x, y) in zip(row, col)]) for col in zip(*B)] for row in A]

if __name__ == "__main__":
    for n in range(1, kmax+1):
        tempoNormal = 0 
        tempoLista = 0
        for _ in range(r):
            A = np.random.randint(low = amin, high = amax, size = (2**n,2**n))
            B = np.random.randint(low = amin, high = amax, size = (2**n,2**n))

            start_time = time.time()
            C = MultiplyMatrix(A,B)
            end_time = time.time()
            tempoNormal += (end_time - start_time)

            start_time = time.time()
            D = list_comp_matrix_multiplication(A,B)
            end_time = time.time()
            tempoLista += (end_time - start_time)
        print(tempoNormal, tempoLista)