import numpy as np
import scipy as sp
from scipy.io import mmread

def conjugate_gradient(A, b, tol=1e-3, max_iter=1000):
    # diagonal preconditioned
    diag = A.diagonal()
    x = np.zeros_like(b)
    r = b
    q = r / diag
    p = q
    r_norm = np.linalg.norm(r)
    tol = tol * r_norm
    rq = r @ q
    for i in range(max_iter):
        r_norm = np.linalg.norm(r)
        if i % 10 == 0 and r_norm < tol:
            print(f'CG converged in {i} iterations.')
            return x
        Ap = A @ p
        alpha = rq / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        q = r / diag
        rq_prev = rq
        rq = r @ q
        beta = rq / rq_prev
        p = q + beta * p

if __name__ == "__main__":
    A = mmread('test_matrix.mtx')
    b = np.ones(A.shape[0])
    print(f"A shape: {A.shape}")
    x = conjugate_gradient(A, b, tol=1e-7)