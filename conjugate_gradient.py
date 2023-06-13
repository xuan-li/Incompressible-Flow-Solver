import numpy as np
import scipy as sp
from scipy.io import mmread
import copy
from timer import Timer
import torch
import taichi as ti

def conjugate_gradient(A, b, diag, tol, max_iter, translation_invariant=False):
    # diagonal preconditioned
    x = torch.zeros(A.shape[0])
    r = b - A @ x
    if translation_invariant:
        r[0] = 0
    r_norm = torch.dot(r, r) ** 0.5
    if r_norm < 1e-10:
        # print(f'CG converged in 0 iterations.')
        return x
    q = r / diag
    p = q.clone()
    tol = tol * r_norm
    rq = torch.dot(r, q)
    for i in range(max_iter):
        r_norm = torch.dot(r, r) ** 0.5
        if r_norm < tol:
            # print(f'CG converged in {i} iterations.')
            break
        Ap = A @ p
        if translation_invariant:
            Ap[0] = 0
        alpha = rq / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        q = r / diag
        rq_prev = rq
        rq = torch.dot(r,q)
        beta = rq / rq_prev
        p = q + beta * p
    
    return x

if __name__ == "__main__":
    # torch.set_default_device('cuda:0')
    torch.set_default_dtype(torch.float64)
    A_sci = mmread('A.mtx')
    A_sci = A_sci.tocsr()
    I = torch.tensor(A_sci.indptr)
    J = torch.tensor(A_sci.indices)
    V = torch.tensor(A_sci.data)
    A = torch.sparse_csr_tensor(I, J, V, size=A_sci.shape)
    b = torch.ones(A.shape[0])
    print(f"A shape: {A.shape}")
    diag = torch.zeros(A.shape[0])
    for i in range(A.shape[0]):
        diag[i] = A[i, i]
    import time
    t = time.time()
    for i in range(10):
        conjugate_gradient(A, b, diag, tol=1e-7, max_iter=10000)
    print(f"Time: {time.time() - t}")