import taichi as ti
import numpy as np
import scipy as sp
from scipy.io import mmread
import copy

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), n: int, row: ti.types.ndarray(), col: ti.types.ndarray(), val: ti.types.ndarray()):
    for i in range(n):
        for j in range(row[i], row[i + 1]):
            A[i, col[j]] += val[j]

@ti.kernel
def add(a:ti.types.ndarray(), b:ti.types.ndarray(), apb:ti.types.ndarray()):
    for i in range(a.shape[0]):
        apb[i] = a[i] + b[i]

@ti.kernel
def subtract(a:ti.types.ndarray(), b:ti.types.ndarray(), asubb:ti.types.ndarray()):
    for i in range(a.shape[0]):
        asubb[i] = a[i] - b[i]

@ti.kernel
def divide(a:ti.types.ndarray(), b:ti.types.ndarray(), adivb:ti.types.ndarray()):
    for i in range(a.shape[0]):
        adivb[i] = a[i] / b[i]

@ti.kernel
def norm(a:ti.types.ndarray()) -> float:
    res = ti.cast(0.0, float)
    for i in range(a.shape[0]):
        ti.atomic_add(res, a[i] * a[i])
    return ti.sqrt(res)

@ti.kernel
def dot(a:ti.types.ndarray(), b:ti.types.ndarray()) -> float:
    res = ti.cast(0.0, float)
    for i in range(a.shape[0]):
        ti.atomic_add(res, a[i] * b[i])
    return res

@ti.kernel
def step_forward(a: ti.types.ndarray(), b: ti.types.ndarray(), dt: float, res: ti.types.ndarray()):
    for i in range(a.shape[0]):
        res[i] = a[i] + b[i] * dt

def conjugate_gradient(A, b, tol, max_iter, translation_invariant = False):
    # diagonal preconditioned
    if translation_invariant:
        b[0] = 0
    x = ti.ndarray(float, shape=A.shape[0])
    diag = ti.ndarray(float, shape=A.shape[0])
    for i in range(A.shape[0]):
        diag[i] = A[i, i]
    Ax = A @ x
    Ax[0] = 0
    r = ti.ndarray(float, shape=A.shape[0])
    q = ti.ndarray(float, shape=A.shape[0])
    subtract(b, Ax, r)
    divide(r, diag, q)
    p = copy.deepcopy(q)
    r_norm = norm(r)
    tol = tol * r_norm
    rq = dot(r, q)
    if r_norm < 1e-10:
        print(f'CG converged in 0 iterations.')
        return x
    for i in range(max_iter):
        r_norm = norm(r)
        if i % 10 == 0 and r_norm < tol:
            print(f'CG converged in {i} iterations.')
            break
        Ap = A @ p
        if translation_invariant:
            Ap[0] = 0
        alpha = rq / dot(p, Ap)
        step_forward(x, p, alpha, x)
        step_forward(r, Ap, -alpha, r)
        divide(r, diag, q)
        rq_prev = rq
        rq = dot(r,q)
        beta = rq / rq_prev
        step_forward(q, p, beta, p)
    
    return x

if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    A_sci = mmread('test_matrix.mtx')
    A_sci = A_sci.tocsr()
    A_builder = ti.linalg.SparseMatrixBuilder(A_sci.shape[0], A_sci.shape[1], A_sci.nnz)
    fill(A_builder, A_sci.shape[0], A_sci.indptr, A_sci.indices, A_sci.data)
    A = A_builder.build()
    b = ti.ndarray(float, shape=A.shape[0])
    b.fill(1)
    print(f"A shape: {A.shape}")
    conjugate_gradient(A, b, tol=1e-7, max_iter=10000)