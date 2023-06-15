import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import torch

def conjugate_gradient(A, b, diag, tol, max_iter, translation_invariant=False, x0=None):
    # diagonal preconditioned
    if x0 is None:
        x = torch.zeros(A.shape[0])
    else:
        x = x0
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

def write_ply(fn, pos):
    pos_data = pos.to_numpy()
    num_particles = len(pos_data)
    dtype = "double" if pos_data.dtype == np.float64 else "float"
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property {dtype} x
property {dtype} y
property {dtype} z
end_header
"""
        f.write(str.encode(header))
        f.write(pos_data.tobytes())

@ti.func
def bilerp(x, f00, f10, f01, f11):
    return (1 - x[0]) * (1 - x[1]) * f00 + x[0] * (1 - x[1]) * f10 + (1 - x[0]) * x[1] * f01 + x[0] * x[1] * f11

@ti.data_oriented
class ImcompressibleFlowSimulation:
    def __init__(self, Lx, Ly, nx, ny, nu, dt=0.01):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.ds = min([self.dx, self.dy])
        self.dt = dt

        # boundary data
        self.vL = ti.Vector.field(2, float, shape=())
        self.vR = ti.Vector.field(2, float, shape=())
        self.vB = ti.Vector.field(2, float, shape=())
        self.vT = ti.Vector.field(2, float, shape=())

        # vx data
        self.vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.laplacian_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vx_prev = ti.field(dtype=float, shape=(nx+1, ny))
        self.vx_id = ti.field(dtype=int, shape=(nx+1, ny))
        self.gradp_x = ti.field(dtype=float, shape=(nx+1, ny))

        # vy data
        self.vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.advect_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.laplacian_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.advect_vy_prev = ti.field(dtype=float, shape=(nx, ny+1))
        self.vy_id = ti.field(dtype=int, shape=(nx, ny+1))
        self.gradp_y = ti.field(dtype=float, shape=(nx, ny+1))

        # cell data
        self.pressure = ti.field(dtype=float, shape=(nx, ny))
        self.div_v = ti.field(dtype=float, shape=(nx, ny))
        self.pressure_id = ti.field(dtype=int, shape=(nx, ny))

        # node data
        self.vorticity = ti.field(dtype=float, shape=(nx+1, ny+1))

        self.nu = nu
        self.num_vel_dof = ti.field(dtype=int, shape=())
        self.num_pressure_dof = ti.field(dtype=int, shape=())
        self.num_surface_constraints = ti.field(dtype=int, shape=())
        self.surface_constraints_layout = ti.root.dynamic(ti.i, nx * ny * 8, chunk_size=1024)
        self.surface_vel = ti.Vector.field(2, dtype=float)
        self.surface_loc = ti.Vector.field(2, dtype=float)
        self.Eu = ti.Vector.field(2, dtype=float)
        self.surface_constraints_layout.place(self.surface_vel, self.surface_loc, self.Eu)
        
        self.L_I = ti.field(dtype=int)
        self.L_J = ti.field(dtype=int)
        self.L_V = ti.field(dtype=float)
        self.L_triplets_num = ti.field(dtype=int, shape=())
        self.Q_I = ti.field(dtype=int)
        self.Q_J = ti.field(dtype=int)
        self.Q_V = ti.field(dtype=float)
        self.Q_triplets_num = ti.field(dtype=int, shape=())
        self.matrix_triplets_layout = ti.root.dynamic(ti.i, 2 ** 30, chunk_size=1024)
        self.matrix_triplets_layout.place(self.L_I, self.L_J, self.L_V, self.Q_I, self.Q_J, self.Q_V)
        
        self.cg_tol = 1e-3
        self.prev_lam = None

    @ti.kernel
    def add_box(self, min_corner_:ti.types.ndarray(), max_corner_:ti.types.ndarray(), vel_:ti.types.ndarray()):
        min_corner = ti.Vector([min_corner_[0], min_corner_[1]])
        max_corner = ti.Vector([max_corner_[0], max_corner_[1]])
        vel = ti.Vector([vel_[0], vel_[1]])
        bbox = max_corner - min_corner
        for i in range(bbox[0] // self.ds):
            p = ti.Vector([min_corner[0] + (i + 0.5) * self.ds, min_corner[1]])
            idx = ti.atomic_add(self.num_surface_constraints[None], 1)
            self.surface_loc[idx] = p
            self.surface_vel[idx] = vel
            p = ti.Vector([min_corner[0] + (i + 0.5) * self.ds, max_corner[1]])
            idx = ti.atomic_add(self.num_surface_constraints[None], 1)
            self.surface_loc[idx] = p
            self.surface_vel[idx] = vel
        for i in range(bbox[1] // self.ds):
            p = ti.Vector([min_corner[0], min_corner[1] + (i + 0.5) * self.ds])
            idx = ti.atomic_add(self.num_surface_constraints[None], 1)
            self.surface_loc[idx] = p
            self.surface_vel[idx] = vel
            p = ti.Vector([max_corner[0], min_corner[1] + (i + 0.5) * self.ds])
            idx = ti.atomic_add(self.num_surface_constraints[None], 1)
            self.surface_loc[idx] = p
            self.surface_vel[idx] = vel
    
    @ti.kernel
    def add_circle(self, center_:ti.types.ndarray(), radius:float, vel_:ti.types.ndarray()):
        center = ti.Vector([center_[0], center_[1]])
        vel = ti.Vector([vel_[0], vel_[1]])
        n_seg = int(2 * np.pi * radius // self.ds)
        dtheta = 2 * np.pi / n_seg
        for i in range(n_seg):
            p = ti.Vector([center[0] + radius * ti.cos(dtheta * i), center[1] + radius * ti.sin(dtheta * i)])
            idx = ti.atomic_add(self.num_surface_constraints[None], 1)
            self.surface_loc[idx] = p
            self.surface_vel[idx] = vel

    @ti.kernel
    def L_triplets_to_torch(self, I:ti.types.ndarray(), J:ti.types.ndarray(), V:ti.types.ndarray()):
        for i in range(self.L_triplets_num[None]):
            I[i] = self.L_I[i]
            J[i] = self.L_J[i]
            V[i] = self.L_V[i]
    
    @ti.kernel
    def Q_triplets_to_torch(self, I:ti.types.ndarray(), J:ti.types.ndarray(), V:ti.types.ndarray()):
        for i in range(self.Q_triplets_num[None]):
            I[i] = self.Q_I[i]
            J[i] = self.Q_J[i]
            V[i] = self.Q_V[i]

    def construct_matrices(self):
        self.label_dof()

        self.fill_laplacian_matrix()
        I = torch.zeros(self.L_triplets_num[None], dtype=torch.long)
        J = torch.zeros(self.L_triplets_num[None], dtype=torch.long)
        V = torch.zeros(self.L_triplets_num[None])
        self.L_triplets_to_torch(I, J, V)
        L = self.L = torch.sparse_coo_tensor(torch.stack([I, J]), V, (self.num_vel_dof[None], self.num_vel_dof[None]))
        
        self.fill_Q_matrix()
        I = torch.zeros(self.Q_triplets_num[None], dtype=torch.long)
        J = torch.zeros(self.Q_triplets_num[None], dtype=torch.long)
        V = torch.zeros(self.Q_triplets_num[None])
        self.Q_triplets_to_torch(I, J, V)
        self.Q = torch.sparse_coo_tensor(torch.stack([I, J]), V, (self.num_vel_dof[None], self.num_pressure_dof[None] + self.num_surface_constraints[None] * 2))
        
        I = torch.arange(self.num_vel_dof[None], dtype=torch.long)
        J = torch.arange(self.num_vel_dof[None], dtype=torch.long)
        V = torch.ones(self.num_vel_dof[None])
        Identity = torch.sparse_coo_tensor(torch.stack([I, J]), V, (self.num_vel_dof[None], self.num_vel_dof[None]))

        self.R = Identity - self.dt * self.nu * 0.5 * L
        self.R_inv = Identity + self.nu * self.dt * 0.5 * L + ((0.5 * self.dt * self.nu) ** 2) * (L @ L)
        
        self.A = self.Q.T @ self.R_inv @ self.Q

        self.R_diag = torch.zeros(self.R.shape[0])
        for i in range(self.R.shape[0]):
            self.R_diag[i] = self.R[i, i]
        self.A_diag = torch.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            self.A_diag[i] = self.A[i, i]

        print("Reinald number: ", 1. / self.nu)

    @ti.kernel
    def fill_laplacian_matrix(self):
        self.L_triplets_num[None] = 0
        for I in ti.grouped(self.laplacian_vx):
            if self.vx_id[I] == -1:
                continue
            idx = ti.atomic_add(self.L_triplets_num[None], 1)
            self.L_I[idx] = self.vx_id[I]
            self.L_J[idx] = self.vx_id[I]
            self.L_V[idx] = -2 / self.dx ** 2 - 2 / self.dy ** 2
            if self.vx_id[I[0]-1, I[1]] != -1: # left is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I[0]-1, I[1]]
                self.L_V[idx] = 1 / self.dx ** 2
            if self.vx_id[I[0]+1, I[1]] != -1: # right is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I[0]+1, I[1]]
                self.L_V[idx] = 1 / self.dx ** 2
            if I[1] > 0:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I[0], I[1]-1]
                self.L_V[idx] = 1 / self.dy ** 2
            else:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I]
                self.L_V[idx] = -1 / self.dy ** 2

            if I[1] < self.ny-1:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I[0], I[1]+1]
                self.L_V[idx] = 1 / self.dy ** 2
            else:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vx_id[I]
                self.L_J[idx] = self.vx_id[I]
                self.L_V[idx] = -1 / self.dy ** 2
            
        for I in ti.grouped(self.laplacian_vy):
            if self.vy_id[I] == -1:
                continue
            idx = ti.atomic_add(self.L_triplets_num[None], 1)
            self.L_I[idx] = self.vy_id[I]
            self.L_J[idx] = self.vy_id[I]
            self.L_V[idx] = -2 / self.dx ** 2 - 2 / self.dy ** 2
            if I[0] > 0: # left is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I[0]-1, I[1]]
                self.L_V[idx] = 1 / self.dx ** 2
            else:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I]
                self.L_V[idx] = -1 / self.dx ** 2
            
            if I[0] < self.nx-1: # right is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I[0]+1, I[1]]
                self.L_V[idx] = 1 / self.dx ** 2
            else:
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I]
                self.L_V[idx] = -1 / self.dx ** 2
                
            if self.vy_id[I[0], I[1]-1] != -1: # bottom is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I[0], I[1]-1]
                self.L_V[idx] = 1 / self.dy ** 2
            if self.vy_id[I[0], I[1]+1] != -1: # top is valid
                idx = ti.atomic_add(self.L_triplets_num[None], 1)
                self.L_I[idx] = self.vy_id[I]
                self.L_J[idx] = self.vy_id[I[0], I[1]+1]
                self.L_V[idx] = 1 / self.dy ** 2

    @ti.kernel
    def fill_Q_matrix(self):
        # [G, H]
        self.Q_triplets_num[None] = 0
        for I in ti.grouped(self.vx):
            if self.vx_id[I] == -1:
                continue
            if I[0] > 0:
                idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                self.Q_I[idx] = self.vx_id[I]
                self.Q_J[idx] = self.pressure_id[I[0]-1, I[1]]
                self.Q_V[idx] = -1 / self.dx
            if I[0] < self.nx:
                idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                self.Q_I[idx] = self.vx_id[I]
                self.Q_J[idx] = self.pressure_id[I]
                self.Q_V[idx] = 1 / self.dx

        for I in ti.grouped(self.vy):
            if self.vy_id[I] == -1:
                continue
            if I[1] > 0:
                idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                self.Q_I[idx] = self.vy_id[I]
                self.Q_J[idx] = self.pressure_id[I[0], I[1]-1]
                self.Q_V[idx] = -1 / self.dy
            if I[1] < self.ny:
                idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                self.Q_I[idx] = self.vy_id[I]
                self.Q_J[idx] = self.pressure_id[I]
                self.Q_V[idx] = 1 / self.dy
        
        for p in range(self.num_surface_constraints[None]):
            xi = self.surface_loc[p]
            rel_pos = ti.Vector([xi[0] / self.dx, xi[1] / self.dy - 0.5])
            base = ti.floor(rel_pos - 0.5).cast(int)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    fx = rel_pos - (base + offset)
                    weight = self.weight(fx[0]) * self.weight(fx[1])
                    idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                    self.Q_I[idx] = self.vx_id[base + offset]
                    self.Q_J[idx] = self.num_pressure_dof[None] + p * 2
                    self.Q_V[idx] = weight

            rel_pos = ti.Vector([xi[0] / self.dx - 0.5, xi[1] / self.dy])
            base = ti.floor(rel_pos - 0.5).cast(int)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    fx = rel_pos - (base + offset)
                    weight = self.weight(fx[0]) * self.weight(fx[1])
                    idx = ti.atomic_add(self.Q_triplets_num[None], 1)
                    self.Q_I[idx] = self.vy_id[base + offset]
                    self.Q_J[idx] = self.num_pressure_dof[None] + p * 2 + 1
                    self.Q_V[idx] = weight
            
    @ti.kernel
    def fill_r1(self, r1: ti.types.ndarray()):
        for I in ti.grouped(self.vx_id):
            if self.vx_id[I] == -1:
                continue
            r1[self.vx_id[I]] += self.vx[I] + 0.5 * self.dt * self.nu * self.laplacian_vx[I] - 0.5 * self.dt * (3 * self.advect_vx[I] - self.advect_vx_prev[I])
            # -Lu_bc
            if I[0] == 1:
                r1[self.vx_id[I]] += 0.5 * self.dt * self.nu * self.vL[None][0] / self.dx ** 2
            elif I[0] == self.nx-1:
                r1[self.vx_id[I]] += 0.5 * self.dt * self.nu * self.vR[None][0] / self.dx ** 2
            if I[1] == 0:
                r1[self.vx_id[I]] += self.dt * self.nu * self.vB[None][0] / self.dy ** 2
            elif I[1] == self.ny-1:
                r1[self.vx_id[I]] += self.dt * self.nu * self.vT[None][0] / self.dy ** 2
        

        for I in ti.grouped(self.vy_id):
            if self.vy_id[I] == -1:
                continue
            r1[self.vy_id[I]] += self.vy[I] + 0.5 * self.dt * self.nu * self.laplacian_vy[I] - 0.5 * self.dt * (3 * self.advect_vy[I] - self.advect_vy_prev[I])
            # -Lu_bc
            if I[0] == 0:
                r1[self.vy_id[I]] += self.dt * self.nu * self.vL[None][1] / self.dx ** 2
            elif I[0] == self.nx-1:
                r1[self.vy_id[I]] += self.dt * self.nu * self.vR[None][1] / self.dx ** 2
            if I[1] == 1:
                r1[self.vy_id[I]] += 0.5 * self.dt * self.nu * self.vB[None][1] / self.dy ** 2
            elif I[1] == self.ny-1:
                r1[self.vy_id[I]] += 0.5 * self.dt * self.nu * self.vT[None][1] / self.dy ** 2
        

    @ti.kernel
    def fill_r2(self, r2: ti.types.ndarray()):
        for I in ti.grouped(self.pressure):
            if I[0] == 0:
                r2[self.pressure_id[I]] -= self.vL[None][0] / self.dx
            if I[0] == self.nx-1:
                r2[self.pressure_id[I]] += self.vR[None][0] / self.dx
            if I[1] == 0:
                r2[self.pressure_id[I]] -= self.vB[None][1] / self.dy
            if I[1] == self.ny-1:
                r2[self.pressure_id[I]] += self.vT[None][1] / self.dy
                
        for i in range(self.num_surface_constraints[None]):
            r2[self.num_pressure_dof[None] + 2 * i + 0] = self.surface_vel[i][0]
            r2[self.num_pressure_dof[None] + 2 * i + 1] = self.surface_vel[i][1]
    
    @ti.func
    def weight(self, i):
        weight = ti.cast(0.0, float)
        if ti.abs(i) < 0.5:
            weight = (1 + ti.sqrt(1 - 3 * i ** 2)) / 3
        elif ti.abs(i) < 1.5:
            weight =  (5 - 3*ti.abs(i) - ti.sqrt(1 - 3 * (1-ti.abs(i))**2)) / 6
        else:
            weight = 0
        return weight
    
    @ti.kernel
    def test_delta(self):
        s = ti.cast(0.0, float)
        for i in range(-10, 10):
            for j in range(-10, 10):
                s += self.weight(i) * self.weight(j)
        print("\sum_{i, j} delta(i, j) ds^2 = ", s)
        
    def reset(self):
        self.vx.fill(0)
        self.vy.fill(0)
        self.pressure.fill(0)
        self.vorticity.fill(0)
        self.num_surface_constraints[None] = 0

    @ti.kernel
    def label_dof(self):
        self.num_pressure_dof[None] = 0
        self.num_vel_dof[None] = 0
        for I in ti.grouped(self.pressure):
            idx = ti.atomic_add(self.num_pressure_dof[None], 1)
            self.pressure_id[I] = idx
        for I in ti.grouped(self.vx):
            if I[0] == 0 or I[0] == self.nx:
                self.vx_id[I] = -1
            else:
                idx = ti.atomic_add(self.num_vel_dof[None], 1)
                self.vx_id[I] = idx
        for I in ti.grouped(self.vy):
            if I[1] == 0 or I[1] == self.ny:
                self.vy_id[I] = -1
            else:
                idx = ti.atomic_add(self.num_vel_dof[None], 1)
                self.vy_id[I] = idx

    @ti.kernel
    def set_wall_vel(self, vL:ti.types.ndarray(), vR:ti.types.ndarray(), vB:ti.types.ndarray(), vT:ti.types.ndarray()):
        self.vL[None] = [vL[0], vL[1]]
        self.vR[None] = [vR[0], vR[1]]
        self.vB[None] = [vB[0], vB[1]]
        self.vT[None] = [vT[0], vT[1]]

        # set left/right wall vx
        for j in range(self.ny):
            self.vx[0, j] = vL[0]
            self.vx[self.nx, j] = vR[0]
        
        # set bottom/top wall vy
        for i in range(self.nx):
            self.vy[i, 0] = vB[1]
            self.vy[i, self.ny] = vT[1]
        
    
    @ti.kernel
    def visualize_u(self, image: ti.types.ndarray()):
        for i, j in ti.ndrange(self.nx, self.ny):
            image[i, j] = 0.5 * (self.vx[i, j] + self.vx[i+1, j])
    
    @ti.kernel
    def visualize_v(self, image: ti.types.ndarray()):
        for i, j in ti.ndrange(self.nx, self.ny):
            image[i, j] = 0.5 * (self.vy[i, j] + self.vy[i, j+1])
    
    @ti.kernel
    def visualize_p(self, image: ti.types.ndarray()):
        for i, j in ti.ndrange(self.nx, self.ny):
            image[i, j] = self.pressure[i, j]
    
    @ti.kernel
    def visualize_vorticity(self, image: ti.types.ndarray()):
        for i, j in ti.ndrange(self.nx, self.ny):
            image[i, j] = 0.25 * (self.vorticity[i, j] + self.vorticity[i+1, j] 
                                  + self.vorticity[i, j+1] + self.vorticity[i+1, j+1])
    
    @ti.kernel
    def compute_gradp(self):
        for I in ti.grouped(self.gradp_x):
            if I[0] == 0 or I[0] == self.nx:
                self.gradp_x[I] = 0
            else:
                self.gradp_x[I] = (self.pressure[I[0], I[1]] - self.pressure[I[0]-1, I[1]]) / self.dx
        for I in ti.grouped(self.gradp_y):
            if I[1] == 0 or I[1] == self.ny:
                self.gradp_y[I] = 0
            else:
                self.gradp_y[I] = (self.pressure[I[0], I[1]] - self.pressure[I[0], I[1]-1]) / self.dy
    
    @ti.kernel
    def compute_div_v(self):
        for I in ti.grouped(self.div_v):
            gradx_vx = (self.vx[I[0] + 1, I[1]] - self.vx[I]) / self.dx
            grady_vy = (self.vy[I[0], I[1] + 1] - self.vy[I]) / self.dy
            self.div_v[I] = gradx_vx + grady_vy
    
    @ti.kernel
    def compute_Eu(self):
        for p in range(self.num_surface_constraints[None]):
            self.Eu[p] = ti.Vector([0.0, 0.0])
            xi = self.surface_loc[p]
            rel_pos = ti.Vector([xi[0] / self.dx, xi[1] / self.dy - 0.5])
            base = ti.floor(rel_pos - 0.5).cast(int)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    fx = rel_pos - (base + offset)
                    weight = self.weight(fx[0]) * self.weight(fx[1])
                    self.Eu[p][0] += weight * self.vx[base + offset]
            rel_pos = ti.Vector([xi[0] / self.dx - 0.5, xi[1] / self.dy])
            base = ti.floor(rel_pos - 0.5).cast(int)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    fx = rel_pos - (base + offset)
                    weight = self.weight(fx[0]) * self.weight(fx[1])
                    self.Eu[p][1] += weight * self.vy[base + offset]

    @ti.kernel
    def compute_vorticity(self):
        for I in ti.grouped(self.vorticity):
            vy_left = ti.cast(0, float)
            if I[0] == 0:
                vy_left = 2 * self.vL[None][1] - self.vy[I]
            else:
                vy_left = self.vy[I[0]-1, I[1]]
            vy_right = ti.cast(0, float)
            if I[0] == self.nx:
                vy_right = 2 * self.vR[None][1] - self.vy[I]
            else:
                vy_right = self.vy[I[0], I[1]]
            vx_bottom = ti.cast(0, float)
            if I[1] == 0:
                vx_bottom = 2 * self.vB[None][0] - self.vx[I]
            else:
                vx_bottom = self.vx[I[0], I[1]-1]
            vx_top = ti.cast(0, float)
            if I[1] == self.ny:
                vx_top = 2 * self.vT[None][0] - self.vx[I]
            else:
                vx_top = self.vx[I[0], I[1]]
            self.vorticity[I] = (vx_top - vx_bottom) / self.dy - (vy_right - vy_left) / self.dx

    @ti.kernel
    def compute_laplacian_v(self):
        for I in ti.grouped(self.laplacian_vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            vxl = self.vx[I[0]-1, I[1]]
            vxr = self.vx[I[0]+1, I[1]]
            vxb = ti.cast(0, float)
            if I[1] == 0:
                vxb = 2 * self.vB[None][0] - self.vx[I]
            else:
                vxb = self.vx[I[0], I[1]-1]
            vxt = ti.cast(0, float)
            if I[1] == self.ny-1:
                vxt = 2 * self.vT[None][0] - self.vx[I]
            else:
                vxt = self.vx[I[0], I[1]+1]
            self.laplacian_vx[I] = (vxr - 2 * self.vx[I] + vxl) / (self.dx ** 2) + (vxt - 2 * self.vx[I] + vxb) / (self.dy ** 2)
        
        for I in ti.grouped(self.laplacian_vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            vyb = self.vy[I[0], I[1]-1]
            vyt = self.vy[I[0], I[1]+1]
            vyl = ti.cast(0, float)
            if I[0] == 0:
                vyl = 2 * self.vL[None][1] - self.vy[I]
            else:
                vyl = self.vy[I[0]-1, I[1]]
            vyr = ti.cast(0, float)
            if I[0] == self.nx - 1:
                vyr = 2 * self.vR[None][1] - self.vy[I]
            else:
                vyr = self.vy[I[0]+1, I[1]]
            self.laplacian_vy[I] = (vyr - 2 * self.vy[I] + vyl) / (self.dx ** 2) + (vyt - 2 * self.vy[I] + vyb) / (self.dx ** 2)

    @ti.kernel
    def compute_advection(self):
        for I in ti.grouped(self.advect_vx):
            self.advect_vx_prev[I] = self.advect_vx[I]
        for I in ti.grouped(self.advect_vy):
            self.advect_vy_prev[I] = self.advect_vy[I]

        for I in ti.grouped(self.advect_vx):
            if I[0] == 0 or I[0] == self.nx:
                self.advect_vx[I] = 0
                continue
            vxl = 0.5 * (self.vx[I] + self.vx[I[0]-1, I[1]])
            vxr = 0.5 * (self.vx[I] + self.vx[I[0]+1, I[1]])
            vxb = ti.cast(0, float)
            if I[1] == 0:
                vxb = self.vB[None][0]
            else:
                vxb = 0.5 * (self.vx[I] + self.vx[I[0], I[1]-1])
            vxt = ti.cast(0, float)
            if I[1] == self.ny - 1:
                vxt = self.vT[None][0]
            else:
                vxt = 0.5 * (self.vx[I] + self.vx[I[0], I[1]+1])
            
            vyb = ti.cast(0, float)
            if I[1] == 0:
                vyb = self.vB[None][1]
            else:
                vyb = 0.5 * (self.vy[I[0] - 1, I[1]] + self.vy[I])
            vyt = ti.cast(0, float)
            if I[1] == self.ny - 1:
                vyt = self.vT[None][1]
            else:
                vyt = 0.5 * (self.vy[I[0]-1, I[1]+1] + self.vy[I[0], I[1]+1])
            self.advect_vx[I] = (vxr ** 2 - vxl ** 2) / self.dx + (vxt * vyt - vxb * vyb) / self.dy
        
        for I in ti.grouped(self.advect_vy):
            if I[1] == 0 or I[1] == self.ny:
                self.advect_vy[I] = 0
                continue
            vyb = 0.5 * (self.vy[I] + self.vy[I[0], I[1]-1])
            vyt = 0.5 * (self.vy[I] + self.vy[I[0], I[1]+1])
            vyl = ti.cast(0, float)
            if I[0] == 0:
                vyl = self.vL[None][1]
            else:
                vyl = 0.5 * (self.vy[I] + self.vy[I[0]-1, I[1]])
            vyr = ti.cast(0, float)
            if I[0] == self.nx - 1:
                vyr = self.vR[None][1]
            else:
                vyr = 0.5 * (self.vy[I] + self.vy[I[0]+1, I[1]])
            
            vxl = ti.cast(0, float)
            if I[0] == 0:
                vxl = self.vL[None][0]
            else:
                vxl = 0.5 * (self.vx[I[0], I[1]] + self.vx[I[0], I[1]-1])
            vxr = ti.cast(0, float)
            if I[0] == self.nx - 1:
                vxr = self.vR[None][0]
            else:
                vxr = 0.5 * (self.vx[I[0]+1, I[1]] + self.vx[I[0]+1, I[1]-1])
            
            self.advect_vy[I] = (vyr * vxr - vyl * vxl) / self.dx + (vyt ** 2 - vyb ** 2) / self.dy
    
    @ti.kernel
    def assign_velocity(self, v:ti.types.ndarray()):
        for I in ti.grouped(self.vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            self.vx[I] = v[self.vx_id[I]]
        for I in ti.grouped(self.vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            self.vy[I] = v[self.vy_id[I]]
    
    @ti.kernel
    def get_velocity(self, v:ti.types.ndarray()):
        for I in ti.grouped(self.vx):
            v[self.vx_id[I]] = self.vx[I]
        for I in ti.grouped(self.vy):
            v[self.vy_id[I]] = self.vy[I]
    
    @ti.kernel
    def update_velocity(self, du:ti.types.ndarray()):
        for I in ti.grouped(self.vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            self.vx[I] += du[self.vx_id[I]]
        
        for I in ti.grouped(self.vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            self.vy[I] += du[self.vy_id[I]]
    
    @ti.kernel
    def divergence_error(self) -> float:
        res = ti.cast(0.0, float)
        for I in ti.grouped(self.div_v):
            ti.atomic_max(res, ti.abs(self.div_v[I]))
        return res

    def substep(self):
        self.compute_advection()
        self.compute_laplacian_v()
        r1 = torch.zeros(self.num_vel_dof[None])
        self.fill_r1(r1)
        v0 = torch.zeros(self.num_vel_dof[None])
        self.get_velocity(v0)
        v = conjugate_gradient(self.R, r1, self.R_diag, self.cg_tol, self.R.shape[0], x0 = v0)
        r2 = torch.zeros(self.num_pressure_dof[None] + self.num_surface_constraints[None] * 2)
        self.fill_r2(r2)
        rhs = self.Q.T @ v - r2
        if self.prev_lam is None:
            self.prev_lam = torch.zeros(self.num_pressure_dof[None] + self.num_surface_constraints[None] * 2)
        lam = conjugate_gradient(self.A, rhs, self.A_diag, self.cg_tol, self.A.shape[0], True, x0=self.prev_lam)
        self.prev_lam = lam
        new_v = v - self.R_inv @ (self.Q @ lam)
        self.assign_velocity(new_v)
    
    @ti.kernel
    def interpolated_velocity(self, pos: ti.types.ndarray(), vel: ti.types.ndarray()):
        pos_vec = ti.Vector([pos[0], pos[1]])
        vel_vec = self._interpolated_velocity(pos_vec)
        vel[0] = vel_vec[0]
        vel[1] = vel_vec[1]

    @ti.func
    def _interpolated_velocity(self, pos):
        x = ti.Vector([pos[0]/ self.dx, pos[1]/ self.dy])
        I00 = ti.floor(ti.Vector([x[0], x[1] - 0.5])).cast(int)
        I01 = I00 + ti.Vector([0, 1])
        I10 = I00 + ti.Vector([1, 0])
        I11 = I00 + ti.Vector([1, 1])
        base = ti.Vector([I00[0], I00[1]+0.5])
        v00 = 2 * self.vB[None][0] - self.vx[I00[0], 0]
        v10 = 2 * self.vB[None][0] - self.vx[I10[0], 0]
        if I00[1] != -1:
            v00 = self.vx[I00]
            v10 = self.vx[I10]
        v01 = 2 * self.vT[None][0] - self.vx[I01[0], self.ny - 1]
        v11 = 2 * self.vT[None][0] - self.vx[I11[0], self.ny - 1]
        if I01[1] != self.ny:
            v01 = self.vx[I01]
            v11 = self.vx[I11]
        vx = bilerp(x - base, v00, v10, v01, v11)

        I00 = ti.floor(ti.Vector([x[0] - 0.5, x[1]])).cast(int)
        I01 = I00 + ti.Vector([0, 1])
        I10 = I00 + ti.Vector([1, 0])
        I11 = I00 + ti.Vector([1, 1])
        base = ti.Vector([I00[0] + 0.5, I00[1]])
        v00 = 2 * self.vL[None][1] - self.vy[0, I00[1]]
        v01 = 2 * self.vL[None][1] - self.vy[0, I01[1]]
        if I00[0] != -1:
            v00 = self.vy[I00]
            v01 = self.vy[I01]
        v10 = 2 * self.vR[None][1] - self.vy[self.nx - 1, I10[1]]
        v11 = 2 * self.vR[None][1] - self.vy[self.nx - 1, I11[1]]
        if I10[0] != self.nx:
            v10 = self.vy[I10]
            v11 = self.vy[I11]
        vy = bilerp(x - base, v00, v10, v01, v11)
        return ti.Vector([vx, vy])

    @ti.kernel
    def advect_particles(self, pos: ti.types.ndarray()):
        for p in range(pos.shape[0]):
            v = self._interpolated_velocity(ti.Vector([pos[p][0], pos[p][1]]))
            pos[p][0] += self.dt * v[0]
            pos[p][1] += self.dt * v[1]
            if pos[p][0] < 0:
                pos[p][0] += self.Lx
            elif pos[p][0] >= self.Lx:
                pos[p][0] -= self.Lx
            if pos[p][1] < 0:
                pos[p][1] += self.Ly
            elif pos[p][1] >= self.Ly:
                pos[p][1] -= self.Ly


if __name__ == '__main__':
    torch.set_default_device('cuda:0')
    torch.set_default_dtype(torch.float64)
    ti.init(arch=ti.cuda, default_fp=ti.f64, device_memory_fraction=0.7)
    
    import sys
    testcase = int(sys.argv[1])

    torch.random.manual_seed(0)
    
    if testcase == 1:
        dt = 0.002
        frame_dt = 0.1
        Re = 1000
        simulator = ImcompressibleFlowSimulation(1, 1, 256, 256, 1/Re, dt = dt)
        simulator.set_wall_vel(np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.]), np.array([1.,0.]))
        simulator.cg_tol = 1e-2
        radius = 0.1
        center = np.array([0.7, 0.5])
        simulator.add_circle(center, radius, np.array([0., 0.]))
        simulator.construct_matrices()
        pos_data = np.random.rand(10000, 3)
        pos_data[:, 2] = 0
        outside_circle = np.linalg.norm(pos_data[:, :2] - center, axis=1) > radius
        pos_data = pos_data[outside_circle]
        pos = ti.Vector.ndarray(3, float, pos_data.shape[0])
        pos.from_numpy(pos_data)
        import os
        base_folder = f'results/FSI_{Re}'
        os.makedirs(base_folder, exist_ok=True)
        for f in range(200):
            for i in range(int(frame_dt / simulator.dt)):
                simulator.substep()
                simulator.advect_particles(pos)
            print("frame {} done".format(f+1))
            write_ply(f'{base_folder}/{f+1}.ply', pos)
            simulator.compute_vorticity()
            image = np.zeros((simulator.nx, simulator.ny))
            simulator.visualize_vorticity(image)
            image = image.transpose(1, 0)
            mean = np.mean(image)
            std = np.std(image)
            image[image > mean + 0.5 * std] = mean + 0.5 * std
            image[image < mean - 0.5 * std] = mean - 0.5 * std
            plt.clf()
            plt.imshow(image, cmap='jet')
            # set color from blue to red
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig(f'{base_folder}/vorticity_{f+1}.png', bbox_inches='tight')
    
    elif testcase == 2:
        dt = 0.0001
        frame_dt = 0.04
        Re = 5000
        simulator = ImcompressibleFlowSimulation(2, 1, 256, 128, 1/Re, dt = dt)
        simulator.set_wall_vel(np.array([1, 0.]), np.array([1,0.]), np.array([0,0.]), np.array([0,0.]))
        simulator.cg_tol = 1e-2
        radius = 0.2
        center = np.array([0.7, 0.5])
        simulator.add_circle(center, radius, np.array([0., 0.]))
        simulator.construct_matrices()
        pos_data = np.random.rand(20000, 3)
        pos_data[:, 2] = 0
        pos_data[10000:, 0] += 1
        outside_circle = np.linalg.norm(pos_data[:, :2] - center, axis=1) > radius
        pos_data = pos_data[outside_circle]
        pos = ti.Vector.ndarray(3, float, pos_data.shape[0])
        pos.from_numpy(pos_data)
        import os
        base_folder = f'results/channel_{Re}_vel1'
        os.makedirs(base_folder, exist_ok=True)
        for f in range(500):
            for i in range(int(frame_dt / simulator.dt)):
                simulator.substep()
                simulator.advect_particles(pos)
            print("frame {} done".format(f+1))
            write_ply(f'{base_folder}/{f+1}.ply', pos)
            simulator.compute_vorticity()
            image = np.zeros((simulator.nx, simulator.ny))
            simulator.visualize_vorticity(image)
            image = image.transpose(1, 0)
            mean = np.mean(image)
            std = np.std(image)
            image[image > mean + 1 * std] = mean + 1 * std
            image[image < mean - 1 * std] = mean - 1 * std
            plt.clf()
            plt.imshow(image, cmap='jet')
            # set color from blue to red
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig(f'{base_folder}/vorticity_{f+1}.png', bbox_inches='tight')