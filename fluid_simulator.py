import taichi as ti
import numpy as np
from conjugate_gradient import conjugate_gradient


@ti.func
def bilerp(x, f00, f10, f01, f11):
    return (1 - x[0]) * (1 - x[1]) * f00 + x[0] * (1 - x[1]) * f10 + (1 - x[0]) * x[1] * f01 + x[0] * x[1] * f11

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

@ti.data_oriented
class ImcompressibleFlowSimulation:
    def __init__(self, Lx, Ly, nx, ny, nu, dt=0.01, dtype=ti.f32):
        self.vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.pressure = ti.field(dtype=float, shape=(nx, ny))
        self.vorticity = ti.field(dtype=float, shape=(nx+1, ny+1))
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.gradp_x = ti.field(dtype=float, shape=(nx+1, ny))
        self.gradp_y = ti.field(dtype=float, shape=(nx, ny+1))
        self.div_v = ti.field(dtype=float, shape=(nx, ny))
        self.advect_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.laplacian_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vx_prev = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.laplacian_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.advect_vy_prev = ti.field(dtype=float, shape=(nx, ny+1))
        self.vL = ti.Vector.field(2, float, shape=())
        self.vR = ti.Vector.field(2, float, shape=())
        self.vT = ti.Vector.field(2, float, shape=())
        self.vB = ti.Vector.field(2, float, shape=())
        self.nu = nu
        self.pressure_id = ti.field(dtype=int, shape=(nx, ny))
        self.vx_id = ti.field(dtype=int, shape=(nx+1, ny))
        self.vy_id = ti.field(dtype=int, shape=(nx, ny+1))
        self.num_vel_dof = ti.field(dtype=int, shape=())
        self.num_pressure_dof = ti.field(dtype=int, shape=())
        self.dtype=dtype

        self.label_dof()
        Lv_builder = ti.linalg.SparseMatrixBuilder(self.num_vel_dof[None], self.num_vel_dof[None], self.num_vel_dof[None] * 6, dtype=self.dtype)
        self.fill_laplacian_matrix(Lv_builder)
        self.Lv = Lv_builder.build()

        Lp_builder = ti.linalg.SparseMatrixBuilder(self.num_pressure_dof[None], self.num_pressure_dof[None], self.num_pressure_dof[None] * 8, dtype=self.dtype)
        self.fill_pressure_matrix(Lp_builder)
        self.Lp = Lp_builder.build()

        print("Reinald number: ", 1. / self.nu)

    @ti.kernel
    def fill_laplacian_matrix(self, A: ti.types.sparse_matrix_builder()):
        for I in ti.grouped(self.laplacian_vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            A[self.vx_id[I], self.vx_id[I]] += 1 + self.nu * self.dt / (self.dx ** 2)
            if I[0] == 1:
                A[self.vx_id[I], self.vx_id[I[0]+1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
            elif I[0] == self.nx-1:
                A[self.vx_id[I], self.vx_id[I[0]-1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
            else:
                A[self.vx_id[I], self.vx_id[I[0]-1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
                A[self.vx_id[I], self.vx_id[I[0]+1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
            
            if I[1] == 0:
                A[self.vx_id[I], self.vx_id[I[0], I[1]+1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
                A[self.vx_id[I], self.vx_id[I]] += 1 + 1.5 * self.nu * self.dt / (self.dy ** 2)
            elif I[1] == self.ny-1:
                A[self.vx_id[I], self.vx_id[I[0], I[1]-1]]  += -0.5 * self.nu * self.dt / (self.dy ** 2)
                A[self.vx_id[I], self.vx_id[I]] += 1 + 1.5 * self.nu * self.dt / (self.dy ** 2)
            else:
                A[self.vx_id[I], self.vx_id[I[0], I[1]-1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
                A[self.vx_id[I], self.vx_id[I[0], I[1]+1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
                A[self.vx_id[I], self.vx_id[I]] += 1 + self.nu * self.dt / (self.dy ** 2)
        
        for I in ti.grouped(self.laplacian_vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            A[self.vy_id[I], self.vy_id[I]] += 1 + self.nu * self.dt / (self.dy ** 2)
            if I[1] == 1:
                A[self.vy_id[I], self.vy_id[I[0], I[1]+1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
            elif I[1] == self.ny-1:
                A[self.vy_id[I], self.vy_id[I[0], I[1]-1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
            else:
                A[self.vy_id[I], self.vy_id[I[0], I[1]-1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
                A[self.vy_id[I], self.vy_id[I[0], I[1]+1]] += -0.5 * self.nu * self.dt / (self.dy ** 2)
            
            if I[0] == 0:
                A[self.vy_id[I], self.vy_id[I[0]+1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
                A[self.vy_id[I], self.vy_id[I]] += 1 + 1.5 * self.nu * self.dt / (self.dx ** 2)
            elif I[0] == self.nx-1:
                A[self.vy_id[I], self.vy_id[I[0]-1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
                A[self.vy_id[I], self.vy_id[I]] += 1 + 1.5 * self.nu * self.dt / (self.dx ** 2)
            else:
                A[self.vy_id[I], self.vy_id[I[0]-1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
                A[self.vy_id[I], self.vy_id[I[0]+1, I[1]]] += -0.5 * self.nu * self.dt / (self.dx ** 2)
                A[self.vy_id[I], self.vy_id[I]] += 1 + self.nu * self.dt / (self.dx ** 2)

    @ti.kernel
    def fill_advection_rhs(self, advection_rhs: ti.types.ndarray()):
        for I in ti.grouped(self.vx_id):
            if I[0] == 0 or I[0] == self.nx:
                continue
            advection_rhs[self.vx_id[I]] += self.vx[I] + 0.5 * self.dt * self.nu * self.laplacian_vx[I] - 0.5 * self.dt * (3 * self.advect_vx[I] - self.advect_vx_prev[I])
            if I[0] == 1:
                advection_rhs[self.vx_id[I]] += 0.5 * self.nu * self.dt / (self.dx ** 2) * self.vx[I[0]-1, I[1]]
            elif I[0] == self.nx-1:
                advection_rhs[self.vx_id[I]] += 0.5 * self.nu * self.dt / (self.dx ** 2) * self.vx[I[0]+1, I[1]]
            
            if I[1] == 0:
                advection_rhs[self.vx_id[I]] += self.nu * self.dt / (self.dy ** 2) * self.vB[None][1]
            elif I[1] == self.ny-1:
                advection_rhs[self.vx_id[I]] += self.nu * self.dt / (self.dy ** 2) * self.vT[None][1]

        for I in ti.grouped(self.vy_id):
            if I[1] == 0 or I[1] == self.ny:
                continue
            advection_rhs[self.vy_id[I]] += self.vy[I] + 0.5 * self.dt * self.nu * self.laplacian_vy[I] \
                                                - 0.5 * self.dt * (3 * self.advect_vy[I] - self.advect_vy_prev[I])
            if I[1] == 1:
                advection_rhs[self.vy_id[I]] += 0.5 * self.nu * self.dt / (self.dy ** 2) * self.vy[I[0], I[1]-1]
            elif I[1] == self.ny-1:
                advection_rhs[self.vy_id[I]] += 0.5 * self.nu * self.dt / (self.dy ** 2) * self.vy[I[0], I[1]+1]
            
            if I[0] == 0:
                advection_rhs[self.vy_id[I]] += self.nu * self.dt / (self.dx ** 2) * self.vL[None][0]
            elif I[0] == self.nx-1:
                advection_rhs[self.vy_id[I]] += self.nu * self.dt / (self.dx ** 2) * self.vR[None][0]

    @ti.kernel
    def fill_pressure_matrix(self, A: ti.types.sparse_matrix_builder()):
        for I in ti.grouped(self.pressure):
            if I[0] > 0: # left edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dx ** 2)
                A[self.pressure_id[I], self.pressure_id[I[0]-1, I[1]]] -= self.dt / (self.dx ** 2)
            if I[0] < self.nx-1: # right edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dx ** 2)
                A[self.pressure_id[I], self.pressure_id[I[0]+1, I[1]]] -= self.dt / (self.dx ** 2)
            if I[1] > 0: # bottom edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dy ** 2)
                A[self.pressure_id[I], self.pressure_id[I[0], I[1]-1]] -= self.dt / (self.dy ** 2)
            if I[1] < self.ny-1: # top edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dy ** 2)
                A[self.pressure_id[I], self.pressure_id[I[0], I[1]+1]] -= self.dt / (self.dy ** 2)
    
    @ti.kernel
    def fill_pressure_rhs(self, pressure_rhs:ti.types.ndarray()):
        for I in ti.grouped(self.pressure):
            pressure_rhs[self.pressure_id[I]] = -self.div_v[I]
            if I[0] == 0: # left edge
                pressure_rhs[self.pressure_id[I]] -= self.vL[None][0] / self.dx
            if I[0] == self.nx-1: # right edge
                pressure_rhs[self.pressure_id[I]] += self.vR[None][0] / self.dx
            if I[1] == 0: # bottom edge
                pressure_rhs[self.pressure_id[I]] -= self.vB[None][1] / self.dy
            if I[1] < self.ny-1: # top edge
                pressure_rhs[self.pressure_id[I]] -= self.vT[None][1] / self.dy

    def reset(self):
        self.vx.fill(0)
        self.vy.fill(0)
        self.pressure.fill(0)
        self.vorticity.fill(0)
        self.vL.fill(0)
        self.vB.fill(0)
        self.vT.fill(0)
        self.vB.fill(0)

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
            if  I[1] == 0 or I[1] == self.ny:
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
    def test_sample(self):
        period = self.Lx / 6
        for I in ti.grouped(self.vx):
            center = ti.Vector([self.dx * I[0], self.dy * I[1] + 0.5 * self.dy])
            radius = (center - ti.Vector([0, 0])).norm()
            value = ti.sin(2 * np.pi * radius / period)
            self.vx[I] = value
        
        for I in ti.grouped(self.vy):
            center = ti.Vector([self.dx * I[0] + 0.5 * self.dx, self.dy * I[1]])
            radius = (center - ti.Vector([0, 0])).norm()
            value = ti.sin(2 * np.pi * radius / period)
            self.vy[I] = value
        
        for I in ti.grouped(self.pressure):
            center = ti.Vector([self.dx * I[0] + 0.5 * self.dx, self.dy * I[1] + 0.5 * self.dy])
            radius = (center - ti.Vector([0, 0])).norm()
            value = ti.sin(2 * np.pi * radius / period)
            self.pressure[I] = value
        
        for I in ti.grouped(self.vorticity):
            center = ti.Vector([self.dx * I[0], self.dy * I[1]])
            radius = (center - ti.Vector([0, 0])).norm()
            value = ti.sin(2 * np.pi * radius / period)
            self.vorticity[I] = value
    
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
            if self.vx_id[I] != -1:
                self.vx[I] = v[self.vx_id[I]]
        for I in ti.grouped(self.vy):
            if self.vy_id[I] != -1:
                self.vy[I] = v[self.vy_id[I]]
    
    @ti.kernel
    def assign_pressure(self, p:ti.types.ndarray()):
        for I in ti.grouped(self.pressure):
            self.pressure[I] = p[self.pressure_id[I]]
    
    @ti.kernel
    def update_velocity(self):
        for I in ti.grouped(self.vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            self.vx[I] -= self.dt * self.gradp_x[I]
        
        for I in ti.grouped(self.vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            self.vy[I] -= self.dt * self.gradp_y[I]
    
    @ti.kernel
    def divergence_error(self) -> float:
        res = ti.cast(0.0, float)
        for I in ti.grouped(self.div_v):
            ti.atomic_max(res, ti.abs(self.div_v[I]))
        return res
    
    @ti.kernel
    def check_values(self):
        for I in ti.grouped(self.vx):
            self.advect_vx[I]
        for I in ti.grouped(self.vy):
            self.advect_vy[I]
        for I in ti.grouped(self.laplacian_vx):
            self.laplacian_vx[I]
        for I in ti.grouped(self.laplacian_vy):
            self.laplacian_vy[I]
            
    def substep(self):
        self.compute_advection()
        self.compute_laplacian_v()
        advection_rhs = ti.ndarray(float, shape=(self.num_vel_dof[None]))
        advection_rhs.fill(0.0)
        self.fill_advection_rhs(advection_rhs)
        v = conjugate_gradient(self.Lv, advection_rhs, 1e-5, self.num_vel_dof[None])
        self.assign_velocity(v)
        self.compute_div_v()
        pressure_rhs = ti.ndarray(float, shape=(self.num_pressure_dof[None]))
        pressure_rhs.fill(0.0)
        self.fill_pressure_rhs(pressure_rhs)
        p = conjugate_gradient(self.Lp, pressure_rhs, 1e-5, self.num_pressure_dof[None], True)
        self.assign_pressure(p)
        self.compute_gradp()
        self.update_velocity()
        

    @ti.kernel
    def advect_particles(self, pos: ti.types.ndarray()):
        for p in range(pos.shape[0]):
            x = ti.Vector([pos[p][0]/ self.dx, pos[p][1]/ self.dy])
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

            pos[p][0] += self.dt * vx
            pos[p][1] += self.dt * vy
            pos[p][0] %= self.Lx
            pos[p][1] %= self.Ly
            

if __name__ == '__main__':
    default_fp = ti.f64
    ti.init(default_fp=default_fp, arch=ti.cpu)
    dt = 0.1
    frame_dt = 1.0
    simulator = ImcompressibleFlowSimulation(1, 1, 100, 100, 1./1000, dt = dt, dtype=default_fp)
    simulator.set_wall_vel(np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.]), np.array([1.,0.]))
    pos = ti.Vector.ndarray(3, float, 10000)
    pos_data = np.random.rand(10000, 3)
    pos_data[:, 2] = 0
    pos.from_numpy(pos_data)
    import os
    os.makedirs('results_400', exist_ok=True)
    write_ply('results/{}.ply'.format(0), pos)
    for f in range(100):
        for i in range(int(frame_dt/ simulator.dt)):
            simulator.substep()
            #simulator.substep_explicit()
            simulator.advect_particles(pos)
        print(simulator.vx[1, 1], simulator.vy[1, 1])
        write_ply('results/{}.ply'.format(f+1), pos)
        print("frame {} done".format(f+1))
    

    

