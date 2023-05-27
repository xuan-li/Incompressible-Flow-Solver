import taichi as ti
import numpy as np
from conjugate_gradient import conjugate_gradient

@ti.data_oriented
class ImcompressibleFlowSimulation:
    def __init__(self, Lx, Ly, nx, ny, rho, nu):
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
        self.dt = 0.001
        self.gradp_x = ti.field(dtype=float, shape=(nx+1, ny))
        self.gradp_y = ti.field(dtype=float, shape=(nx, ny+1))
        self.div_v = ti.field(dtype=float, shape=(nx, ny))
        self.laplacian_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.laplacian_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.advect_vx = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vy = ti.field(dtype=float, shape=(nx, ny+1))
        self.advect_vx_prev = ti.field(dtype=float, shape=(nx+1, ny))
        self.advect_vy_prev = ti.field(dtype=float, shape=(nx, ny+1))
        self.vL = ti.Vector.field(2, float, shape=())
        self.vR = ti.Vector.field(2, float, shape=())
        self.vT = ti.Vector.field(2, float, shape=())
        self.vB = ti.Vector.field(2, float, shape=())
        self.rho = ti.cast(rho, float)
        self.nu = ti.cast(nu, float)
        self.pressure_id = ti.field(dtype=int, shape=(nx, ny))
        self.vx_id = ti.field(dtype=int, shape=(nx+1, ny))
        self.vy_id = ti.field(dtype=int, shape=(nx, ny+1))
        self.num_vel_dof = ti.field(dtype=int, shape=())
        self.num_pressure_dof = ti.field(dtype=int, shape=())

        self.label_dof()
        Lv_builder = ti.linalg.SparseMatrixBuilder(self.num_vel_dof[None], self.num_vel_dof[None], self.num_vel_dof[None] * 6)
        Lp_builder = ti.linalg.SparseMatrixBuilder(self.num_pressure_dof[None], self.num_pressure_dof[None], self.num_pressure_dof[None] * 8)
        self.fill_laplacian_matrix(Lv_builder)
        self.Lv = Lv_builder.build()
        self.fill_pressure_matrix(Lp_builder)
        self.Lp = Lp_builder.build()
        self.advection_rhs = ti.ndarray(float, shape=(self.num_vel_dof[None]))
        self.pressure_rhs = ti.ndarray(float, shape=(self.num_pressure_dof[None]))

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
    def fill_pressure_matrix(self, A: ti.types.sparse_matrix_builder()):
        for I in ti.grouped(self.pressure):
            if I[0] == 0 and I[1] == 0:
                continue
            if I[0] > 0: # left edge
                if self.pressure_id[I[0]-1, I[1]] != -1:
                    A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dx ** 2) / self.rho
                    A[self.pressure_id[I], self.pressure_id[I[0]-1, I[1]]] -= self.dt / (self.dx ** 2) / self.rho
            if I[0] < self.nx-1: # right edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dx ** 2) / self.rho
                A[self.pressure_id[I], self.pressure_id[I[0]+1, I[1]]] -= self.dt / (self.dx ** 2) / self.rho
            if I[1] > 0: # bottom edge
                if self.pressure_id[I[0], I[1]-1] != -1:
                    A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dy ** 2) / self.rho
                    A[self.pressure_id[I], self.pressure_id[I[0], I[1]-1]] -= self.dt / (self.dy ** 2) / self.rho
            if I[1] < self.ny-1: # top edge
                A[self.pressure_id[I], self.pressure_id[I]] += self.dt / (self.dy ** 2) / self.rho
                A[self.pressure_id[I], self.pressure_id[I[0], I[1]+1]] -= self.dt / (self.dy ** 2) / self.rho
    
    @ti.kernel
    def fill_pressure_rhs(self):
        pass

    @ti.kernel
    def fill_advection_rhs(self):
        for I in ti.grouped(self.vx):
            if I[0] == 0 or I[0] == self.nx:
                continue
            self.advection_rhs[self.vx_id[I]] += self.vx[I] + 0.5 * self.dt * self.nu * self.laplacian_vx[I] \
                                                - 0.5 * self.dt * (3 * self.advect_vx[I] - self.advect_vx_prev[I])
            if I[0] == 1:
                self.advection_rhs[self.vx_id[I]] += 0.5 * self.nu * self.dt / (self.dx ** 2) * self.vx[I-1, I[1]]
            elif I[0] == self.nx-1:
                self.advection_rhs[self.vx_id[I]] += 0.5 * self.nu * self.dt / (self.dx ** 2) * self.vx[I+1, I[1]]
        
        for I in ti.grouped(self.vy):
            if I[1] == 0 or I[1] == self.ny:
                continue
            self.advection_rhs[self.vy_id[I]] += self.vy[I] + 0.5 * self.dt * self.nu * self.laplacian_vy[I] \
                                                - 0.5 * self.dt * (3 * self.advect_vy[I] - self.advect_vy_prev[I])
            if I[1] == 1:
                self.advection_rhs[self.vy_id[I]] += 0.5 * self.nu * self.dt / (self.dy ** 2) * self.vy[I[0], I-1]
            elif I[1] == self.ny-1:
                self.advection_rhs[self.vy_id[I]] += 0.5 * self.nu * self.dt / (self.dy ** 2) * self.vy[I[0], I+1]

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
            if I[0] == 0 and I[1] == 0:
                self.pressure_id[I] = -1
            else:
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
            self.vy[i, 0] = vB [1]
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
                continue
            self.gradp_x[I] = (self.pressure[I[0], I[1]] - self.pressure[I[0]-1, I[1]]) / self.dx
        for I in ti.grouped(self.gradp_y):
            if I[1] == 0 or I[1] == self.ny:
                continue
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
            if self.pressure_id[I] != -1:
                self.pressure[I] = p[self.pressure_id[I]]
    
    @ti.kernel
    def update_velocity(self):
        pass

    def substep(self):
        self.compute_advection()
        self.compute_laplacian_v()
        self.pressure_rhs.fill(0)
        self.advection_rhs.fill(0)
        self.pressure.fill(0)
        self.fill_advection_rhs()
        v = ti.ndarray(float, self.num_vel_dof[None])
        conjugate_gradient(self.Lv, self.advection_rhs, v, 1e-5, self.num_vel_dof[None])
        self.assign_velocity(v)
        self.compute_div_v()
        self.fill_pressure_rhs()
        p = ti.ndarray(float, self.num_pressure_dof[None])
        conjugate_gradient(self.Lp, self.pressure_rhs, p, 1e-5, self.num_pressure_dof[None])
        self.assign_pressure(p)
        self.compute_gradp()
        self.update_velocity()

        




if __name__ == '__main__':
    ti.init(default_fp=ti.f32)
    simulator = ImcompressibleFlowSimulation(1, 1, 64, 64, 1000, 1)
    simulator.substep()
    

    

