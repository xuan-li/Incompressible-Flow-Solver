import taichi as ti
import numpy as np
from fluid_simulator import ImcompressibleFlowSimulation

mx = 2
my = 4

@ti.data_oriented
class DiscreteOperatorTester():
    def __init__(self, Lx, Ly, nx, ny) -> None:
        self.simulator = ImcompressibleFlowSimulation(Lx, Ly, nx, ny, 1)
        self.error_gradp_x = ti.field(dtype=float, shape=())
        self.error_gradp_y = ti.field(dtype=float, shape=())
        self.error_div_v = ti.field(dtype=float, shape=())
        self.error_laplacian_vx = ti.field(dtype=float, shape=())
        self.error_laplacian_vy = ti.field(dtype=float, shape=())
        self.error_advect_vx = ti.field(dtype=float, shape=())
        self.error_advect_vy = ti.field(dtype=float, shape=())

    @ti.func
    def analytical_vx(self, x, y):
        return ti.sin(mx * np.pi * x) * ti.cos(my * np.pi * y)

    @ti.func
    def analytical_vy(self, x, y):
        return ti.cos(mx * np.pi * x) * ti.sin(my * np.pi * y)
    
    @ti.func
    def analytical_pressure(self, x, y):
        return ti.sin(mx * np.pi * x) * ti.cos(mx * np.pi * y)
    
    @ti.func
    def analytical_pressure_grad_x(self, x, y):
        return mx * np.pi * ti.cos(mx * np.pi * x) * ti.cos(mx * np.pi * y)
    
    @ti.func
    def analytical_pressure_grad_y(self, x, y):
        return -mx * np.pi * ti.sin(mx * np.pi * x) * ti.sin(mx * np.pi * y)
    
    @ti.func
    def analytical_div_v(self, x, y):
        grad_vx_x = mx * np.pi * ti.cos(mx * np.pi * x) * ti.cos(my * np.pi * y)
        grad_vy_y = my * np.pi * ti.cos(mx * np.pi * x) * ti.cos(my * np.pi * y)
        return grad_vx_x + grad_vy_y

    @ti.func
    def analytical_laplacian_v(self, x, y):
        laplacian_vx = -(mx * mx + my * my) * np.pi * np.pi * ti.sin(mx * np.pi * x) * ti.cos(my * np.pi * y)
        laplacian_vy = -(mx * mx + my * my) * np.pi * np.pi * ti.cos(mx * np.pi * x) * ti.sin(my * np.pi * y)
        return ti.Vector([laplacian_vx, laplacian_vy], float)

    @ti.func
    def analytical_advection(self, x, y):
        duu_dx = np.pi * mx * ti.sin(2 * mx * np.pi * x) * (ti.cos(my * np.pi * y) ** 2)
        duv_dy = 0.5 * np.pi * my * ti.sin(2 * mx * np.pi * x) * ti.cos(2 * my * np.pi * y)
        duv_dx = 0.5 * np.pi * mx * ti.cos(2 * mx * np.pi * x) * ti.sin(2 * my * np.pi * y)
        dvv_dy = np.pi * my * (ti.cos(mx * np.pi * x) ** 2) * ti.sin(2 * my * np.pi * y)
        return ti.Vector([duu_dx + duv_dy, duv_dx + dvv_dy], float)
    
    @ti.kernel
    def fill_data(self):
        for I in ti.grouped(self.simulator.vx):
            x = I[0] * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            self.simulator.vx[I] = self.analytical_vx(x, y)
                
        for I in ti.grouped(self.simulator.vy):
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy
            self.simulator.vy[I] = self.analytical_vy(x, y)
        
        for I in ti.grouped(self.simulator.pressure):
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            self.simulator.pressure[I] = self.analytical_pressure(x, y)
    
    @ti.kernel
    def compute_error(self):
        self.error_div_v[None] = 0.0
        self.error_gradp_x[None] = 0.0
        self.error_gradp_y[None] = 0.0
        self.error_laplacian_vx[None] = 0.0
        self.error_laplacian_vy[None] = 0.0

        for I in ti.grouped(self.simulator.gradp_x):
            if I[0] == 0 or I[0] == self.simulator.nx:
                continue
            x = I[0] * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            self.error_gradp_x[None] += (self.analytical_pressure_grad_x(x, y) - self.simulator.gradp_x[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.gradp_y):
            if I[1] == 0 or I[1] == self.simulator.ny:
                continue
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy
            self.error_gradp_y[None] += (self.analytical_pressure_grad_y(x, y) - self.simulator.gradp_y[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.div_v):
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            self.error_div_v[None] += (self.analytical_div_v(x, y) - self.simulator.div_v[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.laplacian_vx):
            if I[0] == 0 or I[0] == self.simulator.nx or I[1] == 0 or I[1] == self.simulator.ny-1:
                continue
            x = I[0] * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            lap_vec = self.analytical_laplacian_v(x, y)
            self.error_laplacian_vx[None] += (lap_vec[0] - self.simulator.laplacian_vx[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.laplacian_vy):
            if I[1] == 0 or I[1] == self.simulator.ny or I[0] == 0 or I[0] == self.simulator.nx-1:
                continue
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy
            lap_vec = self.analytical_laplacian_v(x, y)
            self.error_laplacian_vy[None] += (lap_vec[1] - self.simulator.laplacian_vy[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.advect_vx):
            if I[0] == 0 or I[0] == self.simulator.nx:
                continue
            x = I[0] * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            self.error_advect_vx[None] += (self.analytical_advection(x, y)[0] - self.simulator.advect_vx[I]) ** 2 * self.simulator.dx * self.simulator.dy
        
        for I in ti.grouped(self.simulator.advect_vy):
            if I[1] == 0 or I[1] == self.simulator.ny:
                continue
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy
            self.error_advect_vy[None] += (self.analytical_advection(x, y)[1] - self.simulator.advect_vy[I]) ** 2 * self.simulator.dx * self.simulator.dy

        print("dx: ", self.simulator.dx)
        print("Error gradp_x: ", ti.sqrt(self.error_gradp_x[None]))
        print("Error gradp_y: ", ti.sqrt(self.error_gradp_y[None]))
        print("Error div_v: ", ti.sqrt(self.error_div_v[None]))
        print("Error laplacian_vx: ", ti.sqrt(self.error_laplacian_vx[None]))
        print("Error laplacian_vy: ", ti.sqrt(self.error_laplacian_vy[None]))
        print("Error advect_vx: ", ti.sqrt(self.error_advect_vx[None]))
        print("Error advect_vy: ", ti.sqrt(self.error_advect_vy[None]))
        

if __name__ == '__main__':
    ti.init(default_fp=ti.f64, arch=ti.cuda)
    import sys
    N = int(sys.argv[1])
    tester = DiscreteOperatorTester(1.0, 1.0, N, N)
    tester.fill_data()
    tester.simulator.compute_div_v()
    tester.simulator.compute_gradp()
    tester.simulator.compute_laplacian_v()
    tester.simulator.compute_advection()
    tester.compute_error()
    