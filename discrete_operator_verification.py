import taichi as ti
import numpy as np

from fluid_simulator import ImcompressibleFlowSimulation


mx = 2
my = 4

@ti.data_oriented
class DiscreteOperatorTester():
    def __init__(self, Lx, Ly, nx, ny) -> None:
        self.simulator = ImcompressibleFlowSimulation(Lx, Ly, nx, ny)
        self.simulator.reset()
        self.simulator.set_wall_vel(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.error_gradp_x = ti.field(dtype=float, shape=())
        self.error_gradp_y = ti.field(dtype=float, shape=())
        self.error_div_v = ti.field(dtype=float, shape=())
        self.error_laplacian_vx = ti.field(dtype=float, shape=())
        self.error_laplacian_vy = ti.field(dtype=float, shape=())

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
            if I[0] == 0 or I[0] == self.simulator.nx:
                continue
            x = I[0] * self.simulator.dx
            y = I[1] * self.simulator.dy + 0.5 * self.simulator.dy
            lap_vec = self.analytical_laplacian_v(x, y)
            # self.error_laplacian_vx[None] += (lap_vec[0] - self.simulator.laplacian_vx[I]) ** 2 * self.simulator.dx * self.simulator.dy
            x_l = x - self.simulator.dx
            x_r = x + self.simulator.dx
            y_b = y - self.simulator.dy
            y_t = y + self.simulator.dy
            numerical_laplacian_x = (self.analytical_vx(x_r, y) - 2 * self.analytical_vx(x, y) + self.analytical_vx(x_l, y)) / self.simulator.dx ** 2 \
                + (self.analytical_vx(x, y_t) - 2 * self.analytical_vx(x, y) + self.analytical_vx(x, y_b)) / self.simulator.dy ** 2

            self.error_laplacian_vx[None] += (numerical_laplacian_x - lap_vec[0]) ** 2 * self.simulator.dx * self.simulator.dy
        
        # ti.loop_config(serialize=True)
        for I in ti.grouped(self.simulator.laplacian_vy):
            if I[1] == 0 or I[1] == self.simulator.ny:
                continue
            x = I[0] * self.simulator.dx + 0.5 * self.simulator.dx
            y = I[1] * self.simulator.dy
            lap_vec = self.analytical_laplacian_v(x, y)
            # self.error_laplacian_vy[None] += (lap_vec[1] - self.simulator.laplacian_vy[I]) ** 2 * self.simulator.dx * self.simulator.dy

            x_l = x - self.simulator.dx
            x_r = x + self.simulator.dx
            y_b = y - self.simulator.dy
            y_t = y + self.simulator.dy
            numerical_laplacian_y = (self.analytical_vy(x, y_t) - 2 * self.analytical_vy(x, y) + self.analytical_vy(x, y_b)) / self.simulator.dy ** 2 \
                + (self.analytical_vy(x_r, y) - 2 * self.analytical_vy(x, y) + self.analytical_vy(x_l, y)) / self.simulator.dx ** 2
            # print("Numerical laplacian: ", numerical_laplacian_x, " ", numerical_laplacian_y)
            # print("Analytical laplacian: ", " ", lap_vec[0], lap_vec[1])
            # print("Computed laplacian: ", self.simulator.laplacian_vx[I], " ", self.simulator.laplacian_vy[I])

            self.error_laplacian_vy[None] += (lap_vec[1] - numerical_laplacian_y) ** 2 * self.simulator.dx * self.simulator.dy


        print("Error gradp_x: ", ti.sqrt(self.error_gradp_x[None]))
        print("Error gradp_y: ", ti.sqrt(self.error_gradp_y[None]))
        print("Error div_v: ", ti.sqrt(self.error_div_v[None]))
        print("Error laplacian_vx: ", ti.sqrt(self.error_laplacian_vx[None]))
        print("Error laplacian_vy: ", ti.sqrt(self.error_laplacian_vy[None]))
        



if __name__ == '__main__':
    ti.init(default_fp=ti.f32)
    tester = DiscreteOperatorTester(1.0, 1.0, 10, 10)
    tester.fill_data()
    tester.simulator.compute_div_v()
    tester.simulator.compute_gradp()
    tester.simulator.compute_laplacian_v()
    tester.compute_error()
    import pdb
    pdb.set_trace()
    