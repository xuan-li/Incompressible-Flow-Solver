import taichi as ti
import numpy as np

@ti.data_oriented
class ImcompressibleFlowSimulation:
    def __init__(self, Lx, Ly, nx, ny):
        self.vx = ti.field(dtype=ti.f32, shape=(nx+1, ny))
        self.vy = ti.field(dtype=ti.f32, shape=(nx, ny+1))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = 0.01

    def reset(self):
        self.vx.fill(0)
        self.vy.fill(0)
        self.pressure.fill(0)
        self.vorticity.fill(0)

    @ti.kernel
    def set_wall_vel(self, components: ti.types.ndarray()):
        # components = [u_l, u_r, u_b, u_t]
        for i in range(self.nx+1):
            self.vx[i, 0] = components[0]
            self.vx[i, self.ny-1] = components[1]
        for j in range(self.ny+1):
            self.vy[0, j] = components[2]
            self.vy[self.nx-1, j] = components[3]
    
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


if __name__ == '__main__':
    ti.init(default_fp=ti.f32)
    simulator = ImcompressibleFlowSimulation(1, 1, 64, 64)
    simulator.test_sample()

    import matplotlib.pyplot as plt
    # add four subplots

    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    u_img = np.zeros([simulator.nx, simulator.ny])
    simulator.visualize_u(u_img)
    plt.imshow(u_img)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('vx')

    plt.subplot(222)
    v_img = np.zeros([simulator.nx, simulator.ny])
    simulator.visualize_v(v_img)
    plt.imshow(v_img)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('vy')

    plt.subplot(223)
    p_img = np.zeros([simulator.nx, simulator.ny])
    simulator.visualize_p(p_img)
    plt.imshow(p_img)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('pressure')

    plt.subplot(224)
    vorticity_img = np.zeros([simulator.nx, simulator.ny])
    simulator.visualize_vorticity(vorticity_img)
    plt.imshow(vorticity_img)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('vorticity')

    plt.show()
    

    

