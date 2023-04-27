import taichi as ti

@ti.data_oriented
class ImcompressibleFlowSimulation:
    def __init__(self, Lx, Ly, nx, ny):
        self.vx = ti.field(dtype=ti.f32, shape=(nx+1, ny))
        self.vy = ti.field(dtype=ti.f32, shape=(nx, ny+1))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))
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


        