import taichi as ti
import numpy as np
import argparse
import matplotlib.pyplot as plt


ti.init(arch=ti.gpu, default_fp=ti.f32)

argparser = argparse.ArgumentParser()
argparser.add_argument('--nx', type=int, default=100)
argparser.add_argument('--scheme', type=str, default='central')
argparser.add_argument('--T', type=float, default=1)

args = argparser.parse_args()
nx = args.nx
dx = 1 / nx
c = 1

if args.scheme == 'central':
    type=0
    nt = int(1/dx ** 2) + 1
elif args.scheme == 'upwind':
    type=1
    nt = int(1/dx) + 1
dt_max = 1 / nt

f_in = ti.field(dtype=float, shape=(nx+1))
f_out = ti.field(dtype=float, shape=(nx+1))
f_exact = ti.field(dtype=float, shape=(nx+1))
error = ti.field(dtype=float, shape=())

@ti.kernel
def assign_initial_value():
    for i in range(nx+1):
        x = i * dx
        if 0.1 <= x <= 0.2:
            f_in[i] = 1

@ti.kernel
def exact_solution(T: float):
    for i in range(nx+1):
        x = i * dx - c * T - ti.floor(i * dx - c * T)
        if 0.1 <= x <= 0.2:
            f_exact[i] = 1

@ti.kernel
def compute_error():
    error[None] = 0
    for i in range(nx):
        error[None] += dx * (f_out[i] - f_exact[i])**2
    error[None] = ti.sqrt(error[None])

@ti.kernel
def timestep(dt: float):
    if ti.static(type == 0):
        for j in range(1, nx):
            #central difference
            f_out[j] = f_in[j] - c * dt / (dx * 2) * (f_in[j+1] - f_in[j-1])
        f_out[0] = f_in[0] - c * dt / (dx * 2) * (f_in[1] - f_in[nx-1])
        f_out[nx] = f_out[0]

    elif ti.static(type == 1):
        for j in range(1, nx):
            f_out[j] = f_in[j] - c * dt / dx * (f_in[j] - f_in[j-1])
        f_out[0] = f_in[0] - c * dt / dx * (f_in[0] - f_in[nx-1])
        f_out[nx] = f_in[0]
    
    for j in range(nx+1):
        f_in[j] = f_out[j]

assign_initial_value()

plt.plot(np.arange(0, 1+0.5*dx, dx),f_in.to_numpy(), linestyle='-', label='T=0', linewidth=3)

t = 0
T = float(args.T)
while t < T:
    if T - t < dt_max:
        timestep(T - t)
        t = T
    else:
        timestep(dt_max)
        t += dt_max
    if abs(t - T/2) < dt_max/2:
        plt.plot(np.arange(0, 1+0.5*dx, dx),f_in.to_numpy(), linestyle='--', label=f't={T/2}', linewidth=3)

plt.plot(np.arange(0, 1+0.5*dx, dx),f_in.to_numpy(), linestyle='-.', label=f't={T}', linewidth=3)
plt.legend()

exact_solution(T)
compute_error()

print("========== INFO ============")
print(f"dx = {dx}")
print(f"dt = {dt_max}")
print(f"T = {T}")
print(f"scheme = {args.scheme}")
print(f"error(T) = {error[None]}")
print("============================")

plt.show()


# grid = f_out.to_numpy()
# import matplotlib.pyplot as plt
# plt.plot(np.arange(0, 1+0.5*dx, dx),grid)
# plt.show()
