import taichi as ti
import numpy as np
import sys
import argparse


ti.init(arch=ti.gpu, default_fp=ti.f32)

argparser = argparse.ArgumentParser()
argparser.add_argument('--nx', type=int, default=100)
argparser.add_argument('--scheme', type=str, default='central')

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
dt= 1 / nt

f = ti.field(dtype=float, shape=(nt+1, nx+1))

@ti.kernel
def assign_initial_value():
    for i in range(nx+1):
        x = i * dx
        if 0.1 <= x <= 0.2:
            f[0, i] = 1

@ti.kernel
def timestep(i: ti.i32):
    if ti.static(type == 0):
        for j in range(1, nx):
            #central difference
            f[i+1, j] = f[i, j] - c * dt / (dx * 2) * (f[i, j+1] - f[i, j-1])
        f[i+1, 0] = f[i, 0] - c * dt / (dx * 2) * (f[i, 1] - f[i, nx-1])
        f[i+1, nx] = f[i+1, 0]

    elif ti.static(type == 1):
        for j in range(1, nx):
            f[i+1, j] = f[i, j] - c * dt / dx * (f[i, j] - f[i, j-1])
        f[i+1, 0] = f[i, 0] - c * dt / dx * (f[i, 0] - f[i, nx-1])
        f[i+1, nx] = f[i+1, 0]

assign_initial_value()
for i in range(nt):
    timestep(i)

grid = f.to_numpy()

import matplotlib.pyplot as plt
plt.plot(np.arange(0, 1+0.5*dx, dx), grid[0])
plt.plot(np.arange(0, 1+0.5*dx, dx),grid[nt//3])
plt.plot(np.arange(0, 1+0.5*dx, dx),grid[nt//3*2])
plt.plot(np.arange(0, 1+0.5*dx, dx),grid[nt])
plt.legend(['t=0', 't=1/3', 't=2/3', 't=1'])
plt.show()
