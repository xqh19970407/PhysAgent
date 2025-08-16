"""
====================================================================
Rain Film Flow Simulation on a Moving Window
====================================================================

Author  : Student Agent
Date    : August 16, 2025
Version : 1.0
Language: Python 3.x

Description:
------------
This script simulates the dynamics of rainwater film on a window,
including main droplet movement, branching, random rain drops, 
evaporation, and film thinning. Visualization is done via matplotlib
animation.

Modules:
--------
- build_2d_laplacian(): Construct 2D Laplacian operator
- initialize_fields(): Initialize film thickness and lifetime
- add_main_droplet(): Update main droplet position and volume
- add_branch(): Generate droplet branches
- add_random_drops(): Add random rain droplets
- apply_droplet_fall_and_evap(): Apply droplet fall and evaporation
- update(): Single time-step update function for animation
- run_simulation(): Create and run animation

Requirements:
-------------
- numpy
- scipy
- matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse

# ----------------------
# Physical parameters
# ----------------------
params = {
    'μ': 1e-3,
    'ρ': 1000,
    'g': 9.81,
    'σ': 0.036,
    'τ_air': 36600,
    'L': 1.0,
    'W': 0.5,
    'T': 10.0,
    'branch_prob': 0.85,
    'drop_thresh': 8e-2,
    'drop_amount': 5e-5,
    'drop_amount_branch': 5e-5,
    'evap_rate': 1e-4
}

# ----------------------
# Numerical parameters
# ----------------------
nx, ny = 400, 200
dx = params['L'] / nx
dy = params['W'] / ny
dt = 0.0002
nt = int(params['T'] / dt)

# ----------------------
# Field initialization
# ----------------------
def initialize_fields(nx, ny):
    h = np.zeros((ny, nx))
    h_life = np.zeros_like(h)
    return h, h_life

# ----------------------
# 2D Laplacian operator
# ----------------------
def build_2d_laplacian(nx, ny, dx, dy):
    D2x = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    D2y = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    Laplacian = sparse.kron(sparse.eye(ny), D2x) + sparse.kron(D2y, sparse.eye(nx))
    return Laplacian

# ----------------------
# Main droplet update
# ----------------------
def add_main_droplet(h, main_x, main_y, direction, frame, nx, ny, h_min_move,
                     branch_count, last_branch_y, N_branch_target):
    max_wave = 3
    h_center = h[main_y, main_x]
    wave_calc = max_wave * (1 - h_center / 0.001)
    wave_amplitude = max(0, int(wave_calc))
    step_x = - round(1 / np.tan(np.radians(15))) * direction
    step_x += np.random.randint(-wave_amplitude, wave_amplitude + 1)
    step_y = -1
    main_x += step_x
    main_y += step_y

    main_x = np.clip(main_x, 3, nx - 3)
    main_y = np.clip(main_y, 3, ny - 3)

    # Add droplet volume
    if frame % 50 == 0:
        h[max(0, main_y-1):min(ny, main_y+2),
          max(0, main_x-1):min(nx, main_x+2)] += 5e-4

    return main_x, main_y

# ----------------------
# Branch generation
# ----------------------
def add_branch(h, main_x, main_y, branch_count, last_branch_y, nx, ny):
    if branch_count < 5 and np.random.rand() < params['branch_prob'] and main_y < last_branch_y - 20:
        branch_count += 1
        last_branch_y = main_y
        branch_len = np.random.randint(40, 80)
        branch_x, branch_y = main_x, main_y
        for _ in range(branch_len):
            branch_step_x = - round(1 / np.tan(np.radians(15 + np.random.uniform(-15, 15))))
            branch_step_y = -1
            branch_x += branch_step_x + np.random.randint(-1, 2)
            branch_y += branch_step_y + np.random.randint(-1, 2)
            branch_x = np.clip(branch_x, 3, nx - 3)
            branch_y = np.clip(branch_y, 3, ny - 3)
            h[max(0, branch_y-1):min(ny, branch_y+2),
              max(0, branch_x-1):min(nx, branch_x+2)] += params['drop_amount_branch']
    return h, branch_count, last_branch_y

# ----------------------
# Random rain drops
# ----------------------
def add_random_drops(h, nx, ny, frame):
    if frame % 500 == 0:
        rand_x = np.random.randint(0, nx, size=10)
        rand_y = np.random.randint(0, ny, size=10)
        for rx, ry in zip(rand_x, rand_y):
            h[ry, rx] += 2e-4
    return h

# ----------------------
# Droplet fall and evaporation
# ----------------------
def apply_droplet_fall_and_evap(h, h_life):
    drop_mask = h > params['drop_thresh']
    h[drop_mask] -= params['drop_amount'] * (h[drop_mask] / params['drop_thresh'])

    h_life[h>0] += dt
    h[h_life>3.0] = 0
    h_life[h==0] = 0

    h -= params['evap_rate'] * dt
    h = np.maximum(h, 0)
    return h, h_life

# ----------------------
# Animation update
# ----------------------
def update(frame, im, h, h_life, Laplacian, main_x, main_y, direction,
           branch_count, last_branch_y):
    # Driving force
    u = (h**2 / (3 * params['μ'])) * (-params['τ_air'])
    v = (h**2 / (3 * params['μ'])) * (-params['ρ'] * params['g'])
    flow_mask = h > 1e-5
    u[~flow_mask] = 0
    v[~flow_mask] = 0

    flux_x = h * u
    flux_y = h * v

    div_flux = np.zeros_like(h)
    div_flux[:, 1:-1] -= (flux_x[:, 2:] - flux_x[:, :-2]) / (2 * dx)
    div_flux[1:-1, :] -= (flux_y[2:, :] - flux_y[:-2, :]) / (2 * dy)

    h_flat = h.flatten()
    lap_h = Laplacian @ h_flat
    surface_term = (params['σ'] * 3 * h_flat**3 / (3 * params['μ'])) * lap_h
    surface_term = surface_term.reshape(ny, nx)

    h += dt * (div_flux + surface_term)
    h = np.maximum(h, 0)

    main_x, main_y = add_main_droplet(h, main_x, main_y, direction, frame, nx, ny, 1e-5, branch_count, last_branch_y, 5)
    h, branch_count, last_branch_y = add_branch(h, main_x, main_y, branch_count, last_branch_y, nx, ny)
    h = add_random_drops(h, nx, ny, frame)
    h, h_life = apply_droplet_fall_and_evap(h, h_life)

    display_h = np.where(h > 1e-6, h, 0)
    grad_h_x = np.gradient(display_h, dx, axis=1)**2
    grad_h_y = np.gradient(display_h, dy, axis=0)**2
    grad_h = np.sqrt(grad_h_x + grad_h_y)
    im.set_data(np.sqrt(display_h + 5e-5 * grad_h))
    im.set_clim(vmin=0, vmax=0.001)

    return im,

# ----------------------
# Run simulation
# ----------------------
def run_simulation():
    h, h_life = initialize_fields(nx, ny)
    Laplacian = build_2d_laplacian(nx, ny, dx, dy)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(h, cmap='plasma', origin='lower',
                   extent=[0, params['L'], 0, params['W']],
                   vmin=0, vmax=0.001)
    ax.set_xlabel('Window Length (m)')
    ax.set_ylabel('Window Height (m)')

    main_x, main_y, direction = nx - 5, ny - 5, 1
    branch_count, last_branch_y = 0, ny + 20

    ani = FuncAnimation(fig, update, frames=nt, interval=10, blit=True,
                        fargs=(im, h, h_life, Laplacian, main_x, main_y, direction,
                               branch_count, last_branch_y))
    plt.tight_layout()
    plt.savefig('', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_simulation()
