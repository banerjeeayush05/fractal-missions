import numpy as np

V = 1.0
R = 30.0

n, dx = 128, 1.0
x = (np.arange(n) - n/2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

phi = np.sqrt(X**2 + Y**2) - R

def godunov_loop(phi, dx, V):
    nx, ny = phi.shape
    out = np.zeros_like(phi)

    for i in range(nx):
        for j in range(ny):
            im, ip = (i - 1) % nx, (i + 1) % nx
            jm, jp = (j - 1) % ny, (j + 1) % ny

            a = (phi[i, j] - phi[im, j]) / dx
            b = (phi[ip, j] - phi[i, j]) / dx
            c = (phi[i, j] - phi[i, jm]) / dx
            d = (phi[i, jp] - phi[i, j]) / dx

            v = V[i, j] if np.ndim(V) else V

            if v > 0:
                gx = max(a, 0.0)**2 + min(b, 0.0)**2
                gy = max(c, 0.0)**2 + min(d, 0.0)**2
            else:
                gx = min(a, 0.0)**2 + max(b, 0.0)**2
                gy = min(c, 0.0)**2 + max(d, 0.0)**2

            out[i, j] = np.sqrt(gx + gy)
    return out

def phi_radius(phi, dx):
    row = phi[:, n//2]                      # a line through the centre
    i = np.where((row[:-1] < 0) & (row[1:] >= 0))[0][-1]
    r = x[i] + dx * (-row[i]) / (row[i+1] - row[i])
    return r

t, dt = 10, 0.1
t = np.arange(0, t, dt)

for i in range(len(t)):
    phi = phi - V * dt * godunov_loop(phi, dx, V)
    actual_radius = R + V * t[i]
    if i % 10 == 0:
        measured_radius = phi_radius(phi, dx)
        print(f"t = {t[i]:.1f}, measured radius = {measured_radius:.4f}, actual radius = {actual_radius:.4f}, error = {100*(measured_radius-actual_radius)/actual_radius:+.4f} %")