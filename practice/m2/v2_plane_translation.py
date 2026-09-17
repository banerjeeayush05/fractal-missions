import numpy as np

n, dx = 128, 1.0
x = (np.arange(n) - n/2 + 0.5) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

V, dt, N = 1.0, 0.1, 120
y0 = x[0] + 5*dx

def grad_mag(phi, V):
    a = (phi - np.roll(phi,  1, 0)) / dx
    b = (np.roll(phi, -1, 0) - phi) / dx

    lo = 2*phi[:, :1]  - phi[:, 1:2]
    hi = 2*phi[:, -1:] - phi[:, -2:-1]
    p  = np.concatenate([lo, phi, hi], axis=1)
    c = (phi - p[:, :-2]) / dx
    d = (p[:, 2:] - phi) / dx

    pos = (np.maximum(a,0)**2 + np.minimum(b,0)**2 +
           np.maximum(c,0)**2 + np.minimum(d,0)**2)
    neg = (np.minimum(a,0)**2 + np.maximum(b,0)**2 +
           np.minimum(c,0)**2 + np.maximum(d,0)**2)
    return np.sqrt(np.where(V > 0, pos, neg))

def interface_y(phi):
    ys = []
    for i in range(n):
        col = phi[i]
        k = np.where((col[:-1] < 0) & (col[1:] >= 0))[0][-1]
        ys.append(x[k] + dx * (-col[k]) / (col[k+1] - col[k]))
    return np.array(ys)

phi = Y - y0
for _ in range(N):
    phi = phi - dt * V * grad_mag(phi, V)

pos      = interface_y(phi)
expected = y0 + V * dt * N
error    = pos.mean() - expected
flatness = pos.max() - pos.min()

print(f"expected {expected:.6f}")
print(f"measured {pos.mean():.6f}")
print(f"error    {error:+.3e}")
print(f"flatness {flatness:.3e}")

assert abs(error) < 1e-12, "V2 position"
assert flatness  < 1e-12, "V2 flatness"
print("V2 PASS")