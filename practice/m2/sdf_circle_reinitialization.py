import numpy as np

n, dx = 128, 1.0
x = (np.arange(n) - n/2 + 0.5) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

phi = np.sqrt(X**2 + Y**2) - 30.0        # circle of radius 30, phi < 0 inside

def area(phi):                                        # FIX: a function, so it re-reads phi
    return (0.5*(1 - np.tanh(phi/(1.5*dx)))).sum() * dx*dx

def grad_mag(phi, S):                                 # FIX: upwind, not np.gradient
    a = (phi - np.roll(phi,  1, 0))/dx
    b = (np.roll(phi, -1, 0) - phi)/dx
    c = (phi - np.roll(phi,  1, 1))/dx
    d = (np.roll(phi, -1, 1) - phi)/dx
    pos = np.maximum(a,0)**2 + np.minimum(b,0)**2 + np.maximum(c,0)**2 + np.minimum(d,0)**2
    neg = np.minimum(a,0)**2 + np.maximum(b,0)**2 + np.minimum(c,0)**2 + np.maximum(d,0)**2
    return np.sqrt(np.where(S > 0, pos, neg))

a0 = area(phi)
print(a0)

phi0 = phi.copy()                                     # FIX: frozen reference
S = phi0/np.sqrt(phi0**2 + dx**2)                     # smoothed sign (swap for np.sign(phi0))
# S = np.sign(phi0)                     # smoothed sign (swap for np.sign(phi0))

for i in range(5):
    phi = phi - 0.5*dx * S * (grad_mag(phi, S) - 1.0) # FIX: dtau = 0.5*dx

a1 = area(phi)
print(a1)
print(f"drift = {100*(a1-a0)/a0:+.4f} %")