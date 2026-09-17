import numpy as np

n, dx = 128, 1.0
x = (np.arange(n) - n/2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

# a circle of radius 30: phi < 0 inside
phi = np.sqrt(X**2 + Y**2) - 30.0

# check: |grad phi| should be 1 everywhere for a true SDF
gy, gx = np.gradient(phi, dx)
print(np.abs(np.sqrt(gx**2 + gy**2) - 1))