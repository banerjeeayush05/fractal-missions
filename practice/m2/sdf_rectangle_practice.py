import numpy as np

n, dx = 128, 1.0
x = (np.arange(n) - n/2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

# a rectangle of width a and height b: phi < 0 inside
a, b = 40.0, 20.0

dx = np.abs(X) - a
dy = np.abs(Y) - b

inside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
outside = min(max(dx, dy), 0)

phi = inside + outside