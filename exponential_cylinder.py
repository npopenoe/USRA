import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

N = 10000
points = []
decay_rate = 2.0  # adjust the rate to control the decay

for _ in range(N):
    r = np.sqrt(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(0, 1)
    
    # apply exponential decay -- f(x) = le^(-lx) where l = 1/decayrate    
    z_weighted = np.random.exponential(scale=decay_rate) # range (0, 10)
    z_weighted = np.clip(z_weighted, 0, 10) # values less than 0 set to 0 and greater than 10 set to 10
    points.append(Point(x, y, z_weighted))

# model points with decay
xs = [p.x for p in points]
ys = [p.y for p in points]
zs = [p.z for p in points]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, marker='.', alpha=0.4)
plt.show()