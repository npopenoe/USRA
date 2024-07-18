import numpy as np

class Point: 
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

class Parabolic:
    def __init__(self, radius_curv, K_val, z_position):
        self.radius_curv = radius_curv
        self.K_val = K_val
        self.z_position = z_position

    def reflect(self, point):
        x, y = point.x, point.y
        z = (self.radius_curv + np.sqrt(self.radius_curv**2 - (self.K_val + 1) * (x**2 + y**2))) / (self.K_val + 1)
        return Point(x, y, z)

    def calculate_normal(self, point):
        return np.array([2 * point.x, 2 * point.y, -2 * self.radius_curv + 2 * (self.K_val + 1) * point.z])

class Hyperbolic:
    def __init__(self, radius_curv, K_val, z_position):
        self.radius_curv = radius_curv
        self.K_val = K_val
        self.z_position = z_position

    def surface(self, x, y):
        z = self.z_position + (self.radius_curv - np.sqrt(self.radius_curv**2 - (self.K_val + 1) * (x**2 + y**2))) / (self.K_val + 1)
        return z

    def calculate_normal(self, point):
        return np.array([point.x/(np.sqrt(self.radius_curv**2 + 3*(point.x**2+point.y**2))), point.y/(np.sqrt(self.radius_curv**2 + 3*(point.x**2+point.y**2))), 1])

class CassegrainGeometry:
    def __init__(self, primary_radius_curv, secondary_radius_curv, primary_K, secondary_K, primary_z_position, secondary_z_position):
        self.primary = Parabolic(primary_radius_curv, primary_K, primary_z_position)
        self.secondary = Hyperbolic(secondary_radius_curv, secondary_K, secondary_z_position)

