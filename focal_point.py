import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_telescope import CassegrainTelescope, Point, HyperbolicMirror, ParabolicMirror

# Choose a specific point on the primary mirror
def specific_point_on_primary_mirror(mirror):
    x = 3
    y = 0
    z = (x**2 + y**2) / (4 * mirror.focal_length)
    return Point(x, y, z)

# Use the gradient to calculate the normal vector for both mirrors
def calculate_normal(point, mirror):
    if isinstance(mirror, ParabolicMirror):
        return np.array([2 * point.x / (4 * mirror.focal_length), 2 * point.y / (4 * mirror.focal_length), -1])
    elif isinstance(mirror, HyperbolicMirror):
        return np.array([2 * point.x / mirror.focal_length, 2 * point.y / mirror.focal_length, -1])

# Use R = I - 2(I.N)N to calculate the reflected ray
def reflect(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal

# Handle the calculation of bouncing rays
def trace_ray(telescope, point):
    target_primary = specific_point_on_primary_mirror(telescope.primary)

    # Direction of the ray from the initial point to the target point on the primary mirror is calculated and normalized
    ray_direction = np.array([target_primary.x - point.x, target_primary.y - point.y, target_primary.z - point.z])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # The normal to the primary mirror at the target point is calculated. The ray is then reflected using this normal
    normal_primary = calculate_normal(target_primary, telescope.primary)
    reflected_primary = reflect(ray_direction, normal_primary)

    # Intersection of the reflected ray with the secondary mirror
    t_secondary = telescope.secondary.intersect(target_primary, reflected_primary)
    if t_secondary is None:
        return None

    target_secondary = Point(target_primary.x + t_secondary * reflected_primary[0],
                             target_primary.y + t_secondary * reflected_primary[1],
                             target_primary.z + t_secondary * reflected_primary[2])

    # Calculate the normal to the secondary mirror at the intersection point and reflect the ray
    normal_secondary = calculate_normal(target_secondary, telescope.secondary)
    reflected_secondary = reflect(reflected_primary, normal_secondary)

    # Intersection of the secondary reflected ray with the focal plane of the primary mirror
    t_focal = (telescope.primary.focal_length - target_secondary.z) / reflected_secondary[2]
    focal_point = Point(target_secondary.x + t_focal * reflected_secondary[0],
                        target_secondary.y + t_focal * reflected_secondary[1],
                        telescope.primary.focal_length)

    return point, target_primary, target_secondary, focal_point, normal_secondary

def visualize_single_ray_path(telescope, ray_path):
    if ray_path is None:
        print("No successful ray found")
        return

    point, target_primary, target_secondary, focal_point, normal_secondary = ray_path

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial point
    ax.scatter(point.x, point.y, point.z, color='blue', marker='o', label='Initial Point')

    # Plot the primary reflection
    ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', label='Primary Reflection')
    ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='red', label='Secondary Reflection')

    # Plot the primary focal point
    primary_focal_point = Point(0, 0, telescope.primary.focal_length)
    ax.scatter(primary_focal_point.x, primary_focal_point.y, primary_focal_point.z, color='green', marker='o', label='Primary Focal Point')

    # Plot the primary mirror
    theta = np.linspace(0, 2 * np.pi, 100)
    r = telescope.primary.diameter / 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2) / (4 * telescope.primary.focal_length)
    ax.plot_wireframe(X, Y, Z, color='green', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    plt.legend()
    plt.show()

# Test point located directly above mirror
if __name__ == "__main__":
    telescope = CassegrainTelescope(primary_focal_length=17.5, secondary_focal_length=21, primary_diameter=10, secondary_diameter=1.4, secondary_position_z=4.5)
    test_point = Point(3, 0, 20)  # Using the specific test point
    ray_path = trace_ray(telescope, test_point)
    visualize_single_ray_path(telescope, ray_path)
