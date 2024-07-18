import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_geo import CassegrainGeometry, Point, Hyperbolic, Parabolic

def calculate_normal(point, mirror):
    if isinstance(mirror, Parabolic):
        return np.array([2 * point.x, 2 * point.y, -2 * mirror.radius_curv])
    elif isinstance(mirror, Hyperbolic):
        R = mirror.radius_curv
        return np.array([point.x / np.sqrt(R**2 + 3 * (point.x**2 + point.y**2)), 
                         point.y / np.sqrt(R**2 + 3 * (point.x**2 + point.y**2)), 
                         1])

# uses R = I - 2(I.N)N to calc reflected ray
def bounce_1(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal
def bounce_2(ray_direction, normal):
    return -ray_direction - 4 * np.dot(ray_direction, normal) * normal

def trace_ray(telescope, point):
    # Calculate the direction of the ray to the primary mirror (vertical downward)
    ray_direction = np.array([0, 0, -1])
    
    # Find the intersection with the primary mirror
    z_primary = 0  # The primary mirror is at z = 0
    t_primary = (z_primary - point.z) / ray_direction[2]
    target_primary = Point(
        point.x + t_primary * ray_direction[0],
        point.y + t_primary * ray_direction[1],
        z_primary
    )
    
    # Calculate the normal to the primary mirror at the target point
    normal_primary = calculate_normal(target_primary, telescope.primary)
    reflected_primary = bounce_1(ray_direction, normal_primary)
    
    # Calculate intersection with the secondary mirror
    z_secondary = telescope.secondary.z_position
    t_secondary = (z_secondary - target_primary.z) / reflected_primary[2]
    target_secondary = Point(
        target_primary.x + t_secondary * reflected_primary[0],
        target_primary.y + t_secondary * reflected_primary[1],
        z_secondary
    )
    
    # Calculate the normal to the secondary mirror at the intersection point
    normal_secondary = calculate_normal(target_secondary, telescope.secondary)
    reflected_secondary = bounce_2(reflected_primary, normal_secondary)

    return point, target_primary, target_secondary, reflected_secondary

def visualize_rays(telescope, rays):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for ray_path in rays:
        if ray_path is None:
            continue

        point, target_primary, target_secondary, reflected_secondary = ray_path

        # Plot the initial point
        ax.scatter(point.x, point.y, point.z, color='blue', marker='o', label='Atmospheric Particle')

        # Plot the primary reflection
        ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', label='Photon Ray')
        ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='steelblue', label='Primary Reflection')

        # Plot the green line using the halved angle of reflection
        green_line_length = 6.0  # Adjust this length as needed
        green_endpoint = Point(target_secondary.x + green_line_length * reflected_secondary[0],
                               target_secondary.y + green_line_length * reflected_secondary[1],
                               target_secondary.z + green_line_length * reflected_secondary[2])
        ax.plot([target_secondary.x, green_endpoint.x], [target_secondary.y, green_endpoint.y], [target_secondary.z, green_endpoint.z], color='palevioletred', label='Secondary Reflection')

    # Create a mask for the circular mirrors
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius

    # Plot the primary mirror with a circular mask
    theta = np.linspace(0, 2 * np.pi, 100)
    r_primary = 5  # Actual radius of the primary mirror
    x_primary = np.linspace(-r_primary, r_primary, 100)
    y_primary = np.linspace(-r_primary, r_primary, 100)
    X_primary, Y_primary = np.meshgrid(x_primary, y_primary)
    mask_primary = circular_mask(X_primary, Y_primary, r_primary)
    Z_primary = np.zeros_like(X_primary)
    Z_primary[mask_primary] = (X_primary[mask_primary]**2 + Y_primary[mask_primary]**2) / (4 * telescope.primary.radius_curv)
    Z_primary[~mask_primary] = np.nan
    ax.plot_wireframe(X_primary, Y_primary, Z_primary, color='green', alpha=0.3)

    # Plot the secondary mirror with a circular mask
    r_secondary = 0.7  # Actual radius of the secondary mirror
    x_secondary = np.linspace(-r_secondary, r_secondary, 100)
    y_secondary = np.linspace(-r_secondary, r_secondary, 100)
    X_secondary, Y_secondary = np.meshgrid(x_secondary, y_secondary)
    mask_secondary = circular_mask(X_secondary, Y_secondary, r_secondary)
    Z_secondary = np.zeros_like(X_secondary)
    Z_secondary[mask_secondary] = telescope.secondary.surface(X_secondary[mask_secondary], Y_secondary[mask_secondary])
    Z_secondary[~mask_secondary] = np.nan
    ax.plot_wireframe(X_secondary, Y_secondary, Z_secondary, color='purple', alpha=0.3)

    # Plot focal points for reference
    ax.scatter(0, 0, 17.5, color='green', marker='o', label='Primary Focal Point', alpha = 0.3)
    ax.scatter(0, 0, -2.5, color='purple', marker='o', label='Back Focal Dist', alpha = 0.3)  

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    plt.show()

if __name__ == "__main__":
    cassegrain_geo = CassegrainGeometry(
        primary_radius_curv=34.974, 
        secondary_radius_curv=4.8, 
        primary_K=-1, 
        secondary_K=-4, 
        primary_z_position=0, 
        secondary_z_position=15.28
    )

    test_points = [Point(4.5, -0.3, 20), Point(-3, 2, 17), Point(2.5, -4, 26), Point(1, 3.5, 30), Point(-4.8, -2, 24)]
    rays = [trace_ray(cassegrain_geo, point) for point in test_points]
    visualize_rays(cassegrain_geo, rays)
