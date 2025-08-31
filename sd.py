import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sphere radius
r = 3

# Create spherical coordinates meshgrid
phi = np.linspace(0, np.pi, 100)  # polar angle from 0 to pi
theta = np.linspace(0, 2 * np.pi, 100)  # azimuthal angle from 0 to 2pi
phi, theta = np.meshgrid(phi, theta)

# Convert spherical coordinates to Cartesian for the sphere surface
x_sphere = r * np.sin(phi) * np.cos(theta)
y_sphere = r * np.sin(phi) * np.sin(theta)
z_sphere = r * np.cos(phi)

# Plot the sphere surface
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.3, edgecolor='none')

# Plot the paraboloid surface z = x^2 + y^2
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z_paraboloid = X**2 + Y**2

# Mask values of Z that are above 9 (outside the volume)
Z_paraboloid[Z_paraboloid > 9] = np.nan

# Plot the paraboloid surface
ax.plot_surface(X, Y, Z_paraboloid, color='orange', alpha=0.5)

# Plot the plane z = 9 as a transparent surface to visualize the top boundary
Z_plane = 9 * np.ones_like(X)
ax.plot_surface(X, Y, Z_plane, color='green', alpha=0.2)

# Set limits and labels
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 9])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of the Integration Region')

plt.show()