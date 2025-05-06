import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

# 1. Parameters
num_planes = 5
grid_y_cells_base = 10
grid_x_cells_base = grid_y_cells_base * 2
grid_y_cells = grid_y_cells_base * 4
grid_x_cells = grid_x_cells_base * 4
plane_spacing = 5
tunnel_half_width_x0 = 0.8
tunnel_half_width_y0 = 0.5
tunnel_growth_per_plane = 0.1

# 2. Create Figure and 3D Axes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove subplot margins

# 3. Create X and Y Grid for Planes
x = np.linspace(0, grid_x_cells_base, grid_x_cells + 1)
y = np.linspace(0, grid_y_cells_base, grid_y_cells + 1)
X, Y = np.meshgrid(x, y)

# 4. Plot Planes and Random Points
for i in range(num_planes):
    Z = np.full_like(X, i * plane_spacing)
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3, edgecolor='black', linewidth=0.3)

    rand_x = np.random.uniform(0, grid_x_cells_base, 700)
    rand_y = np.random.uniform(0, grid_y_cells_base, 700)
    rand_z = np.full(700, i * plane_spacing)
    ax.scatter(rand_x, rand_y, rand_z, color='blue', s=3, alpha=0.5)

# 5. Define Red Linear Track
x0 = grid_x_cells_base / 3
y0 = grid_y_cells_base / 3
x_slope = 0.8
y_slope = 0.5
line_x = [x0 + i * x_slope for i in range(num_planes)]
line_y = [y0 + i * y_slope for i in range(num_planes)]
line_z = [i * plane_spacing for i in range(num_planes)]
ax.plot(line_x, line_y, line_z, color='red', marker='o', markersize=4, linewidth=1.5)

# 6. Create Tunnel Rectangles
rectangles = []
for i in range(num_planes):
    cx, cy, cz = line_x[i], line_y[i], line_z[i]
    hwx = tunnel_half_width_x0 + i * tunnel_growth_per_plane
    hwy = tunnel_half_width_y0 + i * tunnel_growth_per_plane
    corners = [
        (cx - hwx, cy - hwy, cz),
        (cx + hwx, cy - hwy, cz),
        (cx + hwx, cy + hwy, cz),
        (cx - hwx, cy + hwy, cz),
    ]
    rectangles.append(corners)

# 7. Draw Tunnel Wireframe
for corners in rectangles:
    for j in range(4):
        x1, y1, z1 = corners[j]
        x2, y2, z2 = corners[(j + 1) % 4]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='green', linewidth=1.5)

# 8. Connect Rectangles Between Planes
for i in range(num_planes - 1):
    for j in range(4):
        x1, y1, z1 = rectangles[i][j]
        x2, y2, z2 = rectangles[i + 1][j]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='green', linewidth=1.5)

# 9. Create Tunnel Faces
faces = []
for i in range(num_planes - 1):
    r1 = rectangles[i]
    r2 = rectangles[i + 1]
    for j in range(4):
        face = [r1[j], r1[(j + 1) % 4], r2[(j + 1) % 4], r2[j]]
        faces.append(face)
faces.append(rectangles[0])
faces.append(rectangles[-1][::-1])
tunnel_poly = Poly3DCollection(faces, facecolors='green', alpha=0.2, edgecolor=None)
ax.add_collection3d(tunnel_poly)

# 10. Compute Axis Ranges and Set Limits
xrange = grid_x_cells_base
yrange = grid_y_cells_base
zrange = num_planes * plane_spacing
ax.set_xlim(0, xrange)
ax.set_ylim(0, yrange)
ax.set_zlim(0, zrange)
ax.set_box_aspect([xrange, yrange, zrange])

# 11. Hide Axes
ax.set_axis_off()

# 12. Animation Function
def rotate(angle):
    ax.view_init(elev=20, azim=angle)
    return (ax,)

# 13. Create the Animation
num_frames = 360
interval = 50
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, num_frames, 2),
                              interval=interval)

# 14. Save the Animation as a GIF
ani.save('rotating_tunnel_3d.gif', writer='pillow', fps=20)
plt.close()

print("Animation saved as rotating_tunnel_3d.gif")

