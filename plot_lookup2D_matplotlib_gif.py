# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# # 1. Parameters
# num_planes = 5  # Total number of planes (for reference, though we'll only use one)
# grid_y_cells_base = 10
# grid_x_cells_base = grid_y_cells_base * 2
# plane_spacing = 5
# tunnel_half_width_x0 = 0.8
# tunnel_half_width_y0 = 0.5
# tunnel_growth_per_plane = 0.1
#
# # 2. Select the Plane
# plane_index = 2  # Let's pick the third plane (index 2)
#
# # 3. Calculate Plane-Specific Data
# plane_z = plane_index * plane_spacing
# plane_x_range = grid_x_cells_base
# plane_y_range = grid_y_cells_base
#
# # 4. Create Figure and 2D Axes
# fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
# ax.set_aspect('equal')  # Ensure equal aspect ratio for 2D
#
# # 5. Generate Points for the Selected Plane
# rand_x = np.random.uniform(0, grid_x_cells_base, 700)
# rand_y = np.random.uniform(0, grid_y_cells_base, 700)
#
# # 6. Define Rectangle for the Selected Plane
# cx = grid_x_cells_base / 3 + plane_index * 0.8  # Adjusted x-center for the selected plane
# cy = grid_y_cells_base / 3 + plane_index * 0.5  # Adjusted y-center for the selected plane
# hwx = tunnel_half_width_x0 + plane_index * tunnel_growth_per_plane
# hwy = tunnel_half_width_y0 + plane_index * tunnel_growth_per_plane
# rect_corners = [
#     (cx - hwx, cy - hwy),
#     (cx + hwx, cy - hwy),
#     (cx + hwx, cy + hwy),
#     (cx - hwx, cy + hwy),
# ]
#
# # 7. Plot Static Elements (Points and Rectangle)
# point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
# rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
# ax.add_patch(rect)
#
# # 8. Set Axis Limits (based on the grid size)
# ax.set_xlim(0, grid_x_cells_base)
# ax.set_ylim(0, grid_y_cells_base)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title(f'2D View of Plane {plane_index + 1}')
#
# # 9. Animation Function: Update Gridlines
# def animate(frame):
#     """
#     Function to update the gridlines for each frame of the animation.
#
#     Args:
#         frame (int): The current frame number.
#     """
#     ax.clear()  # Clear the axes for redrawing
#
#     # Plot static elements (again, to ensure they're on top of the grid)
#     point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
#     rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
#     ax.add_patch(rect)
#
#     # Calculate grid density based on the frame number.
#     num_grid_lines = frame # let the frame number control the number of grid lines
#
#     # Create gridlines
#     x_grid = np.linspace(0, grid_x_cells_base, num_grid_lines + 1) #number of lines is frame number
#     y_grid = np.linspace(0, grid_y_cells_base, num_grid_lines + 1)
#
#     # Draw gridlines
#     for x_line in x_grid:
#         ax.plot([x_line, x_line], [0, grid_y_cells_base], color='gray', linestyle='--', linewidth=0.5)
#     for y_line in y_grid:
#         ax.plot([0, grid_x_cells_base], [y_line, y_line], color='gray', linestyle='--', linewidth=0.5)
#
#     #restore the axis limits
#     ax.set_xlim(0, grid_x_cells_base)
#     ax.set_ylim(0, grid_y_cells_base)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'2D View of Plane {plane_index + 1}')
#
#     return ()  # Return an empty tuple, as required by FuncAnimation
#
# # 10. Create the Animation
# num_frames = 20 # Adjust the number of frames for the animation duration
# interval = 50  # Adjust for animation speed
# ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, num_frames), interval=interval)
#
# # 11. Save the Animation as a GIF
# ani.save('2d_plane_animation.gif', writer='pillow', fps=10)  # Adjust fps as needed
# plt.close()
# print("2D animation with increasing grid density saved as 2d_plane_animation.gif")


# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# # 1. Parameters
# num_planes = 5  # Total number of planes (for reference, though we'll only use one)
# grid_y_cells_base = 10
# grid_x_cells_base = grid_y_cells_base * 2
# plane_spacing = 5
# tunnel_half_width_x0 = 0.8
# tunnel_half_width_y0 = 0.5
# tunnel_growth_per_plane = 0.1
#
# # 2. Select the Plane
# plane_index = 2  # Let's pick the third plane (index 2)
#
# # 3. Calculate Plane-Specific Data
# plane_z = plane_index * plane_spacing
# plane_x_range = grid_x_cells_base
# plane_y_range = grid_y_cells_base
#
# # 4. Create Figure and 2D Axes
# fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
# ax.set_aspect('equal')  # Ensure equal aspect ratio for 2D
#
# # 5. Generate Points for the Selected Plane
# rand_x = np.random.uniform(0, grid_x_cells_base, 700)
# rand_y = np.random.uniform(0, grid_y_cells_base, 700)
#
# # 6. Define Rectangle for the Selected Plane
# cx = grid_x_cells_base / 3 + plane_index * 0.8  # Adjusted x-center for the selected plane
# cy = grid_y_cells_base / 3 + plane_index * 0.5  # Adjusted y-center for the selected plane
# hwx = tunnel_half_width_x0 + plane_index * tunnel_growth_per_plane
# hwy = tunnel_half_width_y0 + plane_index * tunnel_growth_per_plane
# rect_corners = [
#     (cx - hwx, cy - hwy),
#     (cx + hwx, cy - hwy),
#     (cx + hwx, cy + hwy),
#     (cx - hwx, cy + hwy),
# ]
#
# # 7. Plot Static Elements (Points and Rectangle)
# point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
# rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
# ax.add_patch(rect)
#
# # 8. Set Axis Limits (based on the grid size)
# ax.set_xlim(0, grid_x_cells_base)
# ax.set_ylim(0, grid_y_cells_base)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title(f'2D View of Plane {plane_index + 1}')
#
# # 9. Animation Function: Update Gridlines
# def animate(frame):
#     """
#     Function to update the gridlines for each frame of the animation,
#     ensuring square-like grid cells.
#
#     Args:
#         frame (int): The current frame number.
#     """
#     ax.clear()  # Clear the axes for redrawing
#
#     # Plot static elements (again, to ensure they're on top of the grid)
#     point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
#     rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
#     ax.add_patch(rect)
#
#     # Calculate grid density, ensuring square-like cells
#     num_grid_lines_x = frame  # Number of grid lines in x-direction
#     num_grid_lines_y = int(frame * (grid_y_cells_base / grid_x_cells_base))  # Scale y-lines based on aspect ratio
#
#     # Ensure at least 1 grid line in each direction.
#     num_grid_lines_x = max(1, num_grid_lines_x)
#     num_grid_lines_y = max(1, num_grid_lines_y)
#
#     # Create gridlines
#     x_grid = np.linspace(0, grid_x_cells_base, num_grid_lines_x + 1)
#     y_grid = np.linspace(0, grid_y_cells_base, num_grid_lines_y + 1)
#
#     # Draw gridlines
#     for x_line in x_grid:
#         ax.plot([x_line, x_line], [0, grid_y_cells_base], color='gray', linestyle='--', linewidth=0.5)
#     for y_line in y_grid:
#         ax.plot([0, grid_x_cells_base], [y_line, y_line], color='gray', linestyle='--', linewidth=0.5)
#
#     # Restore the axis limits
#     ax.set_xlim(0, grid_x_cells_base)
#     ax.set_ylim(0, grid_y_cells_base)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'2D View of Plane {plane_index + 1}')
#
#     return ()  # Return an empty tuple, as required by FuncAnimation
#
# # 10. Create the Animation
# num_frames = 20  # Adjust the number of frames for the animation duration
# interval = 50  # Adjust for animation speed
# ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, num_frames), interval=interval)
#
# # 11. Save the Animation as a GIF
# ani.save('2d_plane_animation.gif', writer='pillow', fps=10)  # Adjust fps as needed
# plt.close()
# print("2D animation with increasing grid density saved as 2d_plane_animation.gif")



# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# # 1. Parameters
# num_planes = 5  # Total number of planes (for reference, though we'll only use one)
# grid_y_cells_base = 10
# grid_x_cells_base = grid_y_cells_base * 2
# plane_spacing = 5
# tunnel_half_width_x0 = 0.8
# tunnel_half_width_y0 = 0.5
# tunnel_growth_per_plane = 0.1
#
# # 2. Select the Plane
# plane_index = 2  # Let's pick the third plane (index 2)
#
# # 3. Calculate Plane-Specific Data
# plane_z = plane_index * plane_spacing
# plane_x_range = grid_x_cells_base
# plane_y_range = grid_y_cells_base
#
# # 4. Create Figure and 2D Axes
# fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
# ax.set_aspect('equal')  # Ensure equal aspect ratio for 2D
#
# # 5. Generate Points for the Selected Plane
# rand_x = np.random.uniform(0, grid_x_cells_base, 700)
# rand_y = np.random.uniform(0, grid_y_cells_base, 700)
#
# # 6. Define Rectangle for the Selected Plane
# cx = grid_x_cells_base / 3 + plane_index * 0.8  # Adjusted x-center for the selected plane
# cy = grid_y_cells_base / 3 + plane_index * 0.5  # Adjusted y-center for the selected plane
# hwx = tunnel_half_width_x0 + plane_index * tunnel_growth_per_plane
# hwy = tunnel_half_width_y0 + plane_index * tunnel_growth_per_plane
# rect_corners = [
#     (cx - hwx, cy - hwy),
#     (cx + hwx, cy - hwy),
#     (cx + hwx, cy + hwy),
#     (cx - hwx, cy + hwy),
# ]
#
# # 7. Plot Static Elements (Points and Rectangle)
# point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
# rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
# ax.add_patch(rect)
#
# # 8. Set Axis Limits (based on the grid size)
# ax.set_xlim(0, grid_x_cells_base)
# ax.set_ylim(0, grid_y_cells_base)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title(f'2D View of Plane {plane_index + 1}')
#
# # 9. Animation Function: Update Gridlines
# def animate(frame):
#     """
#     Function to update the gridlines for each frame of the animation,
#     ensuring square-like grid cells.
#
#     Args:
#         frame (int): The current frame number.
#     """
#     ax.clear()  # Clear the axes for redrawing
#
#     # Plot static elements (again, to ensure they're on top of the grid)
#     point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
#     rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
#     ax.add_patch(rect)
#
#     # Calculate grid density, ensuring square-like cells
#     num_grid_lines_x = frame  # Number of grid lines in x-direction
#     num_grid_lines_y = int(frame * (grid_y_cells_base / grid_x_cells_base))  # Scale y-lines based on aspect ratio
#
#     # Ensure at least 1 grid line in each direction.
#     num_grid_lines_x = max(1, num_grid_lines_x)
#     num_grid_lines_y = max(1, num_grid_lines_y)
#
#     # Create gridlines
#     x_grid = np.linspace(0, grid_x_cells_base, num_grid_lines_x + 1)
#     y_grid = np.linspace(0, grid_y_cells_base, num_grid_lines_y + 1)
#
#     # Draw gridlines
#     for x_line in x_grid:
#         ax.plot([x_line, x_line], [0, grid_y_cells_base], color='gray', linestyle='--', linewidth=0.5)
#     for y_line in y_grid:
#         ax.plot([0, grid_x_cells_base], [y_line, y_line], color='gray', linestyle='--', linewidth=0.5)
#
#     # Restore the axis limits
#     ax.set_xlim(0, grid_x_cells_base)
#     ax.set_ylim(0, grid_y_cells_base)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'2D View of Plane {plane_index + 1}')
#
#     return ()  # Return an empty tuple, as required by FuncAnimation
#
# # 10. Create the Animation
# num_frames = 100  # Increased the number of frames from 50 to 100
# interval = 50  # Adjust for animation speed
# ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, num_frames), interval=interval)
#
# # 11. Save the Animation as a GIF
# ani.save('2d_plane_animation.gif', writer='pillow', fps=10)  # Adjust fps as needed
# plt.close()
# print("2D animation with increasing grid density saved as 2d_plane_animation.gif")



# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# # 1. Parameters
# num_planes = 5  # Total number of planes (for reference, though we'll only use one)
# grid_y_cells_base = 10
# grid_x_cells_base = grid_y_cells_base * 2
# plane_spacing = 5
# tunnel_half_width_x0 = 0.8
# tunnel_half_width_y0 = 0.5
# tunnel_growth_per_plane = 0.1
#
# # 2. Select the Plane
# plane_index = 2  # Let's pick the third plane (index 2)
#
# # 3. Calculate Plane-Specific Data
# plane_z = plane_index * plane_spacing
# plane_x_range = grid_x_cells_base
# plane_y_range = grid_y_cells_base
#
# # 4. Create Figure and 2D Axes
# fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
# ax.set_aspect('equal')  # Ensure equal aspect ratio for 2D
#
# # 5. Generate Points for the Selected Plane
# rand_x = np.random.uniform(0, grid_x_cells_base, 700)
# rand_y = np.random.uniform(0, grid_y_cells_base, 700)
#
# # 6. Define Rectangle for the Selected Plane
# cx = grid_x_cells_base / 3 + plane_index * 0.8  # Adjusted x-center for the selected plane
# cy = grid_y_cells_base / 3 + plane_index * 0.5  # Adjusted y-center for the selected plane
# hwx = tunnel_half_width_x0 + plane_index * tunnel_growth_per_plane
# hwy = tunnel_half_width_y0 + plane_index * tunnel_growth_per_plane
# rect_corners = [
#     (cx - hwx, cy - hwy),
#     (cx + hwx, cy - hwy),
#     (cx + hwx, cy + hwy),
#     (cx - hwx, cy + hwy),
# ]
#
# # 7. Define Red Point for the Selected Plane
# x0 = grid_x_cells_base / 3
# y0 = grid_y_cells_base / 3
# x_slope = 0.8
# y_slope = 0.5
# red_point_x = x0 + plane_index * x_slope
# red_point_y = y0 + plane_index * y_slope
#
# # 8. Plot Static Elements (Points, Rectangle, and Red Point)
# point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
# rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
# ax.add_patch(rect)
# red_point = ax.plot(red_point_x, red_point_y, 'ro', markersize=4)[0]  # Plot the red point
#
# # 9. Set Axis Limits (based on the grid size)
# ax.set_xlim(0, grid_x_cells_base)
# ax.set_ylim(0, grid_y_cells_base)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title(f'2D View of Plane {plane_index + 1}')
#
# # 10. Animation Function: Update Gridlines
# def animate(frame):
#     """
#     Function to update the gridlines for each frame of the animation,
#     ensuring square-like grid cells.
#
#     Args:
#         frame (int): The current frame number.
#     """
#     ax.clear()  # Clear the axes for redrawing
#
#     # Plot static elements (again, to ensure they're on top of the grid)
#     point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
#     rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
#     ax.add_patch(rect)
#     red_point = ax.plot(red_point_x, red_point_y, 'ro', markersize=4)[0]  # Redraw the red point
#
#     # Calculate grid density, ensuring square-like cells
#     num_grid_lines_x = frame  # Number of grid lines in x-direction
#     num_grid_lines_y = int(frame * (grid_y_cells_base / grid_x_cells_base))  # Scale y-lines based on aspect ratio
#
#     # Ensure at least 1 grid line in each direction.
#     num_grid_lines_x = max(1, num_grid_lines_x)
#     num_grid_lines_y = max(1, num_grid_lines_y)
#
#     # Create gridlines
#     x_grid = np.linspace(0, grid_x_cells_base, num_grid_lines_x + 1)
#     y_grid = np.linspace(0, grid_y_cells_base, num_grid_lines_y + 1)
#
#     # Draw gridlines
#     for x_line in x_grid:
#         ax.plot([x_line, x_line], [0, grid_y_cells_base], color='gray', linestyle='--', linewidth=0.5)
#     for y_line in y_grid:
#         ax.plot([0, grid_x_cells_base], [y_line, y_line], color='gray', linestyle='--', linewidth=0.5)
#
#     # Restore the axis limits
#     ax.set_xlim(0, grid_x_cells_base)
#     ax.set_ylim(0, grid_y_cells_base)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'2D View of Plane {plane_index + 1}')
#
#     return ()  # Return an empty tuple, as required by FuncAnimation
#
# # 10. Create the Animation
# num_frames = 100
# interval = 50
# ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, num_frames), interval=interval)
#
# # 11. Save the Animation as a GIF
# ani.save('2d_plane_animation.gif', writer='pillow', fps=10)
# plt.close()
# print("2D animation with increasing grid density saved as 2d_plane_animation.gif")



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# 1. Parameters
num_planes = 5  # Total number of planes (for reference, though we'll only use one)
grid_y_cells_base = 10
grid_x_cells_base = grid_y_cells_base * 2
plane_spacing = 5
tunnel_half_width_x0 = 0.8
tunnel_half_width_y0 = 0.5
tunnel_growth_per_plane = 0.1

# 2. Select the Plane
plane_index = 2  # Let's pick the third plane (index 2)

# 3. Calculate Plane-Specific Data
plane_z = plane_index * plane_spacing
plane_x_range = grid_x_cells_base
plane_y_range = grid_y_cells_base

# 4. Create Figure and 2D Axes
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
ax.set_aspect('equal')  # Ensure equal aspect ratio for 2D

# 5. Generate Points for the Selected Plane
rand_x = np.random.uniform(0, grid_x_cells_base, 700)
rand_y = np.random.uniform(0, grid_y_cells_base, 700)

# 6. Define Rectangle for the Selected Plane
cx = grid_x_cells_base / 3 + plane_index * 0.8  # Adjusted x-center for the selected plane
cy = grid_y_cells_base / 3 + plane_index * 0.5  # Adjusted y-center for the selected plane
hwx = tunnel_half_width_x0 + plane_index * tunnel_growth_per_plane
hwy = tunnel_half_width_y0 + plane_index * tunnel_growth_per_plane
rect_corners = [
    (cx - hwx, cy - hwy),
    (cx + hwx, cy - hwy),
    (cx + hwx, cy + hwy),
    (cx - hwx, cy + hwy),
]

# 7. Define Red Point for the Selected Plane
x0 = grid_x_cells_base / 3
y0 = grid_y_cells_base / 3
x_slope = 0.8
y_slope = 0.5
red_point_x = x0 + plane_index * x_slope
red_point_y = y0 + plane_index * y_slope

# 8. Plot Static Elements (Points, Rectangle, and Red Point)
point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
ax.add_patch(rect)
red_point = ax.plot(red_point_x, red_point_y, 'ro', markersize=4)[0]  # Plot the red point

# 9. Set Axis Limits (based on the grid size)
ax.set_xlim(0, grid_x_cells_base)
ax.set_ylim(0, grid_y_cells_base)
# ax.set_xlabel('X')  # Removed x-axis label
# ax.set_ylabel('Y')  # Removed y-axis label
ax.set_title(f'2D View of Plane {plane_index + 1}')

# 10. Animation Function: Update Gridlines
def animate(frame):
    """
    Function to update the gridlines for each frame of the animation,
    ensuring square-like grid cells.

    Args:
        frame (int): The current frame number.
    """
    ax.clear()  # Clear the axes for redrawing

    # Plot static elements (again, to ensure they're on top of the grid)
    point_scatter = ax.scatter(rand_x, rand_y, color='blue', s=3, alpha=0.5)
    rect = plt.Polygon(rect_corners, color='green', alpha=0.2, edgecolor='green', linewidth=1.5)
    ax.add_patch(rect)
    red_point = ax.plot(red_point_x, red_point_y, 'ro', markersize=4)[0]  # Redraw the red point

    # Calculate grid density, ensuring square-like cells
    num_grid_lines_x = frame  # Number of grid lines in x-direction
    num_grid_lines_y = int(frame * (grid_y_cells_base / grid_x_cells_base))  # Scale y-lines based on aspect ratio

    # Ensure at least 1 grid line in each direction.
    num_grid_lines_x = max(1, num_grid_lines_x)
    num_grid_lines_y = max(1, num_grid_lines_y)
    
    # Create gridlines
    x_grid = np.linspace(0, grid_x_cells_base, num_grid_lines_x + 1)
    y_grid = np.linspace(0, grid_y_cells_base, num_grid_lines_y + 1)

    # Draw gridlines
    for x_line in x_grid:
        ax.plot([x_line, x_line], [0, grid_y_cells_base], color='gray', linestyle='--', linewidth=0.5)
    for y_line in y_grid:
        ax.plot([0, grid_x_cells_base], [y_line, y_line], color='gray', linestyle='--', linewidth=0.5)

    # Restore the axis limits
    ax.set_xlim(0, grid_x_cells_base)
    ax.set_ylim(0, grid_y_cells_base)
    # ax.set_xlabel('X')  # Removed x-axis label
    # ax.set_ylabel('Y')  # Removed y-axis label
    ax.set_title(f'2D View of Plane {plane_index + 1}')

    # Remove axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    return ()  # Return an empty tuple, as required by FuncAnimation

# 10. Create the Animation
num_frames = 100
interval = 50
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, num_frames), interval=interval)

# 11. Save the Animation as a GIF
ani.save('2d_plane_animation.gif', writer='pillow', fps=10)
plt.close()
print("2D animation with increasing grid density saved as 2d_plane_animation.gif")
