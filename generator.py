import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pickle


# Physical constants
e = 1.602176634e-19  # elementary charge in C
c = 299792458  # speed of light in m/s
m_positron = 9.1093837015e-31  # positron mass in kg
q_positron = e  # positron charge in C

# Convert units
m_to_cm = 100
m_to_mm = 1000
cm_to_mm = 10
cm_to_m = 0.01
mm_to_m = 0.001
mm_to_cm = 0.1
kG_to_T = 0.1

### chip
npix_x = 1024
npix_y = 512
pix_x  = 0.02924
pix_y  = 0.02688
chipXmm = npix_x*pix_x
chipYmm = npix_y*pix_y
chipXcm = chipXmm*mm_to_cm
chipYcm = chipYmm*mm_to_cm
chipXm  = chipXmm*mm_to_m
chipYm  = chipYmm*mm_to_m

print(f"chipXmm={chipXmm}, chipYmm={chipYmm}")
print(f"chipXcm={chipXcm}, chipYcm={chipYcm}")
print(f"chipXm={chipXm},   chipYm={chipYm}")


# Define magnet elements
class Element:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min * cm_to_m
        self.x_max = x_max * cm_to_m
        self.y_min = y_min * cm_to_m
        self.y_max = y_max * cm_to_m
        self.z_min = z_min * cm_to_m
        self.z_max = z_max * cm_to_m
        # print(f"x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, z_min={self.z_min}, z_max={self.z_max}")
        
    def is_inside(self, x, y, z):
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max and 
                self.z_min <= z <= self.z_max)
                
    def get_vertices(self):
        vertices = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
        ])
        return vertices
        
    def plot_element(self, ax, color, alpha=0.2):
        vertices = self.get_vertices()
        
        # List of sides' vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
        ]
        
        # Plot sides
        for face in faces:
            face = np.array(face)
            ax.plot_surface(
                face[:, 0].reshape(2, 2),
                face[:, 1].reshape(2, 2),
                face[:, 2].reshape(2, 2),
                color=color, alpha=alpha
            )

class Dipole(Element):
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, B_x):
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max)
        self.B_x = B_x  # Tesla
        self.hits = []
        
    def field(self, x, y, z):
        if self.is_inside(x, y, z):
            return np.array([self.B_x, 0, 0])
        else:
            return np.array([0, 0, 0])
    
    def record_hit(self, particle_id, x, y, z, px, py, pz):
        self.hits.append({
            'particle_id': particle_id,
            'x': x, 'y': y, 'z': z,
            'px': px, 'py': py, 'pz': pz
        })

class Quadrupole(Element):
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, gradient):
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max)
        self.gradient = gradient * kG_to_T  # Tesla/m
        self.hits = []
        
    def field(self, x, y, z):
        if self.is_inside(x, y, z):
            B_x = self.gradient * y
            B_y = self.gradient * x
            return np.array([B_x, B_y, 0])
        else:
            return np.array([0, 0, 0])
    
    def record_hit(self, particle_id, x, y, z, px, py, pz):
        self.hits.append({
            'particle_id': particle_id,
            'x': x, 'y': y, 'z': z,
            'px': px, 'py': py, 'pz': pz
        })
    

# Create magnetic elements
dipole = Dipole(
    x_min=-2.2352, x_max=2.2352,
    y_min=-6.1976, y_max=3.3528,
    z_min=1265, z_max=1363,
    B_x=0.219  # Tesla
)

# Calculate detector z position
detector_z_base_cm = (1363 + 303.2155 + 11.43 + 1.05)  # cm
detector_z_base_m = detector_z_base_cm*cm_to_m
detector_z_base_mm = detector_z_base_cm*cm_to_mm

# Create detector planes
class Detector(Element):
    def __init__(self, x_min, x_max, y_min, y_max, z_pos):
        # For detectors, z_min and z_max are the same (thin plane)
        super().__init__(x_min, x_max, y_min, y_max, z_pos, z_pos)
        self.z_pos = z_pos * cm_to_m
        self.hits = []  # To store particle hits
        
    def field(self, x, y, z):
        # Detectors don't generate magnetic fields
        return np.array([0, 0, 0])
        
    def record_hit(self, particle_id, x, y, z, px, py, pz):
        self.hits.append({
            'particle_id': particle_id,
            'x': x, 'y': y, 'z': z,
            'px': px, 'py': py, 'pz': pz
        })
        
    def plot_element(self, ax, color, alpha=0.4):
        # Override to plot as a thin rectangular plane
        vertices = self.get_vertices()
        
        # Create a rectangular plane
        x = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[4][0]]
        y = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[4][1]]
        z = [vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[4][2]]
        
        # Create z values with a small offset for thickness
        # z_with_thickness = [z_val + 0.007 for z_val in z[:-1]]
        # Plot the detector plane
        # ax.plot_surface(
        #     np.array([x[:-1], x[:-1]]).T,
        #     np.array([y[:-1], y[:-1]]).T,
        #     np.array([z[:-1], z_with_thickness]).T,  # Fixed thickness calculation
        #     color=color, alpha=alpha
        # )
        
        L1verts = []
        L1verts.append( np.array([ [x[0],y[0],z[0]],
                                   [x[1],y[1],z[1]],
                                   [x[2],y[2],z[2]],
                                   [x[3],y[3],z[3]] ]) )
        ax.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
        
    def print_hits(self,initial_states):
        if not self.hits:
            print(f"Detector at z={self.z_pos:.2f} m: No hits recorded")
            return
        
        GeV_c_to_kgms = 5.344286e-19  # 1000x more than MeV/c
        print(f"Detector at z={self.z_pos:.2f} m hits:")
        for hit in self.hits:
            # print(f"  Particle {hit['particle_id']}: x={hit['x']:.6f} m, y={hit['y']:.6f} m")
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            pz  = initial_states[pid][5]/GeV_c_to_kgms
            print(f"  Particle {pid}: x={xx:.6f} m, y={yy:.6f} m (pz={pz:.2f} GeV)")

# Define detector y range
detector_y_center_cm = (5.665 + 0.1525 + 3.685) # cm
detector_y_center_m = detector_y_center_cm*cm_to_m

# Create the detector objects
detectors = []
for i in range(5):
    detector = Detector(
        x_min=-chipYcm/2., x_max=+chipYcm/2.,
        y_min=detector_y_center_cm-chipXcm/2., y_max=detector_y_center_cm+chipXcm/2.,
        z_pos=detector_z_base_cm + i
    )
    detectors.append(detector)

quad0 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=368, z_max=468, # cm
    gradient=7.637  # kG/cm
)

quad1 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=591, z_max=691, # cm
    gradient=-28.55  # kG/cm
)

quad2 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=813, z_max=913, # cm
    gradient=7.637  # kG/cm
)

elements = [quad0, quad1, quad2, dipole] + detectors



# Function to calculate total magnetic field at a point
def total_field(position):
    x, y, z = position
    field = np.zeros(3)
    
    for element in elements:
        field += element.field(x, y, z)
    
    return field






# Define the equations of motion for a charged particle in a magnetic field
def particle_motion(t, state):
    """
    State vector: [x, y, z, px, py, pz]
    where (x, y, z) is position in meters
    and (px, py, pz) is momentum in kg*m/s
    """
    x, y, z, px, py, pz = state
    position = np.array([x, y, z])
    momentum = np.array([px, py, pz])
    
    # Relativistic velocity
    p_mag = np.linalg.norm(momentum)
    gamma = np.sqrt(1 + (p_mag/(m_positron*c))**2)  # Relativistic factor
    velocity = momentum / (gamma * m_positron)
    
    # Magnetic field at current position
    B = total_field(position)
    
    # Lorentz force: F = q(v × B)
    force = q_positron * np.cross(velocity, B)
    
    # Rate of change of position is velocity
    dx_dt = velocity[0]
    dy_dt = velocity[1]
    dz_dt = velocity[2]
    
    # Rate of change of momentum is force
    dpx_dt = force[0]
    dpy_dt = force[1]
    dpz_dt = force[2]
    
    return [dx_dt, dy_dt, dz_dt, dpx_dt, dpy_dt, dpz_dt]



# Define an event function to detect collisions with magnet walls
def collision_event(t, state, elements_to_check=None):
    """
    Event function to detect when a particle hits the wall of any magnet element.
    Returns zero when a collision occurs (negative when inside, positive when outside).
    
    Parameters:
    -----------
    t : float
        Current time value
    state : array 
        Current state [x, y, z, px, py, pz]
    elements_to_check : list, optional
        List of elements to check for collisions. If None, checks all magnetic elements
        
    Returns:
    --------
    float
        Distance from nearest wall (negative inside element boundaries, zero at boundary, positive outside)
    """
    if elements_to_check is None:
        elements_to_check = [quad0, quad1, quad2, dipole] 
    
    x, y, z = state[0], state[1], state[2]
    
    # Initialize distance to a large value
    min_distance = float('inf')
    
    for element in elements_to_check:
        # Skip if not in z-range of element (with small margin)
        if not (element.z_min - 0.0001 <= z <= element.z_max + 0.0001):
            continue
            
        # Calculate distances to each wall
        dx_min = x - element.x_min
        dx_max = element.x_max - x
        dy_min = y - element.y_min
        dy_max = element.y_max - y
        
        # Find minimum distance to wall
        distances = [dx_min, dx_max, dy_min, dy_max]
        element_min_distance = min(distances)
        
        # Update minimum distance if this element's wall is closer
        min_distance = min(min_distance, element_min_distance)
    
    # If not near any element, return a large positive value
    if min_distance == float('inf'):
        return 1.0
        
    return min_distance


# Set terminal attribute when collision occurs
collision_event.terminal = True



def record_hits(element,z_element,trajectory,particle_id):
    for i in range(len(trajectory.t) - 1):
        z1, z2 = trajectory.y[2][i], trajectory.y[2][i+1]
        
        # If particle trajectory crosses the detector plane
        if (z1 <= z_element <= z2) or (z2 <= z_element <= z1):
            # Linear interpolation to find position at detector plane
            t1, t2 = trajectory.t[i], trajectory.t[i+1]
            fraction = (z_element - z1) / (z2 - z1) if z2 != z1 else 0
            
            t_hit = t1 + fraction * (t2 - t1)
            x_hit = trajectory.y[0][i] + fraction * (trajectory.y[0][i+1] - trajectory.y[0][i])
            y_hit = trajectory.y[1][i] + fraction * (trajectory.y[1][i+1] - trajectory.y[1][i])
            
            px_hit = trajectory.y[3][i] + fraction * (trajectory.y[3][i+1] - trajectory.y[3][i])
            py_hit = trajectory.y[4][i] + fraction * (trajectory.y[4][i+1] - trajectory.y[4][i])
            pz_hit = trajectory.y[5][i] + fraction * (trajectory.y[5][i+1] - trajectory.y[5][i])
            
            # Check if hit is within detector surface
            if (element.x_min <= x_hit <= element.x_max and element.y_min <= y_hit <= element.y_max):
                element.record_hit(particle_id, x_hit, y_hit, z_element, px_hit, py_hit, pz_hit)





# Modified function to propagate particle with collision detection
def propagate_particle_with_collision(particle_id, initial_state, t_span, max_step=1e-9):
    """
    Propagate particle through beamline with collision detection.

    Parameters:
    -----------
    particle_id : int
        Unique identifier for the particle
    initial_state : array
        Initial state [x0, y0, z0, px0, py0, pz0]
    t_span : tuple
        (t_start, t_end) for integration
    max_step : float, optional
        Maximum step size for integrator

    Returns:
    --------
    solution : OdeSolution
        Solution object from solve_ivp
    collision_info : dict or None
        Information about collision if it occurred, None otherwise
    """
    # Use solve_ivp with event detection
    solution = solve_ivp(
        particle_motion,
        t_span,
        initial_state,
        method='RK45', #'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'
        events=collision_event,
        max_step=max_step,
        rtol=1e-8,
        atol=1e-10
    )

    # Check for detector crossings as before
    for det in detectors: record_hits(det,det.z_pos,solution,particle_id)
    ### dipole
    record_hits(dipole,dipole.z_max,solution,particle_id)
    ### quads
    record_hits(quad0,quad0.z_max,solution,particle_id)
    record_hits(quad1,quad1.z_max,solution,particle_id)
    record_hits(quad2,quad2.z_max,solution,particle_id)


    # Check if collision occurred
    collision_info = None
    if solution.t_events[0].size > 0:
        # Collision occurred
        collision_time = solution.t_events[0][0]
        collision_state = np.array([
            np.interp(collision_time, solution.t, solution.y[0]),
            np.interp(collision_time, solution.t, solution.y[1]),
            np.interp(collision_time, solution.t, solution.y[2]),
            np.interp(collision_time, solution.t, solution.y[3]),
            np.interp(collision_time, solution.t, solution.y[4]),
            np.interp(collision_time, solution.t, solution.y[5])
        ])

        # Determine which element was hit
        collision_element = None
        for element in [quad0, quad1, quad2, dipole]:
            if (element.z_min <= collision_state[2] <= element.z_max):
                if abs(collision_state[0] - element.x_min) < 1e-6 or abs(collision_state[0] - element.x_max) < 1e-6 or \
                   abs(collision_state[1] - element.y_min) < 1e-6 or abs(collision_state[1] - element.y_max) < 1e-6:
                    collision_element = element
                    break

        element_name = "unknown"
        if collision_element == quad0:
            element_name = "Quadrupole 0"
        elif collision_element == quad1:
            element_name = "Quadrupole 1"
        elif collision_element == quad2:
            element_name = "Quadrupole 2"
        elif collision_element == dipole:
            element_name = "Dipole"

        collision_info = {
            "time": collision_time,
            "position": collision_state[:3],
            "momentum": collision_state[3:],
            "element": element_name
        }

    return solution, collision_info





# Plot the beamline and particle trajectory
def plot_system(particle_trajectories, initial_states, nmaxtrks=100):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot magnetic elements
    quad0.plot_element(ax, 'blue')
    quad1.plot_element(ax, 'green')
    quad2.plot_element(ax, 'blue')
    dipole.plot_element(ax, 'red')
    
    # Plot detector planes
    detector_colors = ['green', 'green', 'green', 'green', 'green']
    for i, detector in enumerate(detectors):
        detector.plot_element(ax, detector_colors[i % len(detector_colors)])
    
    # Plot particle trajectories
    for i, trajectory in enumerate(particle_trajectories):
        if(i>=nmaxtrks): break
        x, y, z = trajectory.y[0], trajectory.y[1], trajectory.y[2]
        ax.plot(x, y, z, '-', linewidth=1.)
    
    # Set axis labels and limits
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 18)  # Extend z limit to include detectors
    
    # Add legend for elements
    quad_patch = mpatches.Patch(color='blue', alpha=0.5, label='Focusing Quad')
    quad_neg_patch = mpatches.Patch(color='green', alpha=0.5, label='Defocusing Quad')
    dipole_patch = mpatches.Patch(color='red', alpha=0.5, label='Dipole')
    detector_patch = mpatches.Patch(color='purple', alpha=0.5, label='Detector')
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([quad_patch, quad_neg_patch, dipole_patch, detector_patch])
    ax.legend(handles=handles, loc='upper right')
    ax.set_title('Charged Particle Propagation Through Magnetic Elements')
    
    plt.tight_layout()
    plt.savefig("generator_tracks.pdf")
    plt.show()
    
    
    
    
    # Create a separate figure for the detector hit patterns
    # fig_det = plt.figure(figsize=(10, 3))
    # det_grid = plt.GridSpec(1, 5, figure=fig_det)
    fig_det, axs = plt.subplots(1, 5, figsize=(10, 4), sharex=True, sharey=True, tight_layout=True)
    
    # Plot detector hit patterns
    for i, detector in enumerate(detectors):
        # ax_det = fig_det.add_subplot(det_grid[i//2, i%2])
        # Draw detector boundary
        rect = plt.Rectangle(
            (detector.x_min, detector.y_min),
            detector.x_max - detector.x_min,
            detector.y_max - detector.y_min,
            fill=False, edgecolor='gray'
        )
        # ax_det.add_patch(rect)
        axs[i].add_patch(rect)
        
        # Plot hits
        hit_x = [hit['x'] for hit in detector.hits]
        hit_y = [hit['y'] for hit in detector.hits]
        particle_ids = [hit['particle_id'] for hit in detector.hits]
        
        for j, (x, y, pid) in enumerate(zip(hit_x, hit_y, particle_ids)):
            # ax_det.plot(x, y, 'o', markersize=3, label=f"Particle {pid+1}" if j == 0 else "")
            # ax_det.plot(x, y, 'o', markersize=2)
            axs[i].plot(x, y, 'o', markersize=1)
        
        # ax_det.set_xlim(detector.x_min * 1.2, detector.x_max * 1.2)
        # ax_det.set_ylim(detector.y_min * 0.9, detector.y_max * 1.1)
        # ax_det.set_xlabel('X [m]')
        # ax_det.set_ylabel('Y [m]')
        # ax_det.set_title(f'Detector {i+1} at z={detector.z_pos:.2f} m')
        # ax_det.grid(True)
        
        axs[i].set_xlim(detector.x_min * 1.2, detector.x_max * 1.2)
        axs[i].set_ylim(detector.y_min * 0.9, detector.y_max * 1.1)
        axs[i].set_xlabel('X [m]')
        axs[i].set_ylabel('Y [m]')
        # axs[i].set_title(f'Detector {i+1} at z={detector.z_pos:.2f} m')
        axs[i].set_title(f'ALPIDE_{i}')
        axs[i].grid(True)
        
        # if i == 0:  # Only add legend to first plot
        #     ax_det.legend()
        
    plt.tight_layout()
    plt.savefig("generator_scatter.pdf")
    plt.show()
    
    # Return both figures for saving if needed
    return fig, fig_det


def truncated_exp_NK(a,b,how_many):
    a = -np.log(a)
    b = -np.log(b)
    rands = np.exp(-(np.random.rand(how_many)*(b-a) + a))
    return rands[0] if(how_many==1) else rands


# Example usage
if __name__ == "__main__":
    # Define example initial conditions for particles
    # Format: [x0, y0, z0, px0, py0, pz0]
    # We'll use momentum in units of GeV/c and convert to kg*m/s
    GeV_c_to_kgms = 5.344286e-19  # 1000x more than MeV/c
    
    # Energy in GeV
    energy_GeV = 2.5
    
    # Calculate momentum for the given energy (assuming relativistic particles)
    # For a positron: E^2 = (pc)^2 + (mc^2)^2
    # => pc = sqrt(E^2 - (mc^2)^2)
    # => p = sqrt(E^2 - (mc^2)^2)/c
    
    mc2_GeV = 0.000511  # Rest energy of positron in GeV
    p_GeV = np.sqrt(energy_GeV**2 - mc2_GeV**2)  # Momentum in GeV/c
    
    p_kgms = p_GeV * GeV_c_to_kgms  # Convert to kg*m/s
    
    Emin = 0.5 ## GeV 
    Emax = 5.0 ## GeV
    sigmax = 0.0001 ## m (100 um)
    sigmay = 0.0001 ## m (100 um)
    sigmaz = 0.0001 ## m (100 um)
    sigmaPx = 0.0007 ## GeV
    sigmaPy = 0.0007 ## GeV
    Nparticles = 2000
    initial_states = []
    PZ0 = []
    for i in range(Nparticles):
        XX = np.random.normal(0.0,sigmax)
        YY = np.random.normal(0.0,sigmay)
        ZZ = np.random.normal(0.0,sigmaz)
        PZ = truncated_exp_NK(Emin,Emax,1)*GeV_c_to_kgms
        PX = np.random.normal(0.0,sigmaPx*GeV_c_to_kgms)
        PY = np.random.normal(0.0,sigmaPy*GeV_c_to_kgms)
        # PX = np.random.normal(XX*GeV_c_to_kgms,sigmaPx*GeV_c_to_kgms)
        # PY = np.random.normal(YY*GeV_c_to_kgms,sigmaPy*GeV_c_to_kgms)
        PZ0.append(PZ/GeV_c_to_kgms)
        state = [XX,YY,ZZ, PX,PY,PZ]
        initial_states.append(state)
    
    
    
    # Time range for propagation (seconds)
    # For relativistic particles going ~c, need to consider the longest path to the detectors
    # Last detector is at ~18 meters
    tmax = 18 / (0.99 * c)  # Approximate time to travel 18 meters
    t_span = (0, tmax)
    
    
    
    # Propagate particles and check for collisions
    particle_trajectories = []
    particle_collisions   = []
    for i,state in enumerate(initial_states):
        solution, collision = propagate_particle_with_collision(i, state, t_span)
        particle_trajectories.append(solution)
        particle_collisions.append(collision)
        # if collision:
        #     print(f"\nParticle {i} collided with {collision['element']} at position:")
        #     print(f"  x = {collision['position'][0]:.6f} m")
        #     print(f"  y = {collision['position'][1]:.6f} m")
        #     print(f"  z = {collision['position'][2]:.6f} m")
        #     print(f"  Time of collision: {collision['time']:.9f} s")
        print(f"done propagating particle {i}")
    
    # Plot the system with particle trajectories
    main_fig, detector_fig = plot_system(particle_trajectories, initial_states, nmaxtrks=100)

    # Print some diagnostics about final positions
    print("\nFinal particle positions:")
    for i, traj in enumerate(particle_trajectories):
        final_x = traj.y[0][-1]
        final_y = traj.y[1][-1]
        final_z = traj.y[2][-1]
        
        px = traj.y[3][-1]
        py = traj.y[4][-1]
        pz = traj.y[5][-1]
        
        p_tot = np.sqrt(px**2 + py**2 + pz**2)
        angle_x = np.arctan2(px, pz) * 180 / np.pi
        angle_y = np.arctan2(py, pz) * 180 / np.pi
                
        # print(f"Particle {i+1}:")
        # print(f"  Vertex:       x={initial_states[i][0]:.6f} m, y={initial_states[i][1]:.6f} m, z={initial_states[i][2]:.6f} m")
        # print(f"  Momentum:     px={initial_states[i][3]/GeV_c_to_kgms:.4f} GeV, py={initial_states[i][4]/GeV_c_to_kgms:.4f} GeV, pz={initial_states[i][5]/GeV_c_to_kgms:.2f} GeV")
        # print(f"  Hit position: x={final_x:.6f} m, y={final_y:.6f} m, z={final_z:.2f} m")
        # print(f"  Exit angles:  θx={angle_x:.3f}°, θy={angle_y:.3f}°")
    
    # # Print detector hits
    # print("\nDetector Hits:")
    # for i, detector in enumerate(detectors):
    #     detector.print_hits(initial_states)
    
    
    
    npivots = 0
    list_good_tracks = []
    for pivot in detectors[0].hits:
        pivot_pid = pivot['particle_id']
        npivots +=1
        nhits = 0
        for i,detector in enumerate(detectors):
            if(i==0): continue
            for hit in detector.hits:
                if(pivot_pid==hit['particle_id']):
                    nhits += 1
                    break
        if(nhits==len(detectors)-1):
            list_good_tracks.append( pivot_pid )
    print(f"Got {len(list_good_tracks)} tracks with {len(detectors)} hits out of {npivots} pivot points at ALPIDE_0")
    
                

            
    
    
    

    ### plot the hits:
    fig, axs = plt.subplots(1, 5, figsize=(10, 4), sharex=True, sharey=True, tight_layout=True)
    P0 = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        Z = []
        P = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            pz  = initial_states[pid][5]
            X.append(xx*m_to_mm)
            Y.append((yy-detector_y_center_m)*m_to_mm)
            Z.append((zz-detector_z_base_m)*m_to_mm)
            if(pid in list_good_tracks): P.append(pz/GeV_c_to_kgms)
        axs[i].hist2d(X, Y, bins=(100,200),range=[[-chipYmm/2,+chipYmm/2],[-chipXmm/2,+chipXmm/2]])
        if(i==0):
            P0 = P
        axs[i].set_xlabel('X [m]')
        axs[i].set_ylabel('Y [m]')
        axs[i].set_title(f'ALPIDE_{i}')
    plt.tight_layout()
    plt.savefig("generator_occupancy.pdf")
    plt.show()
    
    
    ### plot the exit plane at the dipole:
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(pid not in list_good_tracks): continue
        X.append(point['x'])
        Y.append(point['y'])
    ax.hist2d(X, Y, bins=(200,200),range=[[dipole.x_min,dipole.x_max],[dipole.y_min,dipole.y_max]])
    ax.set_xlim(dipole.x_min * 1.2, dipole.x_max * 1.2)
    ax.set_ylim(dipole.y_min * 0.9 if(dipole.y_min>0) else dipole.y_min*1.1, dipole.y_max * 1.1)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Dipole exit')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("generator_dipole_exit_zoom.pdf")
    plt.show()
    
    
    ### plot the exit plane at the dipole:
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(pid not in list_good_tracks): continue
        X.append(point['x'])
        Y.append(point['y'])
    ax.hist2d(X, Y, bins=(120,120),range=[[-80*mm_to_m,+80*mm_to_m],[-70*mm_to_m,+90*mm_to_m]])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Dipole exit')
    rectD = plt.Rectangle(
        (dipole.x_min, dipole.y_min),
        dipole.x_max - dipole.x_min,
        dipole.y_max - dipole.y_min,
        fill=False, edgecolor='blue'
    )
    ax.add_patch(rectD)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("generator_dipole_exit.pdf")
    plt.show()
    
    
    ### plot the quads exits
    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True, tight_layout=True)
    X0 = []
    Y0 = []
    for point in quad0.hits:
        # pid = point['particle_id']
        # if(pid not in list_good_tracks): continue
        X0.append(point['x'])
        Y0.append(point['y'])
    axs[0].hist2d(X0, Y0, bins=(200,200),range=[[quad0.x_min,quad0.x_max],[quad0.y_min,quad0.y_max]])
    axs[0].set_xlim(quad0.x_min * 1.2, quad0.x_max * 1.2)
    axs[0].set_ylim(quad0.y_min * 0.9 if(quad0.y_min>0) else quad0.y_min*1.1, quad0.y_max * 1.1)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')
    axs[0].set_title(f'Quad0 exit')
    # axs[0].grid(True)
    X1 = []
    Y1 = []
    for point in quad1.hits:
        # pid = point['particle_id']
        # if(pid not in list_good_tracks): continue
        X1.append(point['x'])
        Y1.append(point['y'])
    axs[1].hist2d(X1, Y1, bins=(200,200),range=[[quad1.x_min,quad1.x_max],[quad1.y_min,quad1.y_max]])
    axs[1].set_xlim(quad1.x_min * 1.2, quad1.x_max * 1.2)
    axs[1].set_ylim(quad1.y_min * 0.9 if(quad1.y_min>0) else quad1.y_min*1.1, quad1.y_max * 1.1)
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Y [m]')
    axs[1].set_title(f'Quad1 exit')
    # axs[1].grid(True)
    X2 = []
    Y2 = []
    for point in quad2.hits:
        # pid = point['particle_id']
        # if(pid not in list_good_tracks): continue
        X2.append(point['x'])
        Y2.append(point['y'])
    axs[2].hist2d(X2, Y2, bins=(200,200),range=[[quad2.x_min,quad2.x_max],[quad2.y_min,quad2.y_max]])
    axs[2].set_xlim(quad2.x_min * 1.2, quad2.x_max * 1.2)
    axs[2].set_ylim(quad2.y_min * 0.9 if(quad2.y_min>0) else quad2.y_min*1.1, quad2.y_max * 1.1)
    axs[2].set_xlabel('X [m]')
    axs[2].set_ylabel('Y [m]')
    axs[2].set_title(f'Quad2 exit')
    # axs[2].grid(True)
    plt.tight_layout()
    plt.savefig("generator_quads_exit.pdf")
    plt.show()
    
    
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    XX = []
    PX = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(pid not in list_good_tracks): continue
        XX.append(point['x'])
        px = initial_states[pid][3]/GeV_c_to_kgms
        PX.append( px )   
    axs[0].hist2d(XX, PX, bins=(200,200), range=[[-5e-3,+5e-3],[-5e-4,+5e-4]])
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    axs[0].grid(True)
    YY = []
    PY = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(pid not in list_good_tracks): continue
        YY.append(point['y'])
        py = initial_states[pid][4]/GeV_c_to_kgms
        PY.append( py )
    axs[1].hist2d(YY, PY, bins=(200,200), range=[[-5e-3,+5e-3],[-5e-4,+5e-4]])
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig("generator_divergence.pdf")
    plt.show()
    
    
        
    # ### plot the energy
    # fig, ax = plt.subplots(tight_layout=True)
    # ax.hist(P0, bins=50,range=(1.5,3.5))
    # plt.tight_layout()
    # plt.savefig("generator_energy.pdf")
    # plt.show()
    
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True, tight_layout=True)
    axs[0].hist(PZ0, bins=50,range=(Emin,Emax))
    axs[1].hist(P0,  bins=50,range=(Emin,Emax))

    axs[0].set_xlabel('E [GeV]')
    axs[0].set_ylabel('Particles')
    axs[0].set_title(f'Generated')
    axs[0].grid(True)
    
    axs[1].set_xlabel('E [GeV]')
    axs[1].set_ylabel('Particles')
    axs[1].set_title(f'In Acceptance')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("generator_energy.pdf")
    plt.show()
    
    
    ### save config to pickle
    data = {"initial_states":initial_states, "particle_trajectories":particle_trajectories, "particle_collisions":particle_collisions, "dipole":dipole, "quad0":quad0, "quad1":quad1, "quad2":quad2}
    for i,detector in enumerate(detectors): data.update( {f"ALPIDE_{i}":detector} )
    fpklname = "generator.pkl"
    fpkl = open(fpklname,'wb')
    pickle.dump(data,fpkl,protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    