import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import time
import pickle
import multiprocessing
from matplotlib.animation import FuncAnimation, PillowWriter


import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-mag', metavar='magnets settings (run 502 or run 490)', required=True,  help='magnets settings (run 502 or run 490)')
parser.add_argument('-gen', metavar='particles to generate', required=True,  help='particles to generate')
parser.add_argument('-acc', metavar='require full acceptance?', required=False,  help='require full acceptance?')
parser.add_argument('-mlt', metavar='multi processing?', required=False,  help='multi processing?')
argus = parser.parse_args()
MagnetsSettings = int(argus.mag)
if(MagnetsSettings!=502 and MagnetsSettings!=490):
    print(f"Unsupported magnets settings run: {MagnetsSettings}")
    quit()
Nparticles = int(argus.gen)
if(Nparticles<=0):
    print(f"Unsupported Nparticles: {Nparticles}")
    quit()
fullacc = True if(argus.acc is not None and argus.acc=="1") else False
mltprc  = True if(argus.mlt is not None and argus.mlt=="1") else False




plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'

# Convert units
m_to_cm  = 100
m_to_mm  = 1000
cm_to_mm = 10
cm_to_m  = 0.01
mm_to_m  = 0.001
mm_to_cm = 0.1
kG_to_T  = 0.1
GeV_to_kgms   = 5.39e-19
GeV_to_kg     = 1.8e-27
GeV_to_kgm2s2 = 1.6e-10


# Physical constants
c   = 299792458  # speed of light in m/s
c2  = c*c
e   = 1.602176634e-19  # elementary charge in C
m_e = 9.1093837015e-31  # electron/positron mass in kg
m_p = 1.67262192e-27 # proton/antiproton mass in kg

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

zIP = +30 # cm?
zBe = -84 # cm

# Define detector x range
detector_x_center_cm = -1.0 # cm
# detector_x_center_cm = 0. # cm
detector_x_center_m = detector_x_center_cm*cm_to_m

# Define detector y range
detector_y_center_cm = 5.165 + 0.1525 + 3.685 # cm
detector_y_center_m  = detector_y_center_cm*cm_to_m

# Calculate detector z position
detector_z_base_cm = 1363 + 303.2155 + 11.43 + 1.05  # cm
detector_z_base_m  = detector_z_base_cm*cm_to_m
detector_z_base_mm = detector_z_base_cm*cm_to_mm


# Define magnet elements
class Element:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min * cm_to_m
        self.x_max = x_max * cm_to_m
        self.y_min = y_min * cm_to_m
        self.y_max = y_max * cm_to_m
        self.z_min = z_min * cm_to_m
        self.z_max = z_max * cm_to_m
        
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
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, B_x,B_y,B_z):
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max)
        self.B_x = B_x  # Tesla
        self.B_y = B_y  # Tesla
        self.B_z = B_z  # Tesla
        self.hits = []
        
    def field(self, x, y, z):
        if self.is_inside(x, y, z):
            return np.array([self.B_x, self.B_y, self.B_z])
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
        
        print(f"Detector at z={self.z_pos:.2f} m hits:")
        for hit in self.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            pz  = initial_states[pid][5]/GeV_to_kgms
            print(f"  Particle {pid}: x={xx:.6f} m, y={yy:.6f} m (pz={pz:.2f} GeV)")

class Beampipe(Element):
    def __init__(self, inner_radius_cm, outer_radius_cm, z_min_cm, z_max_cm):
        # For bounding box, use outer radius to define x,y limits
        super().__init__(-outer_radius_cm, outer_radius_cm, 
                         -outer_radius_cm, outer_radius_cm, 
                         z_min_cm, z_max_cm)
        
        self.inner_radius = inner_radius_cm * cm_to_m
        self.outer_radius = outer_radius_cm * cm_to_m
        self.z_min_pipe = z_min_cm * cm_to_m
        self.z_max_pipe = z_max_cm * cm_to_m
        self.hits = []
        
    def field(self, x, y, z):
        # Beampipe doesn't generate magnetic fields
        return np.array([0, 0, 0])
        
    def is_inside_pipe_material(self, x, y, z):
        """Check if point is inside the pipe material (between inner and outer radius)"""
        if not (self.z_min_pipe <= z <= self.z_max_pipe):
            return False
        
        r = np.sqrt(x**2 + y**2)
        return self.inner_radius <= r <= self.outer_radius
    
    def is_outside_vacuum(self, x, y, z):
        """Check if point is outside the vacuum region (beyond inner radius)"""
        if not (self.z_min_pipe <= z <= self.z_max_pipe):
            return False
        
        r = np.sqrt(x**2 + y**2)
        return r > self.inner_radius
        
    def record_hit(self, particle_id, x, y, z, px, py, pz):
        self.hits.append({
            'particle_id': particle_id,
            'x': x, 'y': y, 'z': z,
            'px': px, 'py': py, 'pz': pz
        })
        
    def plot_element(self, ax, color='gray', alpha=0.3):
        """Plot beampipe as a semi-transparent cylinder"""
        # Create cylindrical surface
        z_pipe = np.linspace(self.z_min_pipe, self.z_max_pipe, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        
        # Create meshgrid for outer surface
        Z_outer, THETA_outer = np.meshgrid(z_pipe, theta)
        X_outer = self.outer_radius * np.cos(THETA_outer)
        Y_outer = self.outer_radius * np.sin(THETA_outer)
        
        # Create meshgrid for inner surface  
        X_inner = self.inner_radius * np.cos(THETA_outer)
        Y_inner = self.inner_radius * np.sin(THETA_outer)
        
        # Plot outer surface
        ax.plot_surface(X_outer, Y_outer, Z_outer, 
                       color=color, alpha=alpha, linewidth=0)
        
        # Plot inner surface (slightly more transparent)
        ax.plot_surface(X_inner, Y_inner, Z_outer, 
                       color=color, alpha=alpha*0.5, linewidth=0)
        
        # Add end caps
        r_cap = np.linspace(self.inner_radius, self.outer_radius, 20)
        theta_cap = np.linspace(0, 2*np.pi, 50)
        R_cap, THETA_cap = np.meshgrid(r_cap, theta_cap)
        X_cap = R_cap * np.cos(THETA_cap)
        Y_cap = R_cap * np.sin(THETA_cap)
        
        # Front cap
        Z_cap_front = np.full_like(X_cap, self.z_min_pipe)
        ax.plot_surface(X_cap, Y_cap, Z_cap_front, 
                       color=color, alpha=alpha, linewidth=0)
        
        # Back cap  
        Z_cap_back = np.full_like(X_cap, self.z_max_pipe)
        ax.plot_surface(X_cap, Y_cap, Z_cap_back, 
                       color=color, alpha=alpha, linewidth=0)

########################################################################
########################################################################
########################################################################
########################################################################

# Create the detector objects
detectors = []
for i in range(5):
    detector = Detector(
        x_min=detector_x_center_cm-chipYcm/2., x_max=detector_x_center_cm+chipYcm/2.,
        y_min=detector_y_center_cm-chipXcm/2., y_max=detector_y_center_cm+chipXcm/2.,
        z_pos=detector_z_base_cm + i
    )
    detectors.append(detector)

# Create magnetic elements
quad0 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=367.33336, z_max=464.6664, # cm
    gradient=-7.637 if(MagnetsSettings==502) else -30.68 # kG/m
)
quad1 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=590.3336, z_max=687.6664, # cm
    gradient=+28.55 if(MagnetsSettings==502) else +46.42 # kG/m
)
quad2 = Quadrupole(
    x_min=-2.4610, x_max=2.4610, # cm
    y_min=-2.4610, y_max=2.4610, # cm
    z_min=812.3336, z_max=909.6664, # cm
    gradient=-7.637 if(MagnetsSettings==502) else -30.68 # kG/m
)
xcorr = Dipole(
    x_min=-10.795, x_max=+10.795, 
    y_min=-4.699, y_max=+4.699,
    z_min=987.779, z_max=1011.15,
    B_x=0, B_y=+0.026107, B_z=0  # Tesla
)
dipole = Dipole(
    x_min=-2.2352, x_max=2.2352,
    y_min=-6.3752, y_max=3.1752,
    z_min=1260.34, z_max=1351.78,
    B_x=0.219, B_y=0, B_z=0  # Tesla
)

# Create beampipe from z=0 to entrance of dipole:
beampipe_inner_radius_cm = 2.   # 2 cm inner radius 
beampipe_outer_radius_cm = 2.2  # 2.2 cm outer radius
beampipe_z_start_cm = 0.0       # Start at IP
beampipe_z_end_cm = dipole.z_min * m_to_cm - 0.0  # End 0 cm before dipole entrance
beampipe = Beampipe(
    inner_radius_cm=beampipe_inner_radius_cm,
    outer_radius_cm=beampipe_outer_radius_cm, 
    z_min_cm=beampipe_z_start_cm,
    z_max_cm=beampipe_z_end_cm
)


### collect all elements
elements = [quad0, quad1, quad2, xcorr, dipole, beampipe] + detectors



########################################################################
########################################################################
########################################################################



# To calculate total magnetic field at a point
def total_field(position):
    x, y, z = position
    field = np.zeros(3)
    for element in elements: field += element.field(x, y, z)
    return field



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



# Define the equations of motion for a charged particle in a magnetic field
def particle_motion(t, state):
    # State vector: [x[m],y[m],z[m], px[kg*m/s],py[kg*m/s],pz[kg*m/s], m[kg],q[unit]]
    x,y,z, px,py,pz, m,q = state
    position = np.array([x, y, z])
    momentum = np.array([px, py, pz])
    
    # Relativistic velocity
    p_mag = np.linalg.norm(momentum)
    gamma = np.sqrt(1 + (p_mag/(m*c))**2)  # Relativistic factor
    velocity = momentum / (gamma * m)
    
    # Magnetic field at current position
    B = total_field(position)
    
    # Lorentz force: F = q(v × B)
    force = (q * e) * np.cross(velocity, B)
    
    # Rate of change of position is velocity
    dx_dt = velocity[0]
    dy_dt = velocity[1]
    dz_dt = velocity[2]
    
    # Rate of change of momentum is force
    dpx_dt = force[0]
    dpy_dt = force[1]
    dpz_dt = force[2]
    
    return [dx_dt,dy_dt,dz_dt, dpx_dt,dpy_dt,dpz_dt, m,q]




# Add this function to replace your existing collision_event function
def collision_event(t, state, elements_to_check=None, beampipe=None):
    '''
    Event function to detect when a particle hits walls of magnet elements or beampipe.
    Returns zero when a collision occurs.
    
    Parameters:
    -----------
    t : float
        Current time value
    state : array 
        Current state [x,y,z, px,py,pz, m,q]
    elements_to_check : list, optional
        List of elements to check for collisions
    beampipe : Beampipe, optional
        Beampipe object to check for collisions
        
    Returns:
    --------
    float
        Distance from nearest wall (negative inside boundaries, zero at boundary, positive outside)
    '''
    if elements_to_check is None:
        elements_to_check = [quad0, quad1, quad2, xcorr, dipole] 
    
    x, y, z = state[0], state[1], state[2]
    
    # Initialize distance to a large value
    min_distance = float('inf')
    
    # Check beampipe collision first (most restrictive)
    if beampipe is not None and beampipe.z_min_pipe <= z <= beampipe.z_max_pipe:
        r = np.sqrt(x**2 + y**2)
        # Distance to inner wall of beampipe (negative if outside vacuum)
        distance_to_vacuum_boundary = beampipe.inner_radius - r
        min_distance = min(min_distance, distance_to_vacuum_boundary)
    
    # Check magnet element collisions
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

# Set terminal attribute
collision_event.terminal = True





def propagate_particle_with_collision(particle_id, initial_state, t_span, beampipe=None, max_step=1e-9):
    '''
    Propagate particle through beamline with collision detection including beampipe.

    Parameters:
    -----------
    particle_id : int, Unique identifier for the particle
    initial_state : array, Initial state [x0,y0,z0, px0,py0,pz0, m,q]
    t_span : tuple, (t_start, t_end) for integration
    beampipe : Beampipe, optional, Beampipe object for collision detection
    max_step : float, optional, Maximum step size for integrator

    Returns:
    --------
    solution : OdeSolution
        Solution object from solve_ivp
    collision_info : dict or None
        Information about collision if it occurred, None otherwise
    '''
    # Use solve_ivp with event detection including beampipe
    solution = solve_ivp(
        particle_motion,
        t_span,
        initial_state,
        method='RK45',
        events=lambda t, state: collision_event(t, state, beampipe=beampipe),
        max_step=max_step,
        rtol=1e-8,
        atol=1e-10
    )

    # Check for detector crossings
    for det in detectors: record_hits(det, det.z_pos, solution, particle_id)
    # Record hits on other elements
    record_hits(dipole, dipole.z_max, solution, particle_id)
    record_hits(xcorr,  xcorr.z_max,  solution, particle_id)
    record_hits(quad0,  quad0.z_max,  solution, particle_id)
    record_hits(quad1,  quad1.z_max,  solution, particle_id)
    record_hits(quad2,  quad2.z_max,  solution, particle_id)

    # Record beampipe hits if it exists
    if beampipe is not None: record_hits(beampipe, beampipe.z_max_pipe, solution, particle_id)

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
        element_name = "unknown"

        # Check if beampipe was hit first
        if (beampipe is not None and
            beampipe.z_min_pipe <= collision_state[2] <= beampipe.z_max_pipe):
            r = np.sqrt(collision_state[0]**2 + collision_state[1]**2)
            if abs(r - beampipe.inner_radius) < 1e-6:
                collision_element = beampipe
                element_name = "Beampipe"

        # If not beampipe, check other elements
        if collision_element is None:
            for element in [quad0, quad1, quad2, xcorr, dipole]:
                if (element.z_min <= collision_state[2] <= element.z_max):
                    if (abs(collision_state[0] - element.x_min) < 1e-6 or
                        abs(collision_state[0] - element.x_max) < 1e-6 or
                        abs(collision_state[1] - element.y_min) < 1e-6 or
                        abs(collision_state[1] - element.y_max) < 1e-6):
                        collision_element = element
                        break

            if   collision_element == quad0:  element_name = "Quadrupole 0"
            elif collision_element == quad1:  element_name = "Quadrupole 1"
            elif collision_element == quad2:  element_name = "Quadrupole 2"
            elif collision_element == xcorr:  element_name = "Xcorr"
            elif collision_element == dipole: element_name = "Dipole"

        collision_info = {
            "time": collision_time,
            "position": collision_state[:3],
            "momentum": collision_state[3:],
            "element": element_name
        }

    return solution, collision_info


def get_xy_at_z(solution, collision, z_target):
    ### check if there is NO collision first:
    if collision is not None:
        return None,None
    ### get the trajectory coordinates
    z = solution.y[2]
    x = solution.y[0]
    y = solution.y[1]
    
    ### Loop over each segment and look for a crossing
    for i in range(len(z) - 1):
        z1, z2 = z[i], z[i+1]
        if (z1 <= z_target <= z2) or (z2 <= z_target <= z1):
            # avoid division by zero
            if(z2 == z1): frac = 0.0
            else:         frac = (z_target - z1) / (z2 - z1)
            x_interp = x[i] + frac * (x[i+1] - x[i])
            y_interp = y[i] + frac * (y[i+1] - y[i])
            return x_interp, y_interp

    raise ValueError(f"Trajectory does not cross z = {z_target:.3f} m.")


### TODO this function is needed for multiprocess runs ONLY
def refill_detector_hits(initial_states, trajectories):
    if(len(initial_states)!=len(trajectories)):
        raise ValueError(f"number of initial_states {len(initial_states)} is not the same as number of trajectories {len(trajectories)}")
    ### remove old hits
    for element in elements:
        element.hits = []
    ### refill the hits
    for pid,trajectory in enumerate(trajectories):
        for element in elements:
            z = element.z_pos if(element in detectors) else element.z_max
            record_hits(element,z,trajectory,pid)
    
        

def plot_system(particle_trajectories, particle_collisions, initial_states, pdfname, nmaxtrks=100, ax=None):
    
    isAxNone = (ax is None)
    
    # fig = plt.figure(figsize=(16, 10))
    # ax = fig.add_subplot(111, projection='3d')
    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Plot magnetic elements
    quad0.plot_element(ax,  'cyan')
    quad1.plot_element(ax,  'purple')
    quad2.plot_element(ax,  'cyan')
    xcorr.plot_element(ax,  'orange')
    dipole.plot_element(ax, 'red')
    beampipe.plot_element(ax, 'gray', alpha=0.3)

    # Plot detector planes
    detector_colors = ['green', 'green', 'green', 'green', 'green']
    for i, detector in enumerate(detectors):
        detector.plot_element(ax, detector_colors[i % len(detector_colors)])

    # Plot particle trajectories with collision handling
    collision_count = 0
    for i, (trajectory, collision) in enumerate(zip(particle_trajectories, particle_collisions)):
        if i >= nmaxtrks:
            break
            
        x, y, z = trajectory.y[0], trajectory.y[1], trajectory.y[2]
        
        if collision is not None:
            # Find the index where collision occurred
            collision_z = collision['position'][2]
            collision_idx = None
            
            # Find the closest point to collision in trajectory
            for j, z_val in enumerate(z):
                if z_val >= collision_z:
                    collision_idx = j
                    break
            
            if collision_idx is not None:
                # Plot only up to collision point
                ax.plot(x[:collision_idx+1], y[:collision_idx+1], z[:collision_idx+1], 'r-', linewidth=1.5, alpha=0.8)  # Red for collided particles
                # Mark collision point?
                # ax.scatter(collision['position'][0], collision['position'][1], collision['position'][2], c='red', s=20, marker='x')
                collision_count += 1
            else:
                # Fallback: plot entire trajectory in red
                ax.plot(x, y, z, 'r-', linewidth=1.0, alpha=0.6)
                collision_count += 1
        else:
            # No collision: plot entire trajectory in blue
            ax.plot(x, y, z, 'b-', linewidth=1.0, alpha=0.8)

    if(isAxNone): print(f"Plotted {collision_count} collided trajectories out of {min(nmaxtrks, len(particle_trajectories))}")

    # Set axis labels and limits
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 18)

    # Add legend for elements
    beampipe_patch  = mpatches.Patch(color='gray',   alpha=0.3, label='Beampipe')
    quad0_patch     = mpatches.Patch(color='cyan',   alpha=0.5, label='Quad0')
    quad1_patch     = mpatches.Patch(color='purple', alpha=0.5, label='Quad1')
    quad2_patch     = mpatches.Patch(color='cyan',   alpha=0.5, label='Quad2')
    xcorr_patch     = mpatches.Patch(color='orange', alpha=0.5, label='XCorr')
    dipole_patch    = mpatches.Patch(color='red',    alpha=0.5, label='Dipole')
    collision_patch = mpatches.Patch(color='red',    alpha=0.8, label='Collided particles')
    free_patch      = mpatches.Patch(color='blue',   alpha=0.8, label='Free particles')
    detector_patch  = mpatches.Patch(color='green',  alpha=0.5, label='Detector')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([beampipe_patch, quad0_patch, quad1_patch, quad2_patch, xcorr_patch, dipole_patch, collision_patch, free_patch, detector_patch])
    ax.legend(handles=handles, loc='center left')
    ax.set_title(f'Particle Propagation (MagSet={MagnetsSettings})')

    if(isAxNone):
        plt.tight_layout()
        plt.savefig(f"{pdfname}_tracks.pdf")
        plt.show()
        return fig


def plot_gif(pdfname):    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(angle):
        ax.cla()  # Clear the 3D axis for redrawing
        plot_system(
            particle_trajectories,
            particle_collisions,
            initial_states,
            pdfname=None,
            nmaxtrks=100,
            ax=ax
        )
        ax.view_init(elev=30, azim=angle)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, 360, 5),  # Rotate full circle in 5° steps
        interval=100  # milliseconds per frame
    )

    # Save as animated GIF using Pillow
    ani.save(f"{pdfname}_tracks_rotation.gif", writer=PillowWriter(fps=10))


def plot_scatter(pdfname):
    fig, axs = plt.subplots(1, 5, figsize=(10, 4), sharex=True, sharey=True, tight_layout=True)
    for i, detector in enumerate(detectors):
        rect = plt.Rectangle(
            (detector.x_min, detector.y_min),
            detector.x_max - detector.x_min,
            detector.y_max - detector.y_min,
            fill=False, edgecolor='gray'
        )
        axs[i].add_patch(rect)

        # Plot hits
        hit_x = [hit['x'] for hit in detector.hits]
        hit_y = [hit['y'] for hit in detector.hits]
        particle_ids = [hit['particle_id'] for hit in detector.hits]

        for j, (x, y, pid) in enumerate(zip(hit_x, hit_y, particle_ids)): axs[i].plot(x, y, 'o', markersize=1)

        axs[i].set_xlim(detector.x_min * 1.2, detector.x_max * 1.2)
        axs[i].set_ylim(detector.y_min * 0.9, detector.y_max * 1.1)
        axs[i].set_xlabel('X [m]')
        axs[i].set_ylabel('Y [m]')
        axs[i].set_title(f'ALPIDE_{i}')
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"{pdfname}_scatter.pdf")
    plt.show()

    return fig


def truncated_exp_NK(a,b,how_many):
    a = -np.log(a)
    b = -np.log(b)
    rands = np.exp(-(np.random.rand(how_many)*(b-a) + a))
    return rands[0] if(how_many==1) else rands


def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)




#####################
### main function ###
#####################
if __name__ == "__main__":
    
    print(f"Run multiprocessing: {mltprc}")
    
    print(f"MagnetsSettings: {MagnetsSettings}")
    print(f"quad0: {quad0.gradient} T/m")
    print(f"quad1: {quad1.gradient} T/m")
    print(f"quad2: {quad2.gradient} T/m")
    
    pdfname = f"generator_{MagnetsSettings}"
    
    # Define example initial conditions for particles
    # Format: [x0,y0,z0, px0,py0,pz0, m,q]
    # Momentum in units of GeV/c and convert to kg*m/s
    Emin = 0.5 ## GeV 
    Emax = 5.0 ## GeV
    
    # sigmax = 0.0001 ## m (100 um)
    # sigmay = 0.0001 ## m (100 um)
    # sigmaz = 0.0005 ## m (500 um)
    # sigmaPx = 0.00030 ## GeV
    # sigmaPy = 0.00005 ## GeV
    
    sigmax = 0.00005 ## m (50 um)
    sigmay = 0.00005 ## m (50 um)
    sigmaz = 0.00005 ## m (50 um)
    sigmaPx = 0.0008 ## GeV
    sigmaPy = 0.0004 ## GeV
    '''
    NBW from Arka (PTARMIGAN)
    sigma_x: 0.004166 mm
    sigma_y: 0.0006886 mm
    sigma_z: 7.205e-3 mm
    sigma_px: 0.001037 GeV
    sigma_py: 0.000193 GeV
    '''

    ### origin of particles
    z0 = zBe if(MagnetsSettings==502) else zIP

    ### particle species
    MM = m_e ## kg, positron
    QQ = +1  ## unit charge, positron
    mGeV = (MM*c2)/GeV_to_kgm2s2 ## GeV
    
    initial_states = []
    PZ0 = []
    for i in range(Nparticles):
        XX = np.random.normal(0.0,sigmax)
        YY = np.random.normal(0.0,sigmay)
        ZZ = np.random.normal(z0*cm_to_m,sigmaz)
        PX = np.random.normal(XX,sigmaPx) # GeV
        PY = np.random.normal(YY,sigmaPy) # GeV
        EE = truncated_exp_NK(Emin,Emax,1) # GeV
        PZ = np.sqrt( EE**2 - mGeV**2 - PX**2 - PY**2 ) # GeV
        PZ0.append(PZ) # GeV
        ### convert to proper mks units
        PX *= GeV_to_kgms # kg*m/s
        PY *= GeV_to_kgms # kg*m/s
        PZ *= GeV_to_kgms # kg*m/s
        state = [XX,YY,ZZ, PX,PY,PZ, MM,QQ]
        initial_states.append(state)
    
    
    # Time range for propagation (seconds)
    # For relativistic particles going ~c, need to consider the longest path to the detectors
    # Last detector is at ~18 meters
    tmax = 18 / (0.99 * c)  # Approximate time to travel 18 meters
    t_span = (0, tmax)
    
    
    # Propagate particles and check for collisions (including beampipe)
    particle_trajectories = []
    particle_collisions   = []


    #########################
    ### Run the propagation
    #########################
    if(not mltprc):
        ##########################
        ### no multiprocessing ###
        ##########################
        for i,state in enumerate(initial_states):
            solution, collision = propagate_particle_with_collision(i, state, t_span, beampipe=beampipe)
            particle_trajectories.append(solution)
            particle_collisions.append(collision)
            if(i%100==0): print(f"done propagating particle {i}")
    else:
        ############################
        ### with multiprocessing ###
        ############################
        num_cores = 10 # Number of cores to activate
        pool = multiprocessing.Pool(processes=num_cores) # Create the pool

        # This list will hold the AsyncResult objects. Each AsyncResult corresponds to one submitted task
        async_results = []
        ### loop on the initial states
        for i,state in enumerate(initial_states):
            result = pool.apply_async(propagate_particle_with_collision, args=(i, state, t_span, beampipe), error_callback=collect_errors)
            async_results.append(result)
            if(i%100==0): print(f"done propagating particle {i}")
        # After submitting all tasks:
        pool.close() # No more tasks can be added to the pool
        
        # Collect results from each submitted task. The loop here is over the 'async_results' list, which has N elements
        for i,async_res in enumerate(async_results):
            # .get() blocks until the task associated with this async_res is complete. It retrieves the tuple (obj1_result, obj2_result) returned by my_function_per_task.
            solution, collision = async_res.get()
            # APPEND the individual results for THIS task to the main lists. This doesn't overwrite; it adds a new element to the end of the list.
            particle_trajectories.append(solution)
            particle_collisions.append(collision)
        # Wait for all worker processes to finish and terminate
        pool.join()
        
        ### re-record the hits
        print(f"starting refill:")
        st = time.time()
        refill_detector_hits(initial_states, particle_trajectories)
        et = time.time()
        elapsed_time = et - st
        print('end time:', elapsed_time, 'seconds')
    ##########################
    
    
    
    ############################
    ### Now do some plotting ###
    ############################
    
    
    
    ### Plot the system with particle trajectories
    main_fig = plot_system(particle_trajectories, particle_collisions, initial_states, pdfname, nmaxtrks=100)
    
    ### make the gif:
    plot_gif(pdfname)

    ### Plot the hits as scatter plot (using detector.hits)
    scat_fig = plot_scatter(pdfname)


    ### check acceptance
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


    ### plot the hits COARSELY:
    fig, axs = plt.subplots(1, 5, figsize=(10, 3.5), sharex=True, sharey=True, tight_layout=True)
    P0 = []
    hOcc = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        P = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            pz  = initial_states[pid][5]
            # X.append((xx-detector_x_center_m)*m_to_mm) ## TODO: this is important if we want to plot the hits where the (x,y)=(0,0) point is in the center of the chip
            X.append(xx*m_to_mm)
            # Y.append((yy-detector_y_center_m)*m_to_mm) ## TODO: this is important if we want to plot the hits where the (x,y)=(0,0) point is in the center of the chip
            Y.append(yy*m_to_mm) 
            if(fullacc and pid not in list_good_tracks): continue
            P.append(pz/GeV_to_kgms)
        # hOcc.append( axs[i].hist2d(X, Y, bins=(200,100),range=[[-chipYmm/2,+chipYmm/2],[-chipXmm/2,+chipXmm/2]], rasterized=True) )
        hOcc.append( axs[i].hist2d(X, Y, bins=(200,100),range=[[detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm],[detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm]], rasterized=True) )
        if(i==0): P0 = P
        axs[i].set_xlabel('X [mm]')
        axs[i].set_ylabel('Y [mm]')
        axs[i].set_title(f'ALPIDE_{i}')
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_coarse.pdf")
    plt.show()


    ### plot the hits FINELY:
    fig, axs = plt.subplots(1, 5, figsize=(10, 3.5), sharex=True, sharey=True, tight_layout=True)
    P0 = []
    hOcc = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        P = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            pz  = initial_states[pid][5]
            # X.append((xx-detector_x_center_m)*m_to_mm) ## TODO: this is important if we want to plot the hits where the (x,y)=(0,0) point is in the center of the chip
            X.append(xx*m_to_mm)
            # Y.append((yy-detector_y_center_m)*m_to_mm) ## TODO: this is important if we want to plot the hits where the (x,y)=(0,0) point is in the center of the chip
            Y.append(yy*m_to_mm)
            if(fullacc and pid not in list_good_tracks): continue
            P.append(pz/GeV_to_kgms)
        # hOcc.append( axs[i].hist2d(X, Y, bins=(npix_y+1,npix_x+1),range=[[-chipYmm/2,+chipYmm/2],[-chipXmm/2,+chipXmm/2]], rasterized=True) )
        hOcc.append( axs[i].hist2d(X, Y, bins=(npix_y+1,npix_x+1),range=[[detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm],[detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm]], rasterized=True) )
        if(i==0): P0 = P
        axs[i].set_xlabel('X [mm]')
        axs[i].set_ylabel('Y [mm]')
        axs[i].set_title(f'ALPIDE_{i}')
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_fine.pdf")
    plt.show()
    
    
    ### plot the exit plane at the dipole:
    rectD1 = plt.Rectangle(
        (dipole.x_min, dipole.y_min),
        dipole.x_max - dipole.x_min,
        dipole.y_max - dipole.y_min,
        fill=False, edgecolor='blue'
    )
    rectD2 = plt.Rectangle(
        (dipole.x_min, dipole.y_min),
        dipole.x_max - dipole.x_min,
        dipole.y_max - dipole.y_min,
        fill=False, edgecolor='blue'
    )
    
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        X.append(point['x'])
        Y.append(point['y'])
    hDzoom = ax.hist2d(X, Y, bins=(200,200),range=[[dipole.x_min*1.2,dipole.x_max*1.2],[dipole.y_min*1.1,dipole.y_max*1.1]], rasterized=True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Dipole exit')
    ax.add_patch(rectD1)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
    plt.tight_layout()
    plt.savefig(f"{pdfname}_dipole_exit_zoom.pdf")
    plt.show()
    
    
    ### plot the exit plane at the dipole:
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        X.append(point['x'])
        Y.append(point['y'])
    hD = ax.hist2d(X, Y, bins=(120,120),range=[[-80*mm_to_m,+80*mm_to_m],[-70*mm_to_m,+90*mm_to_m]], rasterized=True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Dipole exit')
    ax.add_patch(rectD2)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
    plt.tight_layout()
    plt.savefig(f"{pdfname}_dipole_exit.pdf")
    plt.show()
    
    
    ### plot the quads exits
    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True, tight_layout=True)
    X0 = []
    Y0 = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        X0.append(point['x'])
        Y0.append(point['y'])
    hq0 = axs[0].hist2d(X0, Y0, bins=(150,150),range=[[quad0.x_min*1.5,quad0.x_max*1.5],[quad0.y_min*1.5,quad0.y_max*1.5]], rasterized=True)
    axs[0].set_xlim(quad0.x_min * 1.5, quad0.x_max * 1.5)
    axs[0].set_ylim(quad0.y_min * 1.5, quad0.y_max * 1.5)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')
    axs[0].set_title(f'Quad0 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[0].grid(True,linewidth=0.25,alpha=0.25)
    X1 = []
    Y1 = []
    for point in quad1.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        X1.append(point['x'])
        Y1.append(point['y'])
    hq1 = axs[1].hist2d(X1, Y1, bins=(150,150),range=[[quad1.x_min*1.5,quad1.x_max*1.5],[quad1.y_min*1.5,quad1.y_max*1.5]], rasterized=True)
    axs[1].set_xlim(quad1.x_min * 1.5, quad1.x_max * 1.5)
    axs[1].set_ylim(quad1.y_min * 1.5, quad1.y_max * 1.5)
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Y [m]')
    axs[1].set_title(f'Quad1 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1].grid(True,linewidth=0.25,alpha=0.25)
    X2 = []
    Y2 = []
    for point in quad2.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        X2.append(point['x'])
        Y2.append(point['y'])
    hq2 = axs[2].hist2d(X2, Y2, bins=(150,150),range=[[quad2.x_min*1.5,quad2.x_max*1.5],[quad2.y_min*1.5,quad2.y_max*1.5]], rasterized=True)
    axs[2].set_xlim(quad2.x_min * 1.5, quad2.x_max * 1.5)
    axs[2].set_ylim(quad2.y_min * 1.5, quad2.y_max * 1.5)
    axs[2].set_xlabel('X [m]')
    axs[2].set_ylabel('Y [m]')
    axs[2].set_title(f'Quad2 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[2].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[2].grid(True,linewidth=0.25,alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{pdfname}_quads_exit.pdf")
    plt.show()
    
    
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    XX = []
    PX = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        XX.append(point['x'])
        px = initial_states[pid][3]/GeV_to_kgms
        PX.append( px )   
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), range=[[-5e-3,+5e-3],[-5e-4,+5e-4]], rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)
    YY = []
    PY = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        YY.append(point['y'])
        py = initial_states[pid][4]/GeV_to_kgms
        PY.append( py )
    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), range=[[-5e-3,+5e-3],[-5e-4,+5e-4]], rasterized=True)
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{pdfname}_divergence.pdf")
    plt.show()
    
    
    ### plot the energies
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    hpz0 = axs[0].hist(PZ0, bins=50,range=(Emin,Emax), rasterized=True)
    hp0  = axs[1].hist(P0,  bins=50,range=(1.5,4.5), rasterized=True)

    axs[0].set_xlim(Emin,Emax)
    axs[0].set_xlabel('E [GeV]')
    axs[0].set_ylabel('Particles')
    axs[0].set_title(f'Generated')
    plt.locator_params(axis='x', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[1].set_xlim(1.5,4.5)
    axs[1].set_yscale("log")
    axs[1].set_xlabel('E [GeV]')
    axs[1].set_ylabel('Particles')
    axs[1].set_title(f'In Acceptance')
    plt.locator_params(axis='x', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f"{pdfname}_energy.pdf")
    plt.show()
    
    
    ### example how to get the x,y at any point along the trajectory
    # for t,trajectory in enumerate(particle_trajectories):
    #     collision = particle_collisions[t]
    #     z = detectors[0].z_pos
    #     x, y = get_xy_at_z(trajectory,collision,z)
    #     if(x is not None and y is not None):
    #         print(f"At z = {z:.4e} m: x={x:.4e} m, y={y:.4e} m")
    
    
    ### save config to pickle
    data = {
        "initial_states": initial_states, 
        "particle_trajectories": particle_trajectories, 
        "particle_collisions": particle_collisions, 
        "dipole": dipole, 
        "quad0": quad0, 
        "quad1": quad1, 
        "quad2": quad2,
        "beampipe": beampipe
    }
    for i,detector in enumerate(detectors): data.update( {f"ALPIDE_{i}":detector} )
    fpklname = "generator.pkl"
    fpkl = open(fpklname,'wb')
    pickle.dump(data,fpkl,protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    