"""
Helper functions for rosette code
Author: Joseph
Created: 3/23/23
"""
import numpy as np
from math import pow, radians, sin, cos

def norm_rows(v):
    """Normalize rows of array v into unit vectors."""
    if np.all(v==0):
        v_unit = np.array([1,0,0])
    else:
        if v.ndim == 1:
            v_norm = np.linalg.norm(v)
            v_unit = v/v_norm
        
        else:
            v_norm = np.linalg.norm(v, axis=1)
            v_unit = v/v_norm[:,None]
    return v_unit

def random_spherical_cap(cone_angle_deg, cone_direction, num_points):
    """
    Generates a desired number of random points on a spherical cap, 
    given a solid angle and cone direction.

    Parameters
    ----------
    cone_angle_deg : float
        Solid angle of the cone used to define the spherical cap, in units of degrees.
    cone_direction : list
        Direction of cone as a list of vector components [x, y, z].
    num_points : int
        Number of points to generate on spherical cap. 
    
    Returns
    ----------
    points_rot 
        List of random points on spherical cap, as numpy array. 
    """
    # generate points on spherical cap centered at north pole
    cone_angle_rad = cone_angle_deg*(np.pi/180)
    z = np.random.uniform(np.cos(cone_angle_rad), 1, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    x = np.sqrt(1-z**2)*np.cos(phi)
    y = np.sqrt(1-z**2)*np.sin(phi)
    points = np.column_stack((x, y, z))

    # rotate points
    north_vector = np.array([0, 0, 1])
    cone_direction_norm = norm_rows(cone_direction)
    u = norm_rows(np.cross(north_vector, cone_direction_norm)) # rotation axis
    rot = np.arccos(np.dot(cone_direction_norm, north_vector)) # rotation angle in radians
    ux = u[0]
    uy = u[1]
    uz = u[2]
    # define rotation matrix
    r11 = np.cos(rot) + (ux**2)*(1 - np.cos(rot))
    r12 = ux*uy*(1 - np.cos(rot)) - uz*np.sin(rot)
    r13 = ux*uz*(1 - np.cos(rot)) + uy*np.sin(rot)
    r21 = uy*ux*(1 - np.cos(rot)) + uz*np.sin(rot)
    r22 = np.cos(rot) + (uy**2)*(1 - np.cos(rot))
    r23 = uy*uz*(1 - np.cos(rot)) - ux*np.sin(rot)
    r31 = uz*ux*(1 - np.cos(rot)) - uy*np.sin(rot)
    r32 = uz*uy*(1 - np.cos(rot)) + ux*np.sin(rot)
    r33 = np.cos(rot) + (uz**2)*(1 - np.cos(rot))
    rot_mat = np.array([[r11, r12, r13], 
                        [r21, r22, r23], 
                        [r31, r32, r33]])
    
    points_rot = np.matmul(rot_mat, points.T)
    points_rot = points_rot.T   

    return points_rot

def rotate_axis_angle(axis, angle):
    """
    Create a transformation matrix with given axis and angle of rotation

    Parameters
    ----------
    axis : np.array
        axis to rotate about
    angle : float
        radians to rotate  

    Returns
    ----------
    transform_mat : np.array
        Transformation matrix
    """
    u = axis / np.linalg.norm(axis)
    ux = u[0]
    uy = u[1]
    uz = u[2]
    x11 = cos(angle) + pow(ux, 2) * (1 - cos(angle))
    x12 = ux * uy * (1 - cos(angle)) - uz * sin(angle)
    x13 = ux * uz * (1-cos(angle)) + uy * sin(angle)

    x21 = uy * ux *(1-cos(angle)) + uz * sin(angle)
    x22 = cos(angle) + pow(uy, 2) * (1-cos(angle))
    x23 = uy * uz * (1-cos(angle)) - ux *sin(angle)

    x31 = uz * ux * (1-cos(angle)) - uy * sin(angle)
    x32 = uz * uy * (1-cos(angle)) + ux * sin(angle)
    x33 = cos(angle) + pow(uz, 2) * (1-cos(angle))
    transform_mat = np.array([
        [x11, x12, x13, 0],
        [x21, x22, x23, 0],
        [x31, x32, x33, 0],
        [0, 0, 0, 1]
    ])
    return transform_mat