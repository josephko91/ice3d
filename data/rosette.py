import pyvista as pv
import numpy as np
from math import pi
import trame
import math
from data import helper
from copy import deepcopy
import miniball
import pymeshfix as mf
from decimal import *

class Rosette:
    """
    Class representing bullet rosette ice crystals
    """
    def __init__(self, a, c, r0, h0, hp, n_arms, hollow=False, hh=0, hh_sigma=0.5, hollow_rand=False):
        # geoemetric parameters
        self.a = a # half max length across basal face
        self.c = c # half max length across prism face
        self.r0 = r0 # radius of center sphere
        self.h0 = h0 # penetration depth of bullets
        self.hp = hp # heights of pyramid of bullets
        self.n_arms = n_arms # number of bullet arms
        self.hollow = hollow # hollowing turned on or off?
        self.hh = hh # height of hollowing cone
        self.hh_sigma = hh_sigma # std dev of hh as a percentage
        self.hollow_rand = hollow_rand# randomize hollowing or not

        # create sphere 
        sphere = pv.Sphere(radius=r0, center=(0, 0, 0), direction=(0, 0, 1), 
                        theta_resolution=30, phi_resolution=30, start_theta=0, 
                        end_theta=360, start_phi=0, end_phi=180)
        self.sphere = sphere.triangulate()

        # create outer shell to "place" bullets on
        r_outer = hp/2 + c - h0 + r0
        if n_arms == 2: # line
            outer_shell = pv.Line(pointa=(-r_outer, 0.0, 0.0), 
                                pointb=(r_outer, 0.0, 0.0), resolution=1)
            outer_coords = outer_shell.points
        elif n_arms == 4: # tetrahedron
            outer_shell = pv.Tetrahedron(radius=r_outer, center=(0.0, 0.0, 0.0))
            outer_coords = outer_shell.points
        elif n_arms == 6: # octahedron
            outer_shell = pv.Octahedron(radius=r_outer, center=(0.0, 0.0, 0.0))
            outer_coords = outer_shell.points
        elif n_arms == 8: # cube
            # Note: this may not be the optimal solution for n=8, check later
            l  = (2*r_outer)/(3**(1/2))
            outer_shell = pv.Cube(center=(0.0, 0.0, 0.0), x_length=l, 
                                y_length=l, z_length=l)
            outer_coords = outer_shell.points
        else: 
            # Modified fibbonaci lattice 
            # Source: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
            epsilon = 0.33
            goldenRatio = (1 + 5**0.5)/2
            i = np.arange(0, n_arms) 
            theta = 2 *pi * i / goldenRatio
            phi = np.arccos(1 - 2*(i+epsilon)/(n_arms-1+2*epsilon))
            x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
            outer_coords = r_outer*(np.column_stack((x, y, z)))

        # create bullet arm
        cyl = pv.Cylinder(center=(0.0, 0.0, c+hp), direction=(0.0, 0.0, -1.0), 
                        radius=a, height=2*c, resolution=6, capping=True)
        pyr = pv.Cone(center=(0.0, 0.0, hp/2), direction=(0.0, 0.0, -1.0), 
                    height=hp, radius=a, capping=True, angle=None, resolution=6)
        cyl = cyl.triangulate()
        pyr = pyr.triangulate()
        bullet = cyl.boolean_union(pyr).triangulate()
        self.original_bullet = bullet

        # copy, translate, and rotate bullets
        self.bullets = {} # save bullets in nested dictionary
        for i in range(len(outer_coords)):
            pt = outer_coords[i]
            # translate
            translate_vector = pt - bullet.center
            if hollow == True: 
                bullet = self.make_hollow(hh, hh_sigma, hollow_rand) # hollow out arm
            bullet_translated = bullet.translate(translate_vector, inplace=False)
            # rotate 
            if pt[2]/r_outer < -1:
                theta = math.degrees(math.acos(-1))
            elif pt[2]/r_outer > 1:
                theta = math.degrees(math.acos(1))
            else:
                theta = math.degrees(math.acos(pt[2]/r_outer))

            if (pt[0]==0 and pt[1]==0):
                bullet_final = bullet_translated.rotate_vector((0, pt[2], -pt[1]), -theta, point=bullet_translated.center)
            else:
                bullet_final = bullet_translated.rotate_vector((pt[1], -pt[0], 0), -theta, point=bullet_translated.center)
            
            # add bullet attributes and mesh to dictionary
            bullet_entry = {}
            bullet_entry['mesh'] = bullet_final.triangulate()
            bullet_entry['xy_scale_factor'] = 1.0
            bullet_entry['z_scale_factor'] = 1.0
            bullet_entry['anchor_point'] = pt
            self.bullets[i] = bullet_entry

    def unify_mesh(self): 
        """
        Create single mesh using boolean union operation
        """
        rosette = self.sphere
        for i in range(self.n_arms):
            bullet = self.bullets[i]
            bullet_mesh = bullet['mesh'].triangulate()
            rosette = rosette.boolean_union(bullet_mesh).triangulate()

            # test feature: added 20230605
            meshfix = mf.MeshFix(rosette)
            meshfix.repair(verbose=False)
            rosette = meshfix.mesh

        self.volume = rosette.volume
        self.model = rosette # final 3d mesh model 

    def copy(self):
        """
        create a new instance
        with the same data as this instance
        """
        # ros_copy = Rosette(self.a, self.c, self.r0, self.h0, self.hp, self.n_arms)
        ros_copy = deepcopy(self)
        return ros_copy

    def plot(self, bg_color='black', obj_color='white', op=0.9, res=720):
        """
        Interactive PyVista visualization
        """
        pl = pv.Plotter(off_screen=True, window_size=[res, res])
        pl.background_color = bg_color

        if hasattr(self, 'model'):
            pl.add_mesh(self.model, show_edges=None, color = obj_color, opacity=op)
        else: 
            pl.add_mesh(self.sphere, show_edges=None, color = obj_color, opacity=op)
            for i in range(self.n_arms):
                bullet = self.bullets[i]['mesh']
                pl.add_mesh(bullet, show_edges=None, color = obj_color, opacity=op)
        return pl

    def random_rotate(self):
        """
        Rotate rosette in a random orientation
        TODO:
        - fix bug related to the reliance on model attribute
        """
        rotated = self.copy()
        # rotated = self
        deg_x = np.random.randint(1, 360)
        deg_y = np.random.randint(1, 360)
        deg_z = np.random.randint(1, 360)
        rotated_model = self.model.rotate_x(deg_x, inplace=False)
        rotated_model.rotate_y(deg_y, inplace=True)
        rotated_model.rotate_z(deg_z, inplace=True)
        rotated.model = rotated_model
        return rotated
    
    def randomize_bullets(self, scaling=False, location=False, inplace=True):
        """
        Randomly perturb the scaling and location of bullets
        """
        # np.random.seed(0) # TEMPORARY
        getcontext().prec = 4
        if inplace:
            rosette = self
        else:
            rosette = self.copy()

        # Perturb bullet scaling
        if scaling == False:
            pass
        else:
            r_outer = rosette.hp/2 + rosette.c - rosette.h0 + rosette.r0
            r_outer_round = Decimal(r_outer)
            bullet_length = 2*rosette.c + rosette.hp
            for i in range(rosette.n_arms):
                sf_basal = np.random.uniform(scaling[0], scaling[1]) # basal scaling factor
                sf_prism = np.random.uniform(scaling[2], scaling[3]) # prism scaling factor
                bullet = rosette.bullets[i]
                pt = bullet['anchor_point']
                pt_2 = Decimal(pt[2].item())
                # theta = math.degrees(math.acos(pt[2]/r_outer))
                theta = math.degrees(math.acos(pt_2/r_outer_round))
                # rotate parallel to north unit vector
                if (pt[0]==0 and pt[1]==0):
                    bullet['mesh'].rotate_vector((0, pt[2], -pt[1]), theta, point=[0,0,0], inplace=True)
                else:
                    bullet['mesh'].rotate_vector((pt[1], -pt[0], 0), theta, point=[0,0,0], inplace=True)
                # scale 
                t_mat = np.array([[sf_basal, 0, 0, 0],
                                  [0,sf_basal, 0, 0],
                                  [0, 0, sf_prism, 0],
                                  [0, 0, 0, 1]])
                bullet['mesh'].transform(t_mat, inplace=True)
                # translate (up or down) to compensate for change in size
                # delta_z = (bullet_length*(1-sf_prism))/2
                delta_z = -(sf_prism-1)*(rosette.r0-rosette.h0)
                bullet['mesh'].translate((0,0,delta_z), inplace=True)

                # rotate back to place
                if (pt[0]==0 and pt[1]==0):
                    bullet['mesh'].rotate_vector((0, pt[2], -pt[1]), -theta, point=[0,0,0], inplace=True)
                else:
                    bullet['mesh'].rotate_vector((pt[1], -pt[0], 0), -theta, point=[0,0,0], inplace=True)

        # Perturb bullet location
        if location==False:
            pass
        else:
            cone_angle_deg = location
            for i in range(rosette.n_arms):
                bullet = rosette.bullets[i]
                pt = np.array(bullet['anchor_point'])
                new_pt = helper.random_spherical_cap(cone_angle_deg, pt, 1)
                new_pt = new_pt[0,:]
                # print('pt: ', pt)
                # print('new_pt: ', new_pt)
                rot_axis = np.cross(pt, new_pt) # rotation axis
                pt_norm = helper.norm_rows(pt)
                theta = math.degrees(np.arccos(np.dot(pt_norm, new_pt))) # rotation angle in degrees
                bullet['mesh'].rotate_vector(rot_axis, theta, point=[0,0,0], inplace=True)

        return rosette

    def render(self, cam): 
        """
        Render orthographic (parallel) projection
        """
        pass

    def calc_mbs(self):
        """
        Calculate minimal bounding sphere (mbs)
        """
        self.mbs = {} # store attributes of sphere as dict

        # use miniball algorithm to find bounding sphere
        mesh_points = np.asarray(self.model.points)
        c, r2 = miniball.get_bounding_ball(mesh_points)
        r = np.sqrt(r2) # r2 = radius squared, r = radius

        self.mbs['c'] = c # center coordinates as np array
        self.mbs['r'] = r # radius of sphere as float
        self.mbs['v'] = (4/3)*np.pi*(r**3)

        return self.mbs

    def calc_rho_eff_ratio(self):
        """
        Calculate effective density ratio
        I.e. volume of rosette / volume of bounding sphere
        """
        rho_eff_ratio = self.volume / self.mbs['v']
        self.rho_eff_ratio = rho_eff_ratio
        return rho_eff_ratio
    
    def repair_mesh(self):
        """
        Fix any potential mesh issues
        """
        meshfix = mf.MeshFix(self.model)
        meshfix.repair(verbose=False)

    def make_hollow(self, hh, hh_sigma=0.5, hollow_rand=False):
        """
        Add hollowing to a bullet arm 
        hh = height of hollowing cone
        hollow_rand = True -> randomly sample depth from normal distribution
        hh_sigma = standard deviation of hh as a percentage 
        Return a hollow bullet
        """
        bullet = self.original_bullet
        if hollow_rand: 
            hh = np.random.normal(hh, hh - hh_sigma * hh)
        cone = pv.Cone(center=(0.0, 0.0, self.hp/2), direction=(0.0, 0.0, 1.0),  
                    height=hh, radius=self.a, capping=True, angle=None, resolution=6)
        cone = cone.triangulate() 
        hollow_bullet = bullet.boolean_difference(cone).triangulate()
        return hollow_bullet
