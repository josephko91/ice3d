import pyvista as pv
import numpy as np
from math import pi, sqrt, pow, acos, degrees, radians, sin, cos
# import trame
import math
# import helper
from data import helper_temp as helper
from copy import deepcopy
import miniball
import pymeshfix as mf
from decimal import *

class Rosette:
    """
    Class representing bullet rosette ice crystals
    """
    def __init__(self, a, c, r0, h0, hp, n_arms, hollow=False, hh=0, hh_sigma=0.05, hollow_rand=False):
        # geoemetric parameters
        self.a = a # half max length across basal face
        self.c = c # half max length across prism face
        self.r0 = r0 # radius of center sphere
        self.h0 = h0 # penetration depth of bullets
        self.hp = hp # heights of pyramid of bullets
        self.n_arms = n_arms # number of bullet arms
        self.hollow = hollow # hollowing turned on or off?
        self.hh = hh # height of hollowing cone as % of cylinder length
        self.hh_sigma = hh_sigma # std dev of hh as a percentage
        self.hollow_rand = hollow_rand# randomize hollowing or not

        # create sphere 
        sphere = pv.Sphere(radius=r0, center=(0, 0, 0), direction=(0, 0, 1), 
                        theta_resolution=30, phi_resolution=20, start_theta=0, 
                        end_theta=360, start_phi=0, end_phi=180)
        self.sphere = sphere.triangulate()

        # create outer shell to "place" bullets on
        r_outer = hp/2 + c - h0 + r0
        self.outer_sphere = pv.Sphere(radius=r_outer, center=(0, 0, 0), direction=(0, 0, 1), 
                        theta_resolution=30, phi_resolution=20, start_theta=0, 
                        end_theta=360, start_phi=0, end_phi=180)
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

        self.outer_coords = outer_coords # for testing 

        # create bullet arm
        cyl = pv.Cylinder(center=(0.0, 0.0, c+hp), direction=(0.0, 0.0, -1.0), 
                        radius=a, height=2*c, resolution=6, capping=True)
        pyr = pv.Cone(center=(0.0, 0.0, hp/2), direction=(0.0, 0.0, -1.0), 
                    height=hp, radius=a, capping=True, angle=None, resolution=6)
        cyl = cyl.triangulate()
        pyr = pyr.triangulate()
        cyl_pts = cyl.points
        cyl_pts[abs(cyl_pts)<1e-10]=0.0 # replace small values with zeros
        cyl.points = cyl_pts
        pyr_pts = pyr.points
        pyr_pts[abs(pyr_pts)<1e-10]=0.0 # replace small values with zeros
        pyr.points = pyr_pts
        bullet = cyl.boolean_union(pyr).triangulate()
        pt_dist = np.linalg.norm(bullet.points, axis=1)
        tip_pt_index = np.argmin(pt_dist) # index of the tip point in bullet.points
        self.bullet_center_default = bullet.center # for testing
        self.bullet_default = bullet

        # copy, translate, and rotate bullets
        self.bullets = {} # save bullets in nested dictionary
        for i in range(len(outer_coords)):
            # print('================')
            # print(f'bullet {i}')
            bullet_entry = {}
            pt = outer_coords[i]
            #rotate 
            bullet_center = self.bullet_center_default # center before any hollowing
            # print(f'bullet center = {bullet_center}')
            # print(f'pt = {pt}')
            cross_prod = np.cross(bullet_center, pt)
            
            if hollow == True: 
                # print('calling make_hollow()')
                bullet = self.make_hollow(hh, hh_sigma, hollow_rand) # hollow out arm
            # print(f'--- bullet {i+1} ---')
            # print(f'cross product = {cross_prod}')
            # print(f'theta = {theta}')
            # print(cross_prod)
            # print(f'bullet center = {bullet_center}')
            # print(np.linalg.norm(bullet_center))
            # print(np.linalg.norm(pt))
            # print((np.linalg.norm(bullet_center)*np.linalg.norm(pt)))
            theta = degrees( acos (np.dot(bullet_center, pt) / (np.linalg.norm(bullet_center)*np.linalg.norm(pt))) )
            # print(f'theta = {theta}')
            if not np.any(np.array(cross_prod)): # if all zeros
                # print('test 1')
                if theta == 180:
                    bullet_rot = bullet.rotate_x(180)
                else:
                    bullet_rot = bullet
                new_center = bullet_rot.center
            else:
                rads = radians(theta)
                transform_mat = helper.rotate_axis_angle(cross_prod, rads)
                rot_mat = transform_mat[:3, :3]
                new_center = rot_mat @ bullet_center # matrix multiply to rotate
                bullet_rot = bullet.transform(transform_mat, inplace=False)

            # translate
            # print(f'new_center = {new_center}')
            translate_vector = pt - new_center

            # print(f'translate_vector = {translate_vector}')
            bullet_final = bullet_rot.translate(translate_vector, inplace=False)
            # bullet_final = bullet_rot

            # add bullet attributes and mesh to dictionary
            bullet_entry['bullet_rot_center'] = new_center
            bullet_entry['mesh'] = bullet_final.triangulate()
            bullet_entry['xy_scale_factor'] = 1.0
            bullet_entry['z_scale_factor'] = 1.0
            bullet_entry['anchor_point'] = pt
            self.bullets[i] = bullet_entry

    def est_vol(self):
        """
        Estimate volume
        """
        bullets = self.bullets
        sphere = self.sphere
        v_sphere = sphere.volume
        a = self.a 
        h = self.h0
        n_arms = self.n_arms
        v_bullets = 0
        v_tips = 0 
        for _, b in bullets.items():
            v_bullets += b['mesh'].volume
            a_scaled = a*b['xy_scale_factor']
            v_tips += ((sqrt(3)/2)*pow(a_scaled, 2)*h)
        v = v_sphere + v_bullets - v_tips
        self.volume = v
        return v

    def est_sa(self):
        """
        Estimate surface area
        """
        bullets = self.bullets
        sphere = self.sphere
        a = self.a 
        h = self.h0
        sa_sphere = sphere.area
        sa_bullets = 0
        sa_tips = 0
        sa_sphere_holes = 0
        for _, b in bullets.items():
            sa_bullets += b['mesh'].area
            a_scaled = a*b['xy_scale_factor']
            sa_tips += 3*a_scaled*sqrt(pow(h, 2) + (3*pow(a_scaled, 2))/4)
            area_hole = (3/2) * sqrt(3) * pow(a_scaled, 2)
            sa_sphere_holes += area_hole
        sa = sa_sphere + sa_bullets - sa_sphere_holes - sa_tips
        return sa 
    
    def create_multiblock(self):
        sphere = self.sphere
        bullets = self.bullets
        block = pv.MultiBlock()
        block.append(sphere)
        for _, b in bullets.items():
            bullet = b['mesh']
            block.append(bullet)
        self.block = block

    def unify_mesh(self): 
        """
        Create single mesh using boolean union operation
        """
        rosette = self.sphere
        for i in range(self.n_arms):
            bullet = self.bullets[i]
            bullet_mesh = bullet['mesh'].triangulate()
            rosette = rosette.boolean_union(bullet_mesh).triangulate()

            # # test feature: added 20230605
            # meshfix = mf.MeshFix(rosette)
            # meshfix.repair(verbose=True)
            # rosette = meshfix.mesh

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
                # theta = math.degrees(math.acos(pt_2/r_outer_round))

                ### --------- TESTING -------------
                # rotate to north unit vector
                unit_north = np.array([0,0,1])
                cross_prod = np.cross(pt, unit_north)
                theta = degrees( acos (np.dot(pt, unit_north) / (np.linalg.norm(pt)*np.linalg.norm(unit_north))) )
                if not np.any(np.array(cross_prod)): # if all zeros
                    if theta == 180:
                        bullet['mesh'].rotate_x(180, inplace=True)
                    else:
                        bullet['mesh'].rotate_x(0, inplace=True)
                    # new_center = bullet_rot.center
                else:
                    rads = radians(theta)
                    transform_mat = helper.rotate_axis_angle(cross_prod, rads)
                    # rot_mat = transform_mat[:3, :3]
                    # new_center = rot_mat @ bullet.center # matrix multiply to rotate
                    bullet['mesh'].transform(transform_mat, inplace=True)
                ### -------------------------------
                # # rotate parallel to north unit vector
                # if (pt[0]==0 and pt[1]==0):
                #     bullet['mesh'].rotate_vector((0, pt[2], -pt[1]), theta, point=[0,0,0], inplace=True)
                # else:
                #     bullet['mesh'].rotate_vector((pt[1], -pt[0], 0), theta, point=[0,0,0], inplace=True)
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

                # # rotate back to place
                # if (pt[0]==0 and pt[1]==0):
                #     bullet['mesh'].rotate_vector((0, pt[2], -pt[1]), -theta, point=[0,0,0], inplace=True)
                # else:
                #     bullet['mesh'].rotate_vector((pt[1], -pt[0], 0), -theta, point=[0,0,0], inplace=True)
                 ### --------- TESTING -------------
                # rotate back to place
                cross_prod = np.cross(unit_north, pt)
                theta = degrees( acos (np.dot(unit_north, pt) / (np.linalg.norm(unit_north)*np.linalg.norm(pt))) )
                if not np.any(np.array(cross_prod)): # if all zeros
                    if theta == 180:
                        bullet['mesh'].rotate_x(180, inplace=True)
                    else:
                        bullet['mesh'].rotate_x(0, inplace=True)
                    # new_center = bullet_rot.center
                else:
                    rads = radians(theta)
                    transform_mat = helper.rotate_axis_angle(cross_prod, rads)
                    # rot_mat = transform_mat[:3, :3]
                    # new_center = rot_mat @ bullet.center # matrix multiply to rotate
                    bullet['mesh'].transform(transform_mat, inplace=True)
                ### -------------------------------
                # update scaling factors 
                bullet['xy_scale_factor'] = sf_basal
                bullet['z_scale_factor'] = sf_prism

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
        unique_pts = np.unique(mesh_points, axis=0)
        c, r2 = miniball.get_bounding_ball(unique_pts)
        r = np.sqrt(r2) # r2 = radius squared, r = radius

        self.mbs['c'] = c # center coordinates as np array
        self.mbs['r'] = r # radius of sphere as float
        self.mbs['v'] = (4/3)*np.pi*(r**3)

        return self.mbs
    
    def calc_mbs2(self):
        """
        Calculate minimal bounding sphere (mbs) without needing to boolean union 
        the different parts together
        """
        self.mbs = {} # store attributes of sphere as dict

        # use miniball algorithm to find bounding sphere
        mesh_points = []
        for _, bullet in self.bullets.items(): 
            pts = bullet['mesh'].points
            mesh_points.append(pts)
        mesh_points = np.vstack(mesh_points)
        unique_pts = np.unique(mesh_points.round(decimals=2), axis=0)
        unique_dist = np.linalg.norm(unique_pts, axis=1)
        indx = np.argsort(unique_dist) # corresponding index of sorted dist
        n_pts = len(unique_pts)
        subset_indx = indx[-int(n_pts/2):]
        unique_pts = unique_pts[subset_indx]
        # print(unique_pts)
        c, r2 = miniball.get_bounding_ball(unique_pts)
        r = np.sqrt(r2) # r2 = radius squared, r = radius

        self.mbs['c'] = c # center coordinates as np array
        self.mbs['r'] = r # radius of sphere as float
        self.mbs['a'] = 4 * np.pi * pow(r, 2) # surface area
        self.mbs['v'] = (4/3)*np.pi*(r**3) # volume

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
    
    def make_hollow(self, hh, hh_sigma=0.05, hollow_rand=False):
        """
        Add hollowing to a bullet arm 
        hh = height of hollowing cone as % of cylinder length
        hollow_rand = True -> randomly sample depth from normal distribution
        hh_sigma = standard deviation of hh as a percentage 
        Return a hollow bullet
        """
        bullet = self.bullet_default
        hollow_length = hh*self.c*2
        # print(f'c = {self.c}')
        # print(f'hh_sigma = {hh_sigma}')
        if hollow_rand: 
            # print(f'hollow length = {hollow_length}')
            sigma = hh_sigma*hollow_length
            # print(f'sigma = {sigma}')
            hh_l = np.random.normal(hollow_length, sigma)
            # print(f'random hollowing length {hh_l}')
        else:
            hh_l = hollow_length
        # 0.01 fudge factor added because of boolean issue
        cone = pv.Cone(center=(0.0, 0.0, 2*self.c+self.hp-hh_l/2+0.05), 
                        direction=(0.0, 0.0, -1.0),  
                        height=hh_l, radius=self.a, capping=True, angle=None, resolution=6)
        cone = cone.triangulate() 
        hollow_bullet = bullet.boolean_difference(cone).triangulate()
        return hollow_bullet