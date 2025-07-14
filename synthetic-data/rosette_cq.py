import numpy as np
import cadquery as cq
import random
import miniball

class Rosette:
    """
    A class for creating and analyzing 3D rosette geometries using CadQuery.
    """

    def __init__(self, base_params, n_arms, s_code, perturb_aspect_ratio, perturb_s_code_switch=0):
        """
        Initialize a Rosette object.

        Parameters:
        -----------
        base_params : list
            [a, c, f_r0, f_hp, f_h0] - base geometric parameters
        n_arms : int
            Number of arms/bullets in the rosette
        s_code : list
            Spherical coordinates for bullet placement
        perturb_aspect_ratio : list
            Aspect ratio perturbations [f_a_1, f_c_1, ..., f_a_n_arms, f_c_n_arms]
        perturb_s_code_switch : int
            1 to perturb s_code, 0 for no perturbation
        """
        self.base_params = base_params
        self.n_arms = n_arms
        self.s_code = s_code
        self.perturb_aspect_ratio = perturb_aspect_ratio
        self.perturb_s_code_switch = perturb_s_code_switch
        self.ros = None
        self._calculated_params = {}

    @staticmethod
    def norm_rows(v):
        if np.all(v == 0):
            v_unit = np.array([1, 0, 0])
        else:
            if v.ndim == 1:
                v_norm = np.linalg.norm(v)
                v_unit = v / v_norm
            else:
                v_norm = np.linalg.norm(v, axis=1)
                v_unit = v / v_norm[:, None]
        return v_unit

    @staticmethod
    def random_spherical_cap(cone_angle_deg, cone_direction, num_points):
        cone_angle_rad = cone_angle_deg * (np.pi / 180)
        z = np.random.uniform(np.cos(cone_angle_rad), 1, num_points)
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        x = np.sqrt(1 - z ** 2) * np.cos(phi)
        y = np.sqrt(1 - z ** 2) * np.sin(phi)
        points = np.column_stack((x, y, z))
        north_vector = np.array([0, 0, 1])
        cone_direction_norm = Rosette.norm_rows(cone_direction)
        u = Rosette.norm_rows(np.cross(north_vector, cone_direction_norm))
        rot = np.arccos(np.dot(cone_direction_norm, north_vector))
        ux, uy, uz = u[0], u[1], u[2]
        rot_mat = np.array([
            [np.cos(rot) + (ux ** 2) * (1 - np.cos(rot)), ux * uy * (1 - np.cos(rot)) - uz * np.sin(rot), ux * uz * (1 - np.cos(rot)) + uy * np.sin(rot)],
            [uy * ux * (1 - np.cos(rot)) + uz * np.sin(rot), np.cos(rot) + (uy ** 2) * (1 - np.cos(rot)), uy * uz * (1 - np.cos(rot)) - ux * np.sin(rot)],
            [uz * ux * (1 - np.cos(rot)) - uy * np.sin(rot), uz * uy * (1 - np.cos(rot)) + ux * np.sin(rot), np.cos(rot) + (uz ** 2) * (1 - np.cos(rot))]
        ])
        points_rot = np.matmul(rot_mat, points.T).T
        return points_rot

    @staticmethod
    def get_perturb_aspect_ratio(n_arms, f_a_c_limits):
        f_a_c = []
        for i in range(n_arms):
            f_a = random.uniform(f_a_c_limits[0], f_a_c_limits[1])
            f_c = random.uniform(f_a_c_limits[2], f_a_c_limits[3])
            f_a_c.append(f_a)
            f_a_c.append(f_c)
        return f_a_c

    @staticmethod
    def get_cone_angle(n_arms):
        min_angles = {4: 109.4712206, 5: 90.0, 6: 90.0, 7: 77.8695421, 8: 74.8584922, 9: 70.5287794, 10: 66.1468220}
        return min_angles[n_arms] / 3

    @classmethod
    def perturb_s_code(cls, n_arms, s_code):
        s_code_perturbed = []
        cone_angle_deg = cls.get_cone_angle(n_arms)
        for i in range(n_arms):
            cone_direction = np.array([s_code[3 * i], s_code[3 * i + 1], s_code[3 * i + 2]])
            points_rot = cls.random_spherical_cap(cone_angle_deg, cone_direction, 1)
            pt = points_rot[0]
            s_code_perturbed.extend(pt)
        return s_code_perturbed

    @staticmethod
    def _calc_r0(f_r0, a, n_arms):
        ymin, ymax = 0.5 * a, 1 * a
        xmin, xmax = 4, 12
        slope = (ymax - ymin) / (xmax - xmin)
        intercept = ymin - (slope * xmin)
        r0 = slope * n_arms + intercept
        r0 = f_r0 * r0
        return r0

    @staticmethod
    def _calc_hp(f_hp, r0, n_arms):
        ymin, ymax = 1 * r0, 1.5 * r0
        xmin, xmax = 4, 12
        slope = (ymax - ymin) / (xmax - xmin)
        intercept = ymin - (slope * xmin)
        hp = slope * n_arms + intercept
        hp = f_hp * hp
        return hp

    @staticmethod
    def _calc_h0(f_h0, r0):
        h0 = r0 / 2
        h0 = f_h0 * h0
        return h0

    @staticmethod
    def _extract_xyz(s_code):
        x, y, z = [], [], []
        for i in range(0, len(s_code), 3):
            x.append(s_code[i])
            y.append(s_code[i + 1])
            z.append(s_code[i + 2])
        return x, y, z

    @staticmethod
    def _create_bullet(a, c, hp, f_a, f_c, workplane):
        n_pyr = 6
        ri = a * np.cos(np.radians(30))
        theta = 90 - np.degrees(np.arctan(hp / ri))
        pyramid = workplane.polygon(n_pyr, f_a * 2 * a).extrude(-f_a * hp, taper=theta)
        n_cyl = 6
        cylinder = workplane.polygon(n_cyl, f_a * 2 * a).extrude(f_c * 2 * c)
        bullet = cylinder.union(pyramid)
        return bullet

    def create_geometry(self):
        a, c, f_r0, f_hp, f_h0 = self.base_params
        r0 = self._calc_r0(f_r0, a, self.n_arms)
        hp = self._calc_hp(f_hp, r0, self.n_arms)
        h0 = self._calc_h0(f_h0, r0)
        self._calculated_params = {'r0': r0, 'hp': hp, 'h0': h0}
        sphere = cq.Workplane().sphere(r0)
        r_outer = r0 + hp - h0

        s_code = self.s_code
        if self.perturb_s_code_switch == 1:
            s_code = self.perturb_s_code(self.n_arms, s_code)

        x, y, z = self._extract_xyz(s_code)
        outer_coords = r_outer * (np.column_stack((x, y, z)))
        bullets = []
        if self.perturb_aspect_ratio:
            f_a_c_limits = self.perturb_aspect_ratio
            f_a_c = self.get_perturb_aspect_ratio(self.n_arms, f_a_c_limits)
        else:
            f_a_c = [1.0, 1.0] * self.n_arms

        for i in range(len(outer_coords)):
            f_a = f_a_c[2 * i]
            f_c = f_a_c[2 * i + 1]
            normal_vector = tuple(outer_coords[i])
            plane = cq.Plane(origin=normal_vector, normal=normal_vector)
            workplane = cq.Workplane(plane)
            bullet = self._create_bullet(a, c, hp, f_a, f_c, workplane)
            bullets.append(bullet)
        ros = sphere.union(bullets[0])
        for i in range(1, self.n_arms):
            ros = ros.union(bullets[i])
        self.ros = ros
        return ros

    def _get_verts(self, threshold):
        if self.ros is None:
            raise ValueError("Rosette geometry not created. Call create_geometry() first.")
        verts = self.ros.vertices()
        origin = cq.Vertex.makeVertex(0, 0, 0)
        filtered_verts = [v for v in verts if v.distance(origin) > threshold / 2]
        final_verts = np.asarray([list(v.Center().toTuple()) for v in filtered_verts])
        final_verts = np.round(final_verts, 2)
        return final_verts

    def _calc_mbs(self, points):
        mbs = {}
        unique_pts = np.unique(points, axis=0)
        c, r2 = miniball.get_bounding_ball(unique_pts)
        r = np.sqrt(r2)
        mbs['c'] = c
        mbs['r'] = r
        mbs['v'] = (4 / 3) * np.pi * (r ** 3)
        mbs['a'] = 4 * np.pi * (r ** 2)
        return mbs

    def calculate_properties(self):
        if self.ros is None:
            raise ValueError("Rosette geometry not created. Call create_geometry() first.")
        try:
            sa = self.ros.val().Area()
            vol = self.ros.val().Volume()
            points = self._get_verts(self.base_params[1])
            mbs = self._calc_mbs(points)
            rho_eff = vol / mbs['v']
            sa_eff = sa / mbs['a']
            return {
                'surface_area': sa,
                'volume': vol,
                'surface_area_eff': sa_eff,
                'density_eff': rho_eff,
                'mbs': mbs
            }
        except Exception as e:
            print(f'Property calculation error: {e}')
            return {
                'surface_area': -999,
                'volume': -999,
                'surface_area_eff': -999,
                'density_eff': -999,
                'mbs': None
            }

    def export_stl(self, filepath):
        if self.ros is None:
            raise ValueError("Rosette geometry not created. Call create_geometry() first.")
        cq.exporters.export(self.ros, filepath)

    def get_record(self, id):
        record = [id]
        record.extend(self.base_params)
        if self.ros is None:
            sa = vol = sa_eff = rho_eff = -666
        else:
            props = self.calculate_properties()
            sa = props['surface_area']
            vol = props['volume']
            sa_eff = props['surface_area_eff']
            rho_eff = props['density_eff']
        record.extend([sa, vol, sa_eff, rho_eff])
        return record