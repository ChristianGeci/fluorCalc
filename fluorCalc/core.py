import numpy as np
import matplotlib.pyplot as plt
import mcint
import random
import xraydb

class experiment:
    def __init__(self,
                theta = 6,
                R = np.sqrt(170/4/np.pi),
                beam_width = 3,
                beam_height = 1,
                photon_flow = 7.7E9,
                quantum_yield = 0.08,
                #alpha = np.pi/2,
                #beta = np.pi/2,
                detector_distance = 10,
                mu_T_Ef = 222.222,
                mu_T_E = 185.1852,
                mu_i = 2.8662,
                nmc = 50000,
                suppress_output = False,
                detector_above_sample = False):
        """
        *Keyword arguments:
            placeholder
        """
        
        def define_sample_coordinates():
            """
            *Summary:
                Define the unit vectors x_s, y_s, and z_s, which form
                the basis of the sample coordinate scheme (in mm).
            *Explanation:
                y_s is normal to the sample surface.
                z_s is parallel to the sample's axis of rotation.
                x_s is the direction along the sample surface that the
                illuminated region "stretches" along as the sample is rotated.
            """
            self.x_s = np.array([1, 0, 0])
            self.y_s = np.array([0, 1, 0])
            self.z_s = np.array([0, 0, 1])
        define_sample_coordinates()

        def process_inputs():
            """Write constructor arguments to object."""
            self.suppress_output = suppress_output
            self.theta = theta*np.pi/180
            self.R = R
            self.d = 2*self.R
            self.detector_distance = detector_distance
            self.beam_width = beam_width
            self.beam_height = beam_height
            self.detector_above_sample = detector_above_sample
            self.quantum_yield = quantum_yield
            self.photon_flow = photon_flow
            self.mu_T_Ef = mu_T_Ef
            self.mu_T_E = mu_T_E
            self.mu_i = mu_i
        process_inputs()
        
        def calculate_illuminated_area():
            """
            *Summary:
                Find the dimensions (in mm) of the sample illuminated by the beam.
                Also calculate the photon flux.
            *Explanation:
                The "beam height" is taken as the vertical dimension of the
                beam from the perspective of the user. If the detector is
                mounted in the plane of the ring (as is typical), then z_s
                is the vertical axis from the user's point of view and the
                length of the illuminated area along the z_s direction
                (z_illum) is determined by the height of the beam. Otherwise,
                z_illum is determined by the width of the beam.
            """
            if (not detector_above_sample):
                self.z_illum = beam_height
                self.x_illum = beam_width/np.sin(self.theta)
            else:
                self.z_illum = beam_width
                self.x_illum = beam_height/np.sin(self.theta)
            self.photon_flux = self.photon_flow/self.z_illum/self.x_illum
        calculate_illuminated_area()
        
        def calculate_detector_position():
            """
            Get the position of the detector <x, y, z> in the
            sample coordinate scheme.
            """
            x = np.sin(self.theta)*detector_distance+self.x_illum/2
            y = np.cos(self.theta)*detector_distance
            z = self.z_illum/2
            self.detector_position = np.array([x, y, z])

            if y - np.sin(self.theta)*self.R <= 0:
                raise ValueError('Detector clips into sample slab!')
        calculate_detector_position()

        def calculate_detector_orientation():
            """
            Get alpha and beta, angles which specify the orientation of 
            a unit vector normal to the detector surface in a polar
            coordinate scheme.
            """
            # It's hard-coded that the detector is at a 90 degree angle
            # with respect to the beam.
            self.beta = 3*np.pi/2 - self.theta
            self.alpha = -np.pi/2
        calculate_detector_orientation()
        
        def define_detector_coordinate_system():
            """Define a new orthonormal set of basis vectors
            x_d, y_d, z_d, where z_d is taken as the direction
            normal to the detector surface. These specify
            "detector coordinates"."""
            self.z_d = np.array([np.sin(self.alpha)*np.cos(self.beta), 
                                 np.sin(self.alpha)*np.sin(self.beta), 
                                 np.cos(self.alpha)])
            self.x_d = np.cross(self.y_s, self.z_d)
            self.x_d /= np.linalg.norm(self.x_d)
            self.y_d = np.cross(self.z_d, self.x_d)
            self.y_d /= np.linalg.norm(self.y_d)
        define_detector_coordinate_system()
        
        def calculate_transformation_matrices():
            """
            Calculate a matrix M that converts from detector coorinates
            to sample coordinates and a matrix W that converts from
            sample coordinates to detector coordinates.
            """
            self.M = np.array([self.x_d, self.y_d, self.z_d]).T
            self.W = np.linalg.inv(self.M)
        calculate_transformation_matrices()

        self.consts = self.photon_flux*self.quantum_yield*self.mu_i/4/np.pi/np.sin(self.theta)

        self.y_depth_limit = self.find_y_depth_limit()

        self.domainsize = -self.z_illum*self.x_illum*self.y_depth_limit*self.R*2*np.pi
        self.nmc = nmc

        self.compute_count_rate()

        self.quick_and_dirty_count_rate()

    def compute_count_rate(self):
        random.seed(1)
        self.result, self.error = mcint.integrate(self.integrand, self.sampler(), measure = self.domainsize, n=self.nmc)

        if (not self.suppress_output):
            print(f"expected count rate: {self.result:.3e} photons per second")
            print("Using n = ", self.nmc)
            print(f"estimated error = {self.error:.2e} = {self.error/self.result*100:.3}%")

    def quick_and_dirty_count_rate(self):
        f = self.mu_T_E/np.sin(self.theta) + self.mu_T_Ef/np.sin(np.pi/2 - self.theta)
        #solid_angle = np.pi*self.R**2/(self.detector_distance**2) #estimation of solid angle
        #apex_angle = np.arcsin(self.R/self.detector_distance)
        apex_angle = np.arctan(self.R/self.detector_distance)
        solid_angle = 4*np.pi*np.sin(apex_angle/2)**2

        g = self.quantum_yield*solid_angle/4/np.pi/f
        self.crude_result = g*self.mu_i*self.photon_flow/np.sin(self.theta)

        if (not self.suppress_output):
            print(f"solid angle fraction: {solid_angle/np.pi*4:.4f}")
            print(f"crude count rate: {self.crude_result:.3e} photons per second, difference of {np.absolute(100*(self.crude_result - self.result)/self.result):.2f}%")


    def polar_to_cartesian(self, r, phi):
        return np.array([r*np.cos(phi), r*np.sin(phi), 0])

    def detector_coordinates_to_sample_coordinates(self, r, phi):
        return np.matmul(self.M, self.polar_to_cartesian(r, phi))+self.detector_position


    def dist(self, x, y, z, r, phi): #computes the distance between a point x,y,z, in the sample and r,phi on the detector
        return np.linalg.norm(np.array([x, y, z]) - self.detector_coordinates_to_sample_coordinates(r, phi))

    def o_atten(self, x, y, z, r, phi):
        return np.exp(y/(self.detector_coordinates_to_sample_coordinates(r, phi)[1]-y)*self.dist(x, y, z, r, phi)*self.mu_T_Ef)

    def in_atten(self, y):
        return np.exp(y/np.sin(self.theta)*self.mu_T_E)

    #function that finds the bound of integration for y by looking at the depth at which the beam's intensity attenuates below 1% of I_0
    def find_y_depth_limit(self):
        for depth in np.arange(0, -100, -0.01/1000):
            if (self.in_atten(depth) < 0.01):
                return depth


    def integrand(self, S):
        x = S[0]
        y = S[1]
        z = S[2]
        r = S[3]
        phi = S[4]
        return self.cosine_term(x, y, z, r, phi)*r/(self.dist(x, y, z, r, phi)**2)*self.o_atten(x,y,z,r,phi)*self.in_atten(y)*self.consts

    def sampler(self):
        while True:
            z = random.uniform(0, self.z_illum)
            y = random.uniform(self.y_depth_limit, 0)
            x = random.uniform(-y/np.tan(self.theta), -y/np.tan(self.theta) + self.x_illum)
            r = random.uniform(0, self.R)
            phi = random.uniform(0, 2*np.pi)
            yield (x, y, z, r, phi)

    #returns a vector in sample coordinates that points from a gives point in the bulk sample to a given point on the detector
    def path(self, x, y, z, r, phi):
        detector_point = self.detector_coordinates_to_sample_coordinates(r, phi)
        return np.array([detector_point[0] - x, detector_point[1] - y, detector_point[2] -z])

    def cosine_term(self, x, y, z, r, phi):
        path_vector = self.path(x, y, z, r, phi)
        return np.absolute(np.dot(path_vector, self.z_d)/np.linalg.norm(path_vector))

class angle_sweep:
    def __init__(self, min_angle, max_angle, step_size, template_experiment, nmc = 0):
        self.angles = np.arange(min_angle, max_angle, step_size)
        self.experiments = []

        Nmc = nmc if nmc != 0 else template_experiment.nmc

        for angle in self.angles:
            self.experiments.append(experiment(
                R = template_experiment.R,
                beam_width = template_experiment.beam_width,
                beam_height = template_experiment.beam_height,
                photon_flow = template_experiment.photon_flow,
                quantum_yield = template_experiment.quantum_yield,
                detector_distance = template_experiment.detector_distance,
                mu_T_Ef = template_experiment.mu_T_Ef,
                mu_T_E = template_experiment.mu_T_E,
                mu_i = template_experiment.mu_i,
                nmc = Nmc,
                suppress_output = True,
                detector_above_sample = template_experiment.detector_above_sample,
                theta = angle))

    def plot_result(self, plot_simple_calculation_result = True, label = "", title = ""):
        yvals = [entry.result for entry in self.experiments]
        yvals = np.array(yvals)

        crude_yvals = [entry.crude_result for entry in self.experiments]
        crude_yvals = np.array(crude_yvals)

        plt.plot(self.angles, yvals, label = label + " integrated calculation")
        if plot_simple_calculation_result:
            plt.plot(self.angles, crude_yvals, label = label + " simple calculation", linestyle='dashed')
        plt.xlabel("angle (°)")
        plt.ylabel("count rate (photons/s)")
        plt.title(title)

        plt.legend()

    def get_optimal_angle(self, suppress_output = False):
        max_angle = 0
        max_flux = 0
        for angle, Experiment in zip(self.angles, self.experiments):
            if Experiment.result > max_flux:
                max_flux = Experiment.result
                max_angle = angle

        if not suppress_output:
            print(f"maximum flux of {max_flux:.3e} at angle {max_angle}")

        return max_angle, max_flux

class mu_package:
    def __init__(self, composition: dict[str, float], density: float, edge: str, detected_photon_energy: float):
        self.composition = composition
        self.density = density
        self.absorbing_element = edge.split()[0]
        self.edge = xraydb.xray_edge(self.absorbing_element, edge.split()[1])
        self.detected_photon_energy = detected_photon_energy
        self.normalize_composition()
        self.get_attenuation_coefficients()
        pass

    def normalize_composition(self):
        stoichimetry_sum = sum(self.composition.values())
        for element in self.composition:
            self.composition[element] /= stoichimetry_sum

    def get_attenuation_coefficients(self):

        def get_incoming_bulk_attenuation():
            result = 0
            for element, fraction in self.bulk_composition.items():
                result += xraydb.mu_elam(element, self.edge.energy) * self.density * fraction
            return result / 10    # factor of 10 converts from 1/cm to 1/mm

        def get_outgoing_bulk_attenuation():
            result = 0
            for element, fraction in self.composition.items():
                result += xraydb.mu_elam(element, self.detected_photon_energy) * self.density * fraction
            return result / 10    # factor of 10 converts from 1/cm to 1/mm
        
        def get_absorbing_atom_attenuation():
            result = xraydb.mu_elam(self.absorbing_element, self.edge.energy) * self.density * self.composition[self.absorbing_element] / 10    # factor of 10 converts from 1/cm to 1/mm
            return result

        self.mu_T_E = get_incoming_bulk_attenuation()
        self.mu_T_Ef = get_outgoing_bulk_attenuation()
        self.mu_i = get_absorbing_atom_attenuation()


            
    @property
    def bulk_composition(self):
        """Return the composition dictionary with the absorbing element removed."""
        return {item: self.composition[item] for item in self.composition if item != self.absorbing_element}


def test_function():
    my_composition = {
        'Mo': 1,
        'O': 2,
        'P': 0.001,
    }

    my_mu = mu_package(my_composition, 6.47/2, "P K", 2300)

    default_experiment = experiment(nmc = 1000)

test_function()