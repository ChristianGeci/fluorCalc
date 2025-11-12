import numpy as np
import matplotlib.pyplot as plt
import mcint
import random
import xraydb

class experiment:
    """
    Encapsulates an XAFS experimental configuration and calculates
    and expected fluorescence count rate for that configuration.
    """
    def __init__(self,
                composition: dict[str, float], density: float, 
                element: str, edge: str, emission_line: str,
                theta = 45,
                R = np.sqrt(170/4/np.pi),
                beam_width = 3, beam_height = 1,
                photon_flow = 7.7E9,
                detector_distance = 10,
                nmc = 50000,
                suppress_output = False,
                detector_above_sample = False):
        """
        *Summary:
            Parses inputs related to experimental configuration and
            calculates expected x-ray fluorescence count rate.
        *Keyword arguments:
            theta -- angle (degrees) of incidence between the beam and sample
            R -- radius (mm) of detector
            beam_width -- width (mm) of x-ray beam 
            beam_height -- height (mm) of x-ray beam
            photon_flow -- intensity (photons per second) of x-ray beam
            quantum yield -- fluorescence yield of absorbing atom (dimensionless)
            detector_distance -- distance (mm) from the center of the detector to the center of the sample
            mu_T_Ef -- attenuation coefficient (1/mm) of fluoresced photons traversing the sample
            mu_T_E -- attenuation coefficient (1/mm) of incident photons traversing the sample
            mu_i -- absorption coefficient (1/mm) of incident photons by the absorbing atom
            nmc -- number of samples in Monte Carlo integral (dimensionless)
            suppress_output -- set to false to prevent printing calculation results to terminal
            detector_above_sample -- specify whether the detector is in the plane of the ring (false) or is mounted above the sample (true)
        """
        
        def process_inputs():
            """Write constructor arguments to object."""
            self.composition = composition
            self.density = density
            self.absorbing_element = element
            self.edge = edge
            self.emission_line = emission_line
            self.suppress_output = suppress_output
            self.theta = theta*np.pi/180
            self.R = R
            self.d = 2*self.R
            self.detector_distance = detector_distance
            self.beam_width = beam_width
            self.beam_height = beam_height
            self.detector_above_sample = detector_above_sample
            self.photon_flow = photon_flow
            self.nmc = nmc
        process_inputs()

        def normalize_composition():
            """
            Normalize the stoichiometric composition such that the relative
            fractions of all elements sum to one.
            """
            stoichimetry_sum = sum(self.composition.values())
            for element in self.composition:
                self.composition[element] /= stoichimetry_sum
        normalize_composition()

        def get_photon_energies_and_quantum_yield():
            """
            Get incident/detected photon energy based on absorbing
            element, absorption edge, and fluorescence line. Also
            get the quantum yield.
            """
            xraydb_edge = xraydb.xray_edge(self.absorbing_element, edge)
            self.incident_photon_energy = xraydb_edge.energy
            xraydb_fluor_yield = xraydb.fluor_yield(element, edge, emission_line, self.incident_photon_energy)
            total_quantum_yield, self.detected_photon_energy, measured_fraction = xraydb_fluor_yield
            self.quantum_yield = total_quantum_yield * measured_fraction
        get_photon_energies_and_quantum_yield()

        def get_attenuation_coefficients():
            """
            Calculate attenuation coefficients based on sample composition
            and photon energies.
            """
            def get_incoming_bulk_attenuation():
                """
                Attenuation coefficient of incident photons traversing
                the bulk sample
                """
                result = 0
                for element, fraction in self.composition.items():
                    result += (xraydb.mu_elam(element, self.incident_photon_energy)
                                * self.density * fraction
                                / 10) # factor of 10 converts from 1/cm to 1/mm
                return result
            def get_outgoing_bulk_attenuation():
                """
                Attenuation coefficient of fluoresced photons traversing
                the bulk sample
                """
                result = 0
                for element, fraction in self.composition.items():
                    result += (xraydb.mu_elam(element, self.detected_photon_energy)
                                * self.density * fraction
                                / 10) # factor of 10 converts from 1/cm to 1/mm
                return result
            def get_absorbing_atom_attenuation():
                """
                Absorption coefficient of incident photons by the
                absorbing element
                """
                absorbing_element_fraction = self.composition[self.absorbing_element]
                result = (xraydb.mu_elam(self.absorbing_element, self.incident_photon_energy)
                        * self.density * absorbing_element_fraction
                        / 10)    # factor of 10 converts from 1/cm to 1/mm
                return result
            self.mu_T_E = get_incoming_bulk_attenuation()
            self.mu_T_Ef = get_outgoing_bulk_attenuation()
            self.mu_i = get_absorbing_atom_attenuation()
        get_attenuation_coefficients()

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

        def lump_constants():
            """
            Lump together a bunch of constants that will just end
            up coming out in front of the integral.
            """
            self.consts = (self.photon_flux
                           * self.quantum_yield
                           * self.mu_i
                           / (4 * np.pi)
                           / np.sin(self.theta))
        lump_constants()
        
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

        def find_y_depth_limit():
            """
            Calculate how deeply into the sample slab we will integrate
            by finding at what point the beam intensity has dropped below
            1%.
            """
            depth = 0
            while True:
                if (self.in_atten(depth) < 0.01):
                    self.y_depth_limit = depth
                    break
                depth -= 1E-5
        find_y_depth_limit()

        def compute_full_integral_count_rate():
            """
            Compute the fluorescence count rate within our integral
            formalism.
            """
            def integrand(S):
                """
                Integrand of the Monte Carlo integral, return a photon count
                rate per unit volume per unit area of phase space.
                """
                x = S[0]
                y = S[1]
                z = S[2]
                r = S[3]
                phi = S[4]
                return (self.consts * r
                        * self.cosine_term(x, y, z, r, phi)
                        / (self.dist(x, y, z, r, phi)**2)
                        * self.out_atten(x,y,z,r,phi)
                        * self.in_atten(y))
            def sampler():
                """Sample phase space for the Monte Carlo integration."""    
                while True:
                    z = random.uniform(0, self.z_illum)
                    y = random.uniform(self.y_depth_limit, 0)
                    x = random.uniform(-y/np.tan(self.theta), -y/np.tan(self.theta) + self.x_illum)
                    r = random.uniform(0, self.R)
                    phi = random.uniform(0, 2*np.pi)
                    yield (x, y, z, r, phi)
            random.seed(1)
            domainsize = -self.z_illum*self.x_illum*self.y_depth_limit*self.R*2*np.pi
            self.result, self.error = mcint.integrate(integrand, sampler(), measure = domainsize, n=self.nmc)
        compute_full_integral_count_rate()

        def compute_simple_count_rate():
            """
            Compute the count rate according to the approximate description
            given in Bunker.
            """
            f = self.mu_T_E/np.sin(self.theta) + self.mu_T_Ef/np.sin(np.pi/2 - self.theta)
            apex_angle = np.arctan(self.R/self.detector_distance)
            self.solid_angle = 4*np.pi*np.sin(apex_angle/2)**2

            g = self.quantum_yield*self.solid_angle/4/np.pi/f
            self.crude_result = g*self.mu_i*self.photon_flow/np.sin(self.theta)
        compute_simple_count_rate()

        def print_output():
            print(f"expected count rate: {self.result:.3e} photons per second")
            print("Using n = ", self.nmc)
            print(f"estimated error = {self.error:.2e} = {self.error/self.result*100:.3}%")
            print(f"solid angle fraction: {self.solid_angle/np.pi*4:.4f}")
            print(f"crude count rate: {self.crude_result:.3e} photons per second, difference of {np.absolute(100*(self.crude_result - self.result)/self.result):.2f}%")
        if not self.suppress_output:
            print_output()

    def polar_to_cartesian(self, r, phi):
        """
        Converts a vector in 2D polar coordinates into 2D Cartesian
        coordinates.
        """
        return np.array([r*np.cos(phi), r*np.sin(phi), 0])

    def detector_coordinates_to_sample_coordinates(self, r, phi):
        """
        Accepts polar coordinates on the detector surface (r, phi)
        and converts them to detector coordinates.
        """
        return np.matmul(self.M, self.polar_to_cartesian(r, phi))+self.detector_position

    def dist(self, x, y, z, r, phi): #
        """
        Return the distance between a point (x, y, z) 
        in the sample and (r, phi) on the detector.
        """
        return np.linalg.norm(self.path(x, y, z, r, phi))

    def out_atten(self, x, y, z, r, phi):
        """
        *Summary:
            Calculates the attenuation experienced by a fluoresced photon
            as it traverses the sample to the detector.
        *Explanation:
            Accepts a point in phase space, represented by the absorption
            site in the sample slab (x, y, z; sample coordinates) and
            the point on the detector where the fluoresced photon is absorbed
            (r, phi; polar coordinates on the surface of the detector).
        """
        distance_thru_sample = (y * self.dist(x, y, z, r, phi)
                                / (self.detector_coordinates_to_sample_coordinates(r, phi)[1]-y))
        return np.exp(distance_thru_sample*self.mu_T_Ef)

    def in_atten(self, y):
        """
        Calculate attenuation experienced by a photon absorbed at
        a given point in phase space (depends only upon depth
        into the sample, measured in the sample coordinate system).
        """
        return np.exp(y/np.sin(self.theta)*self.mu_T_E)

    def path(self, x, y, z, r, phi):
        """
        Returns a vector in sample coordinates that points from a given 
        point in the bulk sample (x, y, z) to a given point on the 
        detector (r, phi).
        """
        sample_point = np.array([x, y, z])
        detector_point = self.detector_coordinates_to_sample_coordinates(r, phi)
        return detector_point - sample_point

    def cosine_term(self, x, y, z, r, phi):
        """
        Return the diminution in flux that occurs to off-normal
        incidence.
        """
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
