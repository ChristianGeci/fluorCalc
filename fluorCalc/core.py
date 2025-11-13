import numpy as np
import matplotlib.pyplot as plt
import mcint
import random
import xraydb
from dataclasses import dataclass, astuple, replace

@dataclass
class Experimental_Configuration:
    composition: dict[str, float]
    density: float
    absorbing_element: str
    edge: str
    emission_line: str
    theta: float
    R: float
    beam_width: float
    beam_height: float
    photon_flow: float
    detector_distance: float
    detector_above_sample: bool
    def __iter__(self):
        return iter(astuple(self))

@dataclass
class Calculation_Result:
    configuration: Experimental_Configuration
    nmc: int
    count_rate: float
    montecarlo_error: float
    crude_count_rate: float
    solid_angle_fraction: float

def xafs_count_rate(c: Experimental_Configuration,
                nmc = 50000,
                suppress_output = False,):
    # Begin big block of helper functions (how to better organize this?)
    def polar_to_cartesian(r, phi):
        """
        Converts a vector in 2D polar coordinates into 2D Cartesian
        coordinates.
        """
        return np.array([r*np.cos(phi), r*np.sin(phi), 0])
    def detector_coordinates_to_sample_coordinates(r, phi):
        """
        Accepts polar coordinates on the detector surface (r, phi)
        and converts them to detector coordinates.
        """
        return np.matmul(M, polar_to_cartesian(r, phi))+detector_position
    def dist(x, y, z, r, phi):
        """
        Return the distance between a point (x, y, z) 
        in the sample and (r, phi) on the detector.
        """
        return np.linalg.norm(path(x, y, z, r, phi))
    def out_atten(x, y, z, r, phi):
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
        distance_thru_sample = (y * dist(x, y, z, r, phi)
                                / (detector_coordinates_to_sample_coordinates(r, phi)[1]-y))
        return np.exp(distance_thru_sample*mu_T_Ef)
    def in_atten(y):
        """
        Calculate attenuation experienced by a photon absorbed at
        a given point in phase space (depends only upon depth
        into the sample, measured in the sample coordinate system).
        """
        return np.exp(y/np.sin(theta)*mu_T_E)
    def path(x, y, z, r, phi):
        """
        Returns a vector in sample coordinates that points from a given 
        point in the bulk sample (x, y, z) to a given point on the 
        detector (r, phi).
        """
        sample_point = np.array([x, y, z])
        detector_point = detector_coordinates_to_sample_coordinates(r, phi)
        return detector_point - sample_point
    def cosine_term(x, y, z, r, phi):
        """
        Return the diminution in flux that occurs to off-normal
        incidence.
        """
        path_vector = path(x, y, z, r, phi)
        return np.absolute(np.dot(path_vector, z_d)/np.linalg.norm(path_vector))
    # End big block of helper functions

    theta = c.theta / 180 * np.pi
    
    def normalize_composition(composition) -> list[str, float]:
        """
        Normalize the stoichiometric composition such that the relative
        fractions of all elements sum to one.
        """
        normalized_composition = {}
        stoichimetry_sum = sum(composition.values())
        for element in composition:
            normalized_composition[element] = composition[element]/stoichimetry_sum
        return normalized_composition
    normalized_composition = normalize_composition(c.composition)

    def get_photon_energies_and_quantum_yield(absorbing_element, edge, emission_line):
        """
        Get incident/detected photon energy based on absorbing
        element, absorption edge, and fluorescence line. Also
        get the quantum yield.
        """
        xraydb_edge = xraydb.xray_edge(absorbing_element, edge)
        incident_photon_energy = xraydb_edge.energy
        xraydb_fluor_yield = xraydb.fluor_yield(absorbing_element, edge, emission_line, incident_photon_energy)
        total_quantum_yield, detected_photon_energy, measured_fraction = xraydb_fluor_yield
        quantum_yield = total_quantum_yield * measured_fraction
        return (incident_photon_energy, detected_photon_energy, quantum_yield)
    (
        incident_photon_energy, 
        detected_photon_energy, 
        quantum_yield
    ) = get_photon_energies_and_quantum_yield(c.absorbing_element, c.edge, c.emission_line)
    
    def get_attenuation_coefficients(composition, density, absorbing_element):
        """
        Calculate attenuation coefficients based on sample composition
        and photon energies.
        """
        def get_bulk_attenuation(photon_energy) -> float:
            """
            Attenuation coefficient of photons traversing
            the bulk sample as a function of their energy
            """
            attenuation_coefficient = 0
            for element, fraction in composition.items():
                attenuation_coefficient += (xraydb.mu_elam(element, incident_photon_energy)
                            * density * fraction
                            / 10) # factor of 10 converts from 1/cm to 1/mm
            return attenuation_coefficient
        def get_absorbing_atom_attenuation(photon_energy: float) -> float:
            """
            Absorption coefficient of photons by the
            absorbing element as a function of their energy
            """
            absorbing_element_fraction = composition[absorbing_element]
            attenuation_coefficient = (xraydb.mu_elam(absorbing_element, photon_energy)
                    * density * absorbing_element_fraction
                    / 10)    # factor of 10 converts from 1/cm to 1/mm
            return attenuation_coefficient
        mu_T_E = get_bulk_attenuation(incident_photon_energy)
        mu_T_Ef = get_bulk_attenuation(detected_photon_energy)
        mu_i = get_absorbing_atom_attenuation(incident_photon_energy)
        return (mu_T_E, mu_T_Ef, mu_i)
    (mu_T_E, mu_T_Ef, mu_i) = get_attenuation_coefficients(c.composition, c.density, c.absorbing_element)

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
        x_s = np.array([1, 0, 0])
        y_s = np.array([0, 1, 0])
        z_s = np.array([0, 0, 1])
        return (x_s, y_s, z_s)
    (x_s, y_s, z_s) = define_sample_coordinates()

    def calculate_illuminated_area(detector_above_sample, beam_height, beam_width, photon_flow):
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
            z_illum = beam_height
            x_illum = beam_width/np.sin(theta)
        else:
            z_illum = beam_width
            x_illum = beam_height/np.sin(theta)
        photon_flux = photon_flow/z_illum/x_illum
        return (photon_flux, x_illum, z_illum)
    (
        photon_flux,
        x_illum,
        z_illum
    ) = calculate_illuminated_area(c.detector_above_sample, c.beam_height, c.beam_width, c.photon_flow)

    def lump_constants():
        """
        Lump together a bunch of constants that will just end
        up coming out in front of the integral.
        """
        consts = (photon_flux
                    * quantum_yield
                    * mu_i
                    / (4 * np.pi)
                    / np.sin(theta))
        return consts
    consts = lump_constants()

    def calculate_detector_position(detector_distance, R):
        """
        Get the position of the detector <x, y, z> in the
        sample coordinate scheme.
        """
        x = np.sin(theta)*detector_distance+x_illum/2
        y = np.cos(theta)*detector_distance
        z = z_illum/2
        if y - np.sin(theta)*R <= 0:
            raise ValueError('Detector clips into sample slab!')
        detector_position = np.array([x, y, z])
        return detector_position
    detector_position = calculate_detector_position(c.detector_distance, c.R)

    def calculate_detector_orientation():
        """
        Get alpha and beta, angles which specify the orientation of 
        a unit vector normal to the detector surface in a polar
        coordinate scheme.
        """
        # It's hard-coded that the detector is at a 90 degree angle
        # with respect to the beam.
        beta = 3*np.pi/2 - theta
        alpha = -np.pi/2
        return (alpha, beta)
    (alpha, beta) = calculate_detector_orientation()

    def define_detector_coordinate_system():
        """Define a new orthonormal set of basis vectors
        x_d, y_d, z_d, where z_d is taken as the direction
        normal to the detector surface. These specify
        "detector coordinates"."""
        z_d = np.array([np.sin(alpha)*np.cos(beta), 
                                np.sin(alpha)*np.sin(beta), 
                                np.cos(alpha)])
        x_d = np.cross(y_s, z_d)
        x_d /= np.linalg.norm(x_d)
        y_d = np.cross(z_d, x_d)
        y_d /= np.linalg.norm(y_d)
        return (x_d, y_d, z_d)
    (x_d, y_d, z_d) = define_detector_coordinate_system()

    def calculate_transformation_matrices():
        """
        Calculate a matrix M that converts from detector coorinates
        to sample coordinates and a matrix W that converts from
        sample coordinates to detector coordinates.
        """
        M = np.array([x_d, y_d, z_d]).T
        W = np.linalg.inv(M)
        return (M, W)
    (M, W) = calculate_transformation_matrices()

    def find_y_depth_limit():
        """
        Calculate how deeply into the sample slab we will integrate
        by finding at what point the beam intensity has dropped below
        1%.
        """
        depth = 0
        while True:
            if (in_atten(depth) < 0.01):
                y_depth_limit = depth
                break
            depth -= 1E-5
        return y_depth_limit
    y_depth_limit = find_y_depth_limit()

    def compute_full_integral_count_rate(R):
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
            return (consts * r
                    * cosine_term(x, y, z, r, phi)
                    / (dist(x, y, z, r, phi)**2)
                    * out_atten(x,y,z,r,phi)
                    * in_atten(y))
        def sampler():
            """Sample phase space for the Monte Carlo integration."""    
            while True:
                z = random.uniform(0, z_illum)
                y = random.uniform(y_depth_limit, 0)
                x = random.uniform(-y/np.tan(theta), -y/np.tan(theta) + x_illum)
                r = random.uniform(0, R)
                phi = random.uniform(0, 2*np.pi)
                yield (x, y, z, r, phi)
        random.seed(1)
        domainsize = -z_illum*x_illum*y_depth_limit*R*2*np.pi
        count_rate, montecarlo_error = mcint.integrate(integrand, sampler(), measure = domainsize, n=nmc)
        return (count_rate, montecarlo_error)
    (count_rate, montecarlo_error) = compute_full_integral_count_rate(c.R)

    def compute_simple_count_rate(R, detector_distance, photon_flow):
        """
        Compute the count rate according to the approximate description
        given in Bunker.
        """
        f = mu_T_E/np.sin(theta) + mu_T_Ef/np.sin(np.pi/2 - theta)
        apex_angle = np.arctan(R/detector_distance)
        solid_angle = 4*np.pi*np.sin(apex_angle/2)**2

        g = quantum_yield*solid_angle/4/np.pi/f
        crude_result = g*mu_i*photon_flow/np.sin(theta)
        return (crude_result, solid_angle)
    (
        crude_result,
        solid_angle
    ) = compute_simple_count_rate(c.R, c.detector_distance, c.photon_flow)

    def print_output():
        print(f"expected count rate: {count_rate:.3e} photons per second")
        print("Using n = ", nmc)
        print(f"estimated error = {montecarlo_error:.2e} = {montecarlo_error/count_rate*100:.3}%")
        print(f"solid angle fraction: {solid_angle/np.pi*4:.4f}")
        print(f"crude count rate: {crude_result:.3e} photons per second, difference of {np.absolute(100*(crude_result - count_rate)/count_rate):.2f}%")
    if not suppress_output:
        print_output()

    def package_result() -> Calculation_Result:
        result = Calculation_Result(
            c,
            nmc,
            count_rate,
            montecarlo_error,
            crude_result,
            solid_angle
        )
        return result
    result = package_result()

    return result

class angle_sweep:
    def __init__(self, min_angle, max_angle, step_size, template_experiment, nmc = 100):

        def generate_configurations():
            angles = np.arange(min_angle, max_angle, step_size)
            configurations = []
            for angle in angles:
                configuration = replace(template_experiment, theta = angle)
                configurations.append(configuration)
            return configurations
        configurations = generate_configurations()

        def calculate_results():
            results = []
            for configuration in configurations:
                results.append(xafs_count_rate(
                    configuration, nmc = nmc, suppress_output=True))
            return results
        self.results = calculate_results()
    
    @property
    def count_rates(self):
        return np.array([result.count_rate for result in self.results])
    @property
    def crude_count_rates(self):
        return np.array([result.crude_count_rate for result in self.results])
    @property
    def angles(self):
        return np.array([result.configuration.theta for result in self.results])
    @property
    def max_count_rate(self):
        return np.max(self.count_rates)
    @property
    def optimal_angle(self):
        max_index = np.argmax(self.count_rates)
        return self.angles[max_index]

    def plot_result(self, **kwargs):
        plt.plot(self.angles, self.count_rates, **kwargs)
    def plot_crude_result(self, **kwargs):
        plt.plot(self.angles, self.crude_count_rates, **kwargs)
    def report_optimal_angle(self):
        print(f"Maximum flux of {self.max_count_rate:.3e} at angle {self.optimal_angle}")