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
        #we define here coordinates for the sample
        self.x_s = np.array([1, 0, 0]) # x is a unit vector on the sample's surface that defines the dimension of illuminated area which shrinks or elongates as the beam's angle of incidence is varied
        self.y_s = np.array([0, 1, 0]) # y is the direction normal to the sample surface
        self.z_s = np.array([0, 0, 1]) # z is a unit vector on the sample's surface that defines the dimension of illuminated area which does not change as the beam's angle of incidence is varied
        
        self.suppress_output = suppress_output
        
        self.theta = theta*np.pi/180
        
        self.R = R
        self.d = 2*self.R
        self.detector_distance = detector_distance
        
        self.beam_width = beam_width
        self.beam_height = beam_height
        self.detector_above_sample = detector_above_sample
        #if the detector is situated above or below the sample (i.e. towards the ceiling or floor) then the beam's height will be spread out as theta changes
        if (detector_above_sample):
            self.z_illum = beam_width #in mm
            self.x_illum = beam_height/np.sin(self.theta) #in mm
        #if the detector is mounted in the same horizontal plane as the sample (as is typical), then the beam's width will be spread out as theta changes
        else:
            self.z_illum = beam_height #in mm
            self.x_illum = beam_width/np.sin(self.theta) #in mm
            

        self.photon_flow = photon_flow #photons/second
        self.photon_flux = self.photon_flow/self.z_illum/self.x_illum #photons/s/mm^2

        self.quantum_yield = quantum_yield
        
        
        #DETECTOR POSITION (in sample coordinates) CHANGES AS YOU CHANGE THETA (if the detector is fixed in labratory coordinates)
        self.x_d = np.sin(self.theta)*detector_distance+self.x_illum/2 #mm
        self.y_d = np.cos(self.theta)*detector_distance #mm
        self.z_d = self.z_illum/2 #mm
        
        #check to make sure the detector is not clipping into the sample
        if self.y_d - np.sin(self.theta)*self.R <= 0:
            raise ValueError('Detector clips into sample slab!')
        
        self.p_d = np.array([self.x_d, self.y_d, self.z_d])

        #angle at which detector is tilted
        #from a beamline user's point of view, the detector and beam are static, and the sample's position and orientation are adjusted.
        #in our calculations, we consider the sample as being static and the beam/detector as being adjusted.
        #the detector should typically be 90° from the beam. Since our picture is of a static sample and a moving beam, the detector's position and orientation must also change if the beam's angle of incidence changes
        
        self.beta = 3*np.pi/2 - self.theta
        self.alpha = -np.pi/2

        #alpha and beta define a unit vector z_2 perpendicular to the surface of the detector, such that
        self.z_2 = np.array([np.sin(self.alpha)*np.cos(self.beta), np.sin(self.alpha)*np.sin(self.beta), np.cos(self.alpha)])

        #basically, we're looking at the two angular components of a spherical coordinate system - if the xy plane is the ground and the z direction is up, alpha is the zenith angle and beta is the azimuthal angle
        
        #we can use cross products to define the two other coordinate axes which are guaranteed to be in the plane of the detector
        self.x_2 = np.cross(self.y_s, self.z_2)/np.linalg.norm(np.cross(self.y_s, self.z_2))
        self.y_2 = np.cross(self.z_2, self.x_2)/np.linalg.norm(np.cross(self.z_2, self.x_2))

        #our matrix that can convert points from detector coordinates into sample coordinates is thus
        self.M = np.array([self.x_2, self.y_2, self.z_2]).T

        #and our matrix that converts from regular coordinates into detector coordinates is this
        self.W = np.linalg.inv(self.M)
        
        #attenuation of fluoresced photons through the bulk material
        self.mu_T_Ef = mu_T_Ef #reciprocal millimeters

        #attenuation of incoming photons through the bulk material
        self.mu_T_E = mu_T_E #reciprocal millimeters

        #absorption coefficient of the absorbing atom of interest
        self.mu_i = mu_i #reciprocal millimeters

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
        return np.matmul(self.M, self.polar_to_cartesian(r, phi))+self.p_d
        
        
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
        return np.absolute(np.dot(path_vector, self.z_2)/np.linalg.norm(path_vector))

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
        
