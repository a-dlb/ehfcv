#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:32:41 2026

@author: antoinedlbck
"""

from constants import G, INCH_TO_M, GEAR_PAIR_EFF, RPM_TO_RADS
from dataclasses import dataclass

# Dataclass for the vehicle, with frozen=True so it can't be changed
# after being declared

@dataclass(frozen=True)
class Vehicle:
    
    vehicle_mass: float = 1195                   # kg
    driver_mass: float = 75                      # kg
    fr_ratio: tuple = (0.47, 0.53)               # - (front, rear) (%)
    tire_width: float = 155e-3                   # m
    tire_aspect_ratio: float = 0.70              # -
    tire_stiffness: float = 160e3                # N/m
    tire_rimsize: float = 19                     # inches
    transmission_eff: float = GEAR_PAIR_EFF ** 2 # -
    gear_ratio: float = 9.665                    # - 
    tire_rr_f0: float = 0.008                    # - 
    tire_rr_f2: float = 0.4e-7                   # (km/h)^-2 
    C_x: float = 0.29                            # -
    frontal_area: float = 2.38                   # m^2 
    
    @property
    def total_mass(self):
        """ Total mass (kg) """
        return self.vehicle_mass + self.driver_mass
    
    @property
    def vertical_force_front(self):
        """ Vertical force on a front tire (N) """
        return self.fr_ratio[0] * self.total_mass * G / 2 
    
    @property
    def vertical_force_rear(self):
        """ Vertical force on a rear tire (N) """
        return self.fr_ratio[1] * self.total_mass * G / 2 
    
    @property
    def tire_unloaded_radius(self):
        """ Unloaded tire radius (m) """
        return 1/2 * (self.tire_rimsize * INCH_TO_M 
                      + 2 * self.tire_width * self.tire_aspect_ratio)
    
    @property
    def tire_effective_radius_front(self):
        """Effective front tire radius (m) """
        
        R_u = self.tire_unloaded_radius
        F_zf = self.vertical_force_front
        k_z = self.tire_stiffness
        
        return R_u - F_zf/(3 * k_z)
    
    @property
    def tire_effective_radius_rear(self):
        """Effective rear tire radius (m) """        
        R_u = self.tire_unloaded_radius
        F_zr = self.vertical_force_rear
        k_z = self.tire_stiffness
        
        return R_u - F_zr/(3 * k_z)
    
    def tractive_force(self, motor, rpm):
        """
        Takes an array of rpm and returns an array of the corresponding tractive force values
        (! PEAK values -> divide by overcharging factor to get nominal)
        
        In:
          rpm (np.array) : motor rotation speed in rpm (< max_rpm)
          motor (class Motor): specific motor to pass to the method
        
        Out:
              (np.array) : tractive force in N 
        """
        
        C_mot = motor.torque(rpm)
        
        return (self.transmission_eff * C_mot 
                * self.gear_ratio / self.tire_effective_radius_rear)
    
    def tractive_power(self, motor, rpm):
        """
        Takes an array of rpm and returns an array of the corresponding tractive power values
        (! PEAK values -> divide by overcharging factor to get nominal)
        
        In:
          rpm (np.array) : motor rotation speed in rpm (< max_rpm)
          motor (class Motor): specific motor to pass to the method
        
        Out:
              (np.array) : tractive power in W
        """    
        
        F_t = self.tractive_force(motor, rpm)
        R_er = self.tire_effective_radius_rear
        gear_ratio = self.gear_ratio
        omega = rpm * RPM_TO_RADS
        
        return F_t * omega * R_er / gear_ratio