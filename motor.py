#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:59:44 2026

@author: antoinedlbck
"""

import numpy as np
from dataclasses import dataclass
from constants import RPM_TO_RADS

@dataclass(frozen=True)
class Motor:
    
    power_peak: float = 125e3           # W
    power_nominal: float = 75e3         # W
    torque_peak: float = 250            # Nm
    max_rpm: float = 11400              # rpm
    rotor_mass: float = 14.2            # kg
    rotor_rad_ext: float = 178.6e-3/2   # m 
    rotor_rad_int: float = 130e-3/2     # m
    rotor_inertia: float = 69138e-6     # kg.m^2 
    
    @property
    def overcharging_factor(self):
        """ Overcharging factor (peak_power / nominal_power) (-) """
        return self.power_peak/self.power_nominal
    
    @property
    def base_speed_rad(self):
        """ Base speed (rad/s) """
        return self.power_peak / self.torque_peak
    
    @property
    def base_speed_rpm(self):
        """ Base speed (rpm) """
        return self.base_speed_rad * 60 / (2 * np.pi)
    
    @property
    def rotor_inertia_cylinder(self):
        """ Inertia of the rotor if it was a full cylinder (kg.m^2) """
        return 1/2 * self.rotor_mass * (self.rotor_rad_ext ** 2)
    
    @property
    def rotor_inertia_hollow(self):
        """ Inertia of the rotor if it was a hollow cylinder (kg.m^2) """
        return 1/2 * self.rotor_mass * (self.rotor_rad_ext ** 2 +  self.rotor_rad_int ** 2)
    
    def _check_rpm(self, rpm):
        if np.any(rpm > self.max_rpm):
            raise ValueError(f"rpm can't exceed max_rpm = {self.max_rpm} rpm")
    
    def torque(self, rpm):
        """ 
        Takes an array of rpm and returns an array of the corresponding torque values
        (! PEAK values -> divide by overcharging_factor to get nominal)
        In:
          rpm (np.array) : motor rotation speed in rpm (< rpm_max)
    
        Out:
              (np.array) : motor torque in Nm
        """
        
        self._check_rpm(rpm)
        
        rpm_base = self.base_speed_rpm
        torque_peak = self.torque_peak
        power_peak = self.power_peak
        
        omega = rpm * RPM_TO_RADS
            
        return np.where(rpm <= rpm_base, torque_peak, power_peak/omega)
    
    def power(self, rpm):
        """ 
        Takes an array of rpm and returns an array of the corresponding power values
        (! PEAK values -> divide by overcharging_factor to get nominal)
        In:
          rpm (np.array) : motor rotation speed in rpm (< rpm_max)
    
        Out:
              (np.array) : motor power in W
        """
        
        self._check_rpm(rpm)
        
        rpm_base = self.base_speed_rpm
        torque_peak = self.torque_peak
        power_peak = self.power_peak
        
        omega = rpm * RPM_TO_RADS
            
        return np.where(rpm <= rpm_base, (torque_peak * omega), power_peak)