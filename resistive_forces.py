#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:17:12 2026

@author: antoinedlbck
"""

from constants import G, KMH_TO_MS, MS_TO_KMH, RHO_AIR
import numpy as np

def rolling_res_coef(f0, f2, speed, speed_in_ms=False):
    """ 
    Rolling resistance coefficient f_RR (-)
    /!\ speed is expected in km/h
    Specify speed_in_ms = True otherwise
    """
    
    if speed_in_ms:
        speed = speed * MS_TO_KMH
    
    return f0 + f2 * (speed ** 2)


def rolling_res_force(vehicle, speed, slope, speed_in_ms=False):
    """
    Rolling resistance force F_rr (N)
    /!\ speed is expected in (km/h) (and slope in (%))
    Specify speed_in_ms = True otherwise.
    """
        
    slope = np.arctan(slope * 1e-2)
    
    f0 = vehicle.tire_rr_f0
    f2 = vehicle.tire_rr_f2
    
    f_RR = rolling_res_coef(f0, f2, speed, speed_in_ms)
    
    m = vehicle.total_mass
    
    return m * G * f_RR * np.cos(slope)


def drag_force(vehicle, speed, speed_in_ms=False):
    """
    Drag force F_D (N)
    /!\ speed is expected in (km/h)
    Specify speed_in_ms = True otherwise
    """
    
    if not speed_in_ms:
        speed = speed * KMH_TO_MS
    
    S = vehicle.frontal_area
    
    return 1/2 * RHO_AIR * vehicle.C_x * S * (speed ** 2)
    

def grading_force(vehicle, slope):
    """
    Grading force F_grad (N)
    /!\ slope is expected in (%)
    """
    
    slope = np.arctan(slope * 1e-2)
        
    m = vehicle.total_mass
    
    return m * G * np.sin(slope)


def total_force(vehicle, speed, slope, speed_in_ms=False):
    """
    Total resistive force F_tot (N)
    /!\ speed is expected in (km/h) (and slope in (%))
    """
    
    F_rr = rolling_res_force(vehicle, speed, slope, speed_in_ms)
    F_drag = drag_force(vehicle, speed, speed_in_ms)
    F_grad = grading_force(vehicle, slope)
    
    return F_rr + F_drag + F_grad


def total_power(vehicle, speed, slope, speed_in_ms=False):
    """
    Total resistive power P_tot (W)
    /!\ speed is expected in (km/h) (and slope in (%))
    """
    
    F_tot = total_force(vehicle, speed, slope, speed_in_ms)
    
    if not speed_in_ms:
        speed = speed * KMH_TO_MS
    
    return F_tot * speed
    