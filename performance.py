#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:42:57 2026

@author: antoinedlbck
"""

from constants import G, RHO_AIR, KMH_TO_MS, RPM_TO_RADS, MS_TO_KMH
import numpy as np
from resistive_forces import drag_force

def top_speed(vehicle, motor, power="peak", v0=0, tol=1e-2, MaxIter=1000):
    """
    Computes the top speed of the vehicle (km/h).
    Returns theoretical top speed (v_maxmax), 
            actual top speed (v_max),
            what the limitation is for v_max (power or redline)
    (meaning if limitation is "power", max speed v_max = v_maxmax,
     if limitation is "redline", max speed is v_max != v_maxmax)
    """
    
    m = vehicle.total_mass
    f0 = vehicle.tire_rr_f0
    f2 = vehicle.tire_rr_f2 / (KMH_TO_MS ** 2) # f2 is in (km/h)^-2 in the class
    C_x = vehicle.C_x
    S = vehicle.frontal_area
    eta_tr = vehicle.transmission_eff
    R_er = vehicle.tire_effective_radius_rear

    if power == "peak":
            P_1 = motor.power_peak
    elif power == "nominal":
            P_1 = motor.power_nominal
    else:
        raise ValueError(f"power must be 'peak' or 'nominal', got '{power}'")
    
    
    A = m * G * f0
    B = m * G * f2 + 0.5 * RHO_AIR * S * C_x
    
    v = [v0]
    
    for k in range(MaxIter - 1):
        
        v_new = np.cbrt( (eta_tr * P_1 - A * v[-1])/B )
        v.append(v_new)
        
        if np.abs(v[k+1] - v[k]) < tol:
            break
        
    else:
        raise RuntimeError(f"No convergence after {MaxIter} iterations for tol = {tol}. Try increasing MaxIter or tol.")
    
    # v_max^max
    v_power = v[-1]
    
    # v_max (75): speed limited by max_rpm
    v_redline = (motor.max_rpm * RPM_TO_RADS * R_er / vehicle.gear_ratio)
    
    if v_power <= v_redline:
        return v_power * MS_TO_KMH, v_redline * MS_TO_KMH, "power"
    
    else:
        return v_power * MS_TO_KMH, v_redline * MS_TO_KMH, "redline"
    
    
def max_slope(vehicle, motor, power="peak", starting_speed=0.0, mu=1.0):
    
    # à check: gestion des unités pour V dans f_RR
    
    b = vehicle.b
    L = vehicle.wheelbase
    h = vehicle.height_CG
    m = vehicle.total_mass
    
    if starting_speed == 0:
        
        f = vehicle.tire_rr_f0 # low speeds -> f2 negligible
        
        ratio = vehicle.tractive_force(motor, rpm=1e-6) / (m * G)
        tractive_force = np.asin(ratio) if abs(ratio) <= 1 else np.pi/2
        
        non_slip = np.atan2(mu * b/L - f, 1 - mu * h/L)

    else:
        
        f0 = vehicle.tire_rr_f0 
        f2 = vehicle.tire_rr_f2
        f = f0 + f2 * (starting_speed ** 2)
        
        # Tractive forces
        F_drag = drag_force(vehicle, starting_speed)
        F_w = vehicle.tractive_force(motor, rpm=0)
        
        d = (F_w - F_drag)/(m * G)
        
        ratio = (d - f * np.sqrt(1 + f ** 2 - d ** 2))/(1 + f ** 2)
        tractive_force = np.asin(ratio) if abs(ratio) <= 1 else np.pi/2
        
        # Non slip
        A = m * G * (1 - mu * h/L)
        B = m * G * (f - mu * b/L)
        C = (mu * h/L - 1) * F_drag
        denom = np.sqrt(A ** 2 + B ** 2) 
        
        non_slip = np.asin(C/denom) - np.asin(A/denom)
        
        
    theta_max = np.minimum(tractive_force, non_slip)
    
    return theta_max