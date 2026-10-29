#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:42:57 2026

@author: antoinedlbck
"""

from constants import G, RHO_AIR, KMH_TO_MS, RPM_TO_RADS, MS_TO_KMH
import numpy as np

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