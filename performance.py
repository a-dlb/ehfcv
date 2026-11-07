#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:42:57 2026

@author: antoinedlbck
"""

from constants import G, RHO_AIR, KMH_TO_MS, RPM_TO_RADS, MS_TO_KMH
import numpy as np
from resistive_forces import drag_force, net_force
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_interp_spline


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
    elif power == "nom":
            P_1 = motor.power_nominal
    else:
        raise ValueError(f"power must be 'peak' or 'nom', got '{power}'")
    
    
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
    
    
def max_slope(vehicle, motor, power="peak", crit="friction", starting_speed=0.0, mu=1.0, speed_units="kmh"):
    """
    Computes the max slope the vehicle can overcome for a given speed and mu (rad)
    
     crit: "friction" (default) or "adhesion"
     power: "peak" (default) or "nom"
     speed_units: "kmh" (default) or "ms"
    """
    
    b = vehicle.b
    L = vehicle.wheelbase
    h = vehicle.height_CG
    m = vehicle.total_mass
    k_oc = motor.overcharging_factor
    
    F_w = vehicle.tractive_force(motor, rpm=1e-6)
    
    if power == "nom":
        F_w /= k_oc
    
    if starting_speed == 0:

        if crit == "traction":
        
            ratio = F_w / (m * G)
            theta_max_traction = np.asin(min(ratio, 1.0)) 
            # If ratio > 1 : max slope is 90° ; tractive force not limiting
            # -> set ratio to 1 to avoid asin error
            
            return theta_max_traction
        
        elif crit == "adhesion":
            
            f = vehicle.tire_rr_f0 # low speeds -> f2 negligible
            theta_max_adhesion = np.atan2(mu * b/L - f, 1 - mu * h/L)
            
            return theta_max_adhesion
        
        else:
            raise ValueError("crit must be 'adhesion' or 'tractio', not {crit}")

    else:
        
        if speed_units == "ms":
            starting_speed = starting_speed * MS_TO_KMH
            
        f0 = vehicle.tire_rr_f0 
        f2 = vehicle.tire_rr_f2
        f = f0 + f2 * (starting_speed ** 2)
        
        F_drag = drag_force(vehicle, starting_speed)
        
        if crit == "traction":
            
            # Tractive forces
            d = (F_w - F_drag)/(m * G)
            
            ratio = (d - f * np.sqrt(1 + f ** 2 - d ** 2))/(1 + f ** 2)
            theta_max_traction = np.asin(min(ratio, 1.0))
            
            return theta_max_traction
        
        elif crit == "adhesion":
            
            # Non slip
            A = m * G * (f - mu * b/L)
            B = m * G * (1 - mu * h/L)
            C = (mu * h/L - 1) * F_drag
            denom = np.sqrt(A ** 2 + B ** 2) 
            
            theta_max_adhesion = np.asin(C/denom) - np.asin(A/denom)
            
            return theta_max_adhesion
        
        else:
            raise ValueError("crit must be 'adhesion' or 'traction', not '{crit}'")
            

def accel_time(vehicle, motor, interval, interval_type="speed", power="peak"):
    
    if interval_type not in ("speed", "distance"):
        raise ValueError(f"interval_type must be 'speed' or 'distance', not '{interval_type}'")
        
    if power not in ("peak", "nom"):
        raise ValueError(f"power must be 'peak' or 'nom', not '{power}'")

    v_min, target = np.asarray(interval, dtype=float)
    m_et = vehicle.total_effective_mass(motor)
    v_cap = vehicle.max_speed # km/h

    if interval_type == "speed":
        
        if target > v_cap:
            raise ValueError(f"Target speed {target} km/h exceeds max_speed {v_cap} km/h")
            
        v_max = target
        
    else:
        v_max = v_cap

    speed = np.linspace(v_min, v_max, 1000) # km/h
    speed_ms = speed * KMH_TO_MS            # m/s

    F_net = net_force(vehicle, motor, speed, slope=0, power=power)
    
    # only problematic in case of a high slope a nom power probably
    if np.any(F_net <= 0):
        raise ValueError("F_net <= 0 in the interval: speed not reachable")
        
    inv_F = 1 / F_net

    if interval_type == "speed":
        return m_et * np.trapezoid(inv_F, speed_ms)

    # Distance mode
    
    # ∆t = t(x)
    dt = m_et * cumulative_trapezoid(inv_F, speed_ms, initial=0)
    
    # ∆x = x(v)
    dx = m_et * cumulative_trapezoid(speed_ms * inv_F, speed_ms, initial=0)

    # dx[-1] = distance reached when v_max is reached
    # -> if target_∆x <= dx[-1],
    # ∆t = t(x(v)) obtained with np.interp
    
    if target <= dx[-1]:
        return np.interp(target, dx, dt)

    # -> if target_∆x > dx[-1]
    # ∆t = dt[-1] + (target_∆x - dx[-1]) / (v_max in m_s)
    #   where dt[-1] = time to reach v_max 
    #         (target_∆x - dx[-1]) = distance left to cover after that
    #         (target_∆x - dx[-1]) / (v_max in m_s) = time to cover remaining distance
    #                                                 at v_max
    
    else:
        return dt[-1] + (target - dx[-1]) / speed_ms[-1]
    