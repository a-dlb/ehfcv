#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:20:05 2026

@author: claravliegen
"""

import numpy as np
import matplotlib.pyplot as plt
from vehicle import Vehicle
from plots import plot_resistive_force, plot_resistive_power
from plot_style import get_style
from resistive_forces import total_force, total_power, drag_force, grading_force, rolling_res_force

PLOT = False
EXPORT = False

def crossing(V_kmh, f1, f2):
    """First speed (km/h) where f1 surpasses f2, None if not in the span."""
    d = f1 - f2
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    # linear interpolation between the two points
    return V_kmh[i] - d[i] * (V_kmh[i+1] - V_kmh[i]) / (d[i+1] - d[i])
    
def main():
    
    vehicle = Vehicle()
    
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    slopes = [0, 5, 10] # % 
    
    print(f"===== Results (V_max = {vehicle.max_speed} km/h) =====")
    
    for s in slopes:
        
        F_aero  = drag_force(vehicle, speed)
        F_roll  = rolling_res_force(vehicle, speed, s)
        F_grade = grading_force(vehicle, s) # returns a scalar
        F_grade = np.full_like(speed, F_grade)
        F_tot   = total_force(vehicle, speed, s)
        P_tot   = total_power(vehicle, speed, s)
    
        V1 = crossing(speed, F_aero, F_roll)
        V2 = crossing(speed, F_aero, F_grade) if s > 0 else None
    
        print(f"\n--- Slope {s*100:.0f} % ---")
        print(f"Rolling resistance       : {F_roll[0]:.1f} N (V=0) -> {F_roll[-1]:.1f} N (V_max)")
        print(f"Grading resistance       : {F_grade[0]:.1f} N")
        print(f"Aero drag at V_max       : {F_aero[-1]:.1f} N")
        print(f"Total resistance at V_max: {F_tot[-1]:.1f} N")
        print(f"Road power at V_max      : {P_tot[-1]/1000:.1f} kW")
        print(f"V1 (drag = rolling)      : {V1:.1f} km/h" if V1 is not None
              else "V1 (drag = rolling)      : out of range")
        if s > 0:
            print(f"V2 (drag = grading)      : {V2:.1f} km/h" if V2 is not None
                  else "V2 (drag = grading)      : out of range")
    
    if PLOT:
        
        with plt.rc_context(get_style()):
        
            plot_resistive_force(vehicle, slopes)
            plot_resistive_power(vehicle, slopes)
    
    return
    

if __name__ == "__main__":
    
    main()