#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:20:49 2026

@author: antoinedlbck
"""

from vehicle import Vehicle
from motor import Motor
from performance import top_speed
from plots import plot_topspeed
from constants import KMH_TO_MS, RADS_TO_RPM
from resistive_forces import total_power
from plot_style import get_style
import matplotlib.pyplot as plt


import numpy as np

PLOT = True
EXPORT = False

# Simple helper to make the prints clearer :) 
def print_section(title):
    
    print("\n" + "=" * 30)
    print(f"{title.upper()}") #.upper() = change the string into uppercase
    print("=" * 30)
    
    return 

def main():
    
    car = Vehicle()
    motor = Motor()
    eta_tr = car.transmission_eff
    R_er = car.tire_effective_radius_rear
    gear_ratio = car.gear_ratio
    k_oc = motor.overcharging_factor
    
    # Top speed
    v_power_peak, v_redline_peak, limitation_peak = top_speed(car, motor, power="peak")
    v_peak = v_power_peak if limitation_peak == "power" else v_redline_peak
    
    v_power_nom, v_redline_nom, limitation_nom = top_speed(car, motor, power="nominal")
    v_nom = v_power_nom if limitation_nom == "power" else v_redline_nom
    
    print_section("Top speed")
    
    print(f"Top speed at max. power, limited by {limitation_peak}")
    print(f"  v_max = {v_peak:.1f} (km/h)")
    if limitation_peak == "redline":
        print(f"  (theoretical greatest speed: v_maxmax = {v_power_peak:.1f} (km/h)")
        
    print("\n")
    
    print(f"Top speed at nominal power, limited by {limitation_nom}")
    print(f"  v_max = {v_nom:.1f} (km/h)")
    if limitation_nom == "redline":
        print(f"  (theoretical greatest speed: v_maxmax = {v_power_nom:.1f} (km/h)")
        
    
    # Plot top speed
    
    if PLOT:
    
        regimes = ["peak", "nominal"]
        
        # vérifier si je peux pas faire ça + joli et rajouter le tracé de vmax etc"""
        with plt.rc_context(get_style()):
            
            for regime in regimes:
            
                path = f"Graphs/{regime}_topspeed.pdf" if EXPORT else None
                
                speed = np.linspace(1e-6, 250, 1000) # km/h
                
                P_max = motor.power_peak # W 
                
                rpm = ((speed * KMH_TO_MS) * gear_ratio / R_er) * RADS_TO_RPM # rpm
                
                under_redline = rpm <= motor.max_rpm
                
                wheel_power = car.tractive_power(motor, rpm[under_redline])
                
                resistive_power = total_power(car, speed, 0) # slope = 0%
                
                plot_topspeed(speed, eta_tr, P_max, wheel_power, resistive_power, under_redline, k_oc, highlight=regime, export_path=path)
    
    return


if __name__ == "__main__":
    
    main()