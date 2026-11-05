#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:20:49 2026

@author: antoinedlbck
"""

from vehicle import Vehicle
from motor import Motor
from performance import top_speed, max_slope, accel_time
from plots import plot_topspeed, plot_max_slopes
from constants import KMH_TO_MS, RADS_TO_RPM
from resistive_forces import total_power
from plot_style import get_style
from friction_coeff import MU
import matplotlib.pyplot as plt

import numpy as np

PLOT = True
EXPORT = False

# Helper to convert rads to % !
def rad2per(theta):
    return np.tan(theta) * 100

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
        
    # Max slope 
    # vérifier le développement de perfo + ajouter le régime nominal (voir TPs en fait)
    # (et réfléchir à cmt implémenter ça dans le rapport)
    print_section("Max slope")
    
    powers = ["peak", "nom"]
    speeds = [0, 80] # km/h
    
    # Store data for plotting
    data = {s: ({}, []) for s in speeds}
    
    for p in powers:
        for s in speeds:
            print(f"From {s} km/h (Power = {p})\n")
    
            theta_max_traction = max_slope(car, motor, power=p, crit="traction", starting_speed=s)
            print(f"  traction max slope (%) : {rad2per(theta_max_traction):.1f} \n")
            
            # Store data for plotting
            data[s][0][p] = rad2per(theta_max_traction)
    
            print("  adhesion max slopes (%) + conclusion (limiting_factor)")
            for name, values in MU.items():
    
                thetas_max_adhesion = [max_slope(car, motor, power="peak", crit="adhesion", starting_speed=s, mu=mu)
                                       for mu in values]
                
                # Store data for plotting
                if p == powers[0]:
                    data[s][1].append((name, list(values), [rad2per(t) for t in thetas_max_adhesion]))
    
                print(f"    {name}: " + " - ".join(
                    f"{rad2per(t):.1f}% ({'adhesion' if t < theta_max_traction else 'traction'})"
                    for t in thetas_max_adhesion))
            print("\n")
    
    
    # Acceleration times
    
    print_section("Acceleration times")
    
    for p in powers:
        
        print(f"Acceleration times at {p} power")
        
        dT1 = accel_time(car, motor, [80, 120], power=p)
        print(f"  From 80 to 120 km/h : ∆t = {dT1:.1f} s \n")
        
        dT2 = accel_time(car, motor, [1e-6, 100], power=p)
        print(f"  From 0 to 100 km/h : ∆t = {dT2:.1f} s \n")
        
        dT3 = accel_time(car, motor, [1e-6, 400], interval_type="distance", power=p)
        print(f"  To cover 400m from rest : ∆t = {dT3:.1f} s " + "\n" * 2)
        
        # 1e-6 to avoid division by 0 


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
                
                vplot = v_peak if regime == "peak" else v_nom
                
                plot_topspeed(speed, eta_tr, P_max, wheel_power, resistive_power, under_redline, k_oc, vplot, highlight=regime, export_path=path)
    
    
        with plt.rc_context(get_style(style="maxslope")):
            
            plot_max_slopes(data, powers, export_path="Graphs/q5_slopes.png")
        
    return


if __name__ == "__main__":
    
    main()