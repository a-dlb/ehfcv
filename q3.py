#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:10:34 2026

@author: antoinedlbck
"""

import numpy as np
import matplotlib.pyplot as plt
from motor import Motor
from vehicle import Vehicle
from plot_style import get_style
from plots import plot_torque_power, plot_traction

PLOT = True
EXPORT = False

def main():
    
    motor = Motor()
    car = Vehicle()
    
    # Overcharging factor
    print("Overcharging factor: ")
    print(f"  k_oc = {motor.overcharging_factor:.3f} (-)")
    print("\n")
    
    # Base speed
    print("Base_speed: ")
    print(f"  rpm_base = {motor.base_speed_rpm:.0f} (rpm)")
    print("\n")

    if PLOT:
        
        # rc_context allows to load style without keeping it for every plot
        # launched from the console 
        with plt.rc_context(get_style()):
            
            rpm = np.linspace(1e-6, motor.max_rpm, 1000)
            
            # Power/torque curves
            for mode in ["peak", "nominal"]:
                
                path = f"Graphs/{mode}_pc.pdf" if EXPORT else None
                plot_torque_power(rpm, motor.torque(rpm), motor.power(rpm),
                                  motor.overcharging_factor, highlight=mode, export_path=path)
                
            # Tractive force curve
            path = "Graphs/traction.pdf" if EXPORT else None
            plot_traction(rpm, car.tractive_force(motor, rpm), export_path=path)
        

if __name__ == "__main__":
    
    main()