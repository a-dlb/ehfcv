#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:20:05 2026

@author: claravliegen
"""

import matplotlib.pyplot as plt
from vehicle import Vehicle
from plots import plot_resistive_force, plot_resistive_power
from plot_style import get_style

PLOT = True
EXPORT = False
    
def main():
    
    vehicle = Vehicle()
    
    slopes = [0, 5, 10] # % 
    
    if PLOT:
        
        with plt.rc_context(get_style()):
        
            plot_resistive_force(vehicle, slopes)
            plot_resistive_power(vehicle, slopes)
    
    return
    

if __name__ == "__main__":
    
    main()