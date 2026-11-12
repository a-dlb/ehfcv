#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:38:47 2026

@author: claravliegen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vehicle import Vehicle
from motor import Motor
from constants import (BATTERY_CAPACITY, P_ACCESSORIES,
                       P_ACCESSORIES_WINTER, P_ACCESSORIES_SUMMER)

from consumption import (motor_efficiency, motor_operating_point, battery_power_constant_speed, 
                         consumption_constant_speed, load_wltp, battery_power_cycle, cycle_energy)
from plot_style import get_style
from plots import plot_consumption, plot_cycle, plot_cycle_alpha, plot_consumption_weather, plot_efficiency_map

PLOT = True
EXPORT = False
WLTP_FILE = "WLTP.xlsx" # Excel 

#-------------------------------- affichage titre -------------------------------------
def print_section(title):
    print("\n" + "=" * 30)
    print(f"{title.upper()}") 
    print("=" * 30)
    return 

#-------------------------------- affichage valeurs -------------------------------------
def main():
    car = Vehicle()
    motor = Motor()
    
    # a. Energy consumption at constant speed
    print_section("Consumption at constant speed")
    speeds = np.array([30, 50, 90, 120])           # km/h
    rpm, C_m = motor_operating_point(car, speeds)
    eta_m = motor_efficiency(motor, rpm, C_m) 
    P_bat = battery_power_constant_speed(car, motor, speeds)
    E = consumption_constant_speed(car, motor, speeds)
    
    # Efficiency map (with and without generator side)
    if PLOT:
        path = "Graphs/efficiency_map.pdf" if EXPORT else None
        with plt.rc_context(get_style()):
            plot_efficiency_map(motor, rpm_max=motor.max_rpm, mirror=False, levels=15, export_path=path)
        path = "Graphs/efficiency_map_gen.pdf" if EXPORT else None
        with plt.rc_context(get_style()):
            plot_efficiency_map(motor, rpm_max=motor.max_rpm, mirror=True, levels=15, export_path=path)       
    
    print(f"{'V (km/h)':>10}{'N (rpm)':>10}{'C_m (Nm)':>10}{'eta_m (%)':>11}"
          f"{'P_bat (kW)':>12}{'E (kWh/km)':>12}")
    print("-" * 65)
    
    for v, N, C, eta, P, e in zip(speeds, rpm, C_m, eta_m, P_bat, E):
        print(f"{v:>10d}{N:>10.0f}{C:>10.2f}{eta * 1e2:>11.1f}"
              f"{P * 1e-3:>12.2f}{e:>12.4f}")
      
        
    # b. Plots
    print_section("Influence of the slope")
    slopes = [0, 5, 10] # %, Influence of the slope
    
    print(f"{'V (km/h)':>10}" + "".join(f"{f'E at {s} % (kWh/km)':>24}" for s in slopes))
    print("-" * 82)
    
    for v in speeds:
        print(f"{v:>10d}" + "".join(
            f"{consumption_constant_speed(car, motor, v, slope=s):>24.4f}" for s in slopes))
    
    
    if PLOT:
        speed = np.linspace(5, 150, 1000) # km/h
        E_slopes = [consumption_constant_speed(car, motor, speed, slope=s) 
            for s in slopes]
        
        with plt.rc_context(get_style()):
            path = "Graphs/consumption.pdf" if EXPORT else None
            plot_consumption(speed, E_slopes, slopes, 
                                     points=speeds, export_path=path)

        print("\nMinimum energy consumption:")
        for s, E_s in zip(slopes, E_slopes):
            k = np.argmin(E_s) # index of the smallest consumption
            print(f"  {s:>2d} % slope : E_min = {E_s[k]:.4f} (kWh/km) at V = {speed[k]:.1f} (km/h)")


    # Influence of the weather (flat road): heating in winter, cooling in summer
    print_section("Influence of the weather")
    weather = {"20 °C (HVAC off)": P_ACCESSORIES,
               "-5 °C (heating)":  P_ACCESSORIES_WINTER,
               "35 °C (cooling)":  P_ACCESSORIES_SUMMER}
    
    print(f"{'V (km/h)':>10}" + "".join(f"{name:>20}" for name in weather) + "   (kWh/km)")
    print("-" * 82)
    
    for v in speeds:
        print(f"{v:>10d}" + "".join(
            f"{consumption_constant_speed(car, motor, v, P_acc=P):>20.4f}" 
            for P in weather.values()))
    
    speed = np.linspace(5, 150, 1000) # km/h
    E_weather = [consumption_constant_speed(car, motor, speed, P_acc=P) 
                 for P in weather.values()]
    
    # Ratio with respect to the consumption at 20 °C
    print("\nRatio with respect to 20 °C:")
    for v in speeds:
        E_ref = consumption_constant_speed(car, motor, v)
        ratio_winter = consumption_constant_speed(car, motor, v, P_acc=P_ACCESSORIES_WINTER) / E_ref
        ratio_summer = consumption_constant_speed(car, motor, v, P_acc=P_ACCESSORIES_SUMMER) / E_ref
        print(f"  {v:>3d} km/h : winter x {ratio_winter:.2f} (+{(ratio_winter - 1) * 1e2:.0f} %) ; "
              f"summer x {ratio_summer:.2f} (+{(ratio_summer - 1) * 1e2:.0f} %)")
        
    
    print("\nMinimum energy consumption:")
    for name, E_w in zip(weather, E_weather):
        k = np.argmin(E_w)
        print(f"  {name:>18} : E_min = {E_w[k]:.4f} (kWh/km) at V = {speed[k]:.1f} (km/h)")
    
    if PLOT:
        with plt.rc_context(get_style()):
            path = "Graphs/consumption_weather.pdf" if EXPORT else None
            plot_consumption_weather(speed, E_weather, list(weather), points=speeds, 
                                     ymax=1.2, export_path=path)
        plt.show()
    
    
    
    
    
     # c. WLTP driving cycle
    print_section("WLTP driving cycle")
    t, v_wltp, phase = load_wltp(WLTP_FILE)
    P_wltp = battery_power_cycle(car, motor, t, v_wltp)
    
    print(f"{'Phase':>12}{'d (km)':>10}{'E_out (kWh)':>13}{'E_in (kWh)':>12}"
          f"{'E_net (kWh)':>13}{'E (kWh/km)':>12}")
    print("-" * 72)
    
    # One row per phase of the cycle
    for name in pd.unique(phase): # phases in the order of the cycle
        mask = phase == name
        d, E_out, E_in = cycle_energy(t[mask], v_wltp[mask], P_wltp[mask])
        print(f"{name:>12}{d:>10.2f}{E_out:>13.3f}{E_in:>12.3f}"
              f"{E_out + E_in:>13.3f}{(E_out + E_in) / d:>12.4f}")
    
    # Whole cycle
    d, E_out, E_in = cycle_energy(t, v_wltp, P_wltp)
    E_net = E_out + E_in
    E_wltp = E_net / d # kWh/km
    
    print("-" * 72)
    print(f"{'Total':>12}{d:>10.2f}{E_out:>13.3f}{E_in:>12.3f}"
          f"{E_net:>13.3f}{E_wltp:>12.4f}")
    
    print(f"\n  Duration: {t[-1] - t[0]:.0f} (s) ; mean speed: {d / (t[-1] - t[0]) * 3600:.1f} (km/h) ; "
      f"maximum speed: {v_wltp.max():.1f} (km/h)")
    print(f"  Maximum battery power: {P_wltp.max() * 1e-3:.1f} (kW) ; "
      f"maximum recovered power: {-P_wltp.min() * 1e-3:.1f} (kW)")
    # Share of the cycle where the battery delivers less than 10 kW
    P_low = 10e3 # W
    share = (P_wltp < P_low).mean() * 1e2 # %
    print(f"  Battery power below {P_low * 1e-3:.0f} kW during {share:.1f} % of the cycle") 

    # Mean powers over the cycle (kW): energy (kWh) / duration (h)
    T_h = (t[-1] - t[0]) / 3600 # duration of the cycle (h)
    print(f"  Mean battery power: out = {E_out / T_h:.2f} (kW) ; "
      f"in = {E_in / T_h:.2f} (kW) ; net = {E_net / T_h:.2f} (kW)")
    print(f"  Recovered / drawn: {-E_in / E_out * 1e2:.1f} %")

    print(f"  Energy consumption: {E_wltp:.4f} (kWh/km)")
    print(f"  Range (battery of {BATTERY_CAPACITY} kWh): {BATTERY_CAPACITY / E_wltp:.1f} (km)")
    print(f"  Range (usable capacity of 18.8 kWh): {18.8 / E_wltp:.1f} (km)")
    # Consumption at a constant speed equal to the mean speed of the cycle
    v_mean = d / (t[-1] - t[0]) * 3600 # km/h
    E_mean = consumption_constant_speed(car, motor, v_mean)
    print(f"  Consumption at a constant speed of {v_mean:.1f} km/h: {E_mean:.4f} (kWh/km) ")
    
    
    
    
    # Effect of the braking energy recovery coefficient alpha
    print_section("Influence of alpha")
    weather = {"20 °C (HVAC off)": P_ACCESSORIES,
               "-5 °C (heating)":  P_ACCESSORIES_WINTER,
               "35 °C (cooling)":  P_ACCESSORIES_SUMMER}
    
    print(f"{'V (km/h)':>10}" + "".join(f"{name:>20}" for name in weather) + "   (kWh/km)")
    print("-" * 82)
    
    print("\n  Effect of alpha:")
    print(f"{'alpha':>8}{'P_in (kW)':>12}{'P_net (kW)':>12}{'E (kWh/km)':>13}"
      f"{'Variation (%)':>15}{'Range (km)':>12}")
    alphas = [0.3, 0.4, 0.5]
    P_alphas = []
    T_h = (t[-1] - t[0]) / 3600 # duration of the cycle (h)

    for alpha in alphas:
        P_a = battery_power_cycle(car, motor, t, v_wltp, alpha=alpha)
        P_alphas.append(P_a)
        d_a, E_out_a, E_in_a = cycle_energy(t, v_wltp, P_a)
        E_a = (E_out_a + E_in_a) / d_a # kWh/km
        print(f"{alpha:>8.1f}{E_in_a / T_h:>12.2f}{(E_out_a + E_in_a) / T_h:>12.2f}{E_a:>13.4f}"
          f"{(E_a / E_wltp - 1) * 1e2:>15.2f}{BATTERY_CAPACITY / E_a:>12.1f}")
    
    # Strongest braking between 1300 and 1500 s: recovered power for each alpha
    window = (t >= 1300) & (t <= 1500)
    k = np.argmin(np.where(window, P_alphas[0], np.inf))
    t_mark = t[k]

    print(f"\n  Strongest braking of the window, at t = {t_mark:.0f} (s):")
    for alpha, P_a in zip(alphas, P_alphas):
        print(f"    alpha = {alpha:.1f} : P_bat = {P_a[k] * 1e-3:.1f} (kW)")
   
    
    if PLOT:
       with plt.rc_context(get_style()):
           path = "Graphs/wltp.pdf" if EXPORT else None
           phase_plot = np.where(phase == "Middle", "Medium", phase) 
           plot_cycle(t, v_wltp, P_wltp, phase=phase_plot, export_path=path)
           path = "Graphs/wltp_alpha.pdf" if EXPORT else None
           plot_cycle_alpha(t, P_alphas, alphas, xlim=(1300, 1500), t_mark=t_mark, export_path=path)
       plt.show()
    return

if __name__ == "__main__":
    
    main()
