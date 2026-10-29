#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:40:27 2026

@author: antoinedlbck
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter

def plot_torque_power(rpm, C_peak, P_peak, k_oc, highlight="peak", export_path=None):
    """
    Power and torque curves (peak + nominal)
    
    highlight puts one or the other in evidence
    """
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nominal" else 0.3

    fig, ax1 = plt.subplots()
    ax1.grid()
    ax1.plot(rpm, C_peak, color="orange", alpha=alpha_peak, label="Peak $C$")
    ax1.plot(rpm, C_peak / k_oc, color="red", alpha=alpha_nom, label="Nominal $C$")
    ax1.set_xlabel(r"Motor speed (10$^3$ rpm)")
    ax1.set_xlim(0, rpm.max())
    ax1.set_ylabel("Torque (Nm)")
    ax1.set_ylim(0, 300)
    ticks = np.append(np.arange(0, 11001, 1000), rpm.max())
    ax1.xaxis.set_major_locator(FixedLocator(ticks))
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1000:g}" if x != 11000 else ""))
    ax1.yaxis.set_major_locator(MultipleLocator(25))

    ax2 = ax1.twinx()
    ax2.plot(rpm, P_peak * 1e-3, alpha=alpha_peak, label="Peak $P$")
    ax2.plot(rpm, P_peak * 1e-3 / k_oc, color="teal", alpha=alpha_nom, label="Nominal $P$")
    ax2.set_ylabel("Power (kW)")
    ax2.set_ylim(0, 180)
    ax2.yaxis.set_major_locator(MultipleLocator(15))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    
    if export_path:
        fig.savefig(export_path, dpi=300)
        
    return fig


def plot_traction(rpm, F_t, export_path=None):
    
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(rpm, F_t)
    ax.set_xlabel(r"Motor speed (10$^3$ rpm)")
    ax.set_ylabel(r"Tractive force ($N$)")
    
    ax.set_xlim(0, rpm.max())
    ticks = np.append(np.arange(0, 11001, 1000), rpm.max())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1000:g}" if x != 11000 else ""))
    
    if export_path:
        fig.savefig(export_path, dpi=300)
        
    return fig


def plot_topspeed(speed, eta_tr, P_max, wheel_power, resistive_power, under_redline, k_oc, highlight, export_path=None):
    
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nominal" else 0.3
    
    fig, ax = plt.subplots()
    ax.grid()
    ax.axhline(eta_tr * P_max * 1e-3, ls="--", color="black", alpha=alpha_peak,
               label=r"Maximum peak tractive power $\eta_t \mathcal{P}_{1, \mathsf{peak}}$")
    ax.axhline(eta_tr * P_max * 1e-3 / k_oc, ls="--", color="black", alpha=alpha_nom,
               label=r"Maximum nominal tractive power $\eta_t \mathcal{P}_{1, \mathsf{nom}}$")
    ax.plot(speed[under_redline], wheel_power * 1e-3, alpha = alpha_peak, 
            label = r"$\mathcal{P}_\mathsf{wheels, peak}$")
    ax.plot(speed[under_redline], wheel_power * 1e-3 / k_oc, alpha = alpha_nom, 
            label = r"$\mathcal{P}_\mathsf{wheels, nom}$")
    ax.plot(speed, resistive_power * 1e-3, 
            label = r"Resistive power $\mathcal{P}_\mathsf{RES}$")
    ax.set_xlim(0, speed.max())
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Power (kW)")
    
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    if export_path:
        fig.savefig(export_path, dpi=300)
    
    return fig
    