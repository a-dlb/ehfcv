#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:40:27 2026

@author: antoinedlbck
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from constants import RADS_TO_RPM, KMH_TO_MS
from resistive_forces import total_force, total_power, drag_force, grading_force, rolling_res_force, net_force

def plot_resistive_force(vehicle, slopes, decompose=True, export_path=None):
    """
    Plot the resistive forces as a function of the speed of the vehicle 
    slope. Plots every contribution and the total force.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    slopes : np.array
        Slopes to consider, in (%).
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    list(plt.fig)
        The plots!
    """
    
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h

    figs = []
        
    
    def crossing(V_kmh, f1, f2):
        """First speed (km/h) where f1 surpasses f2, None if not in the span."""
        d = f1 - f2
        idx = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        # linear interpolation between the two points
        return V_kmh[i] - d[i] * (V_kmh[i+1] - V_kmh[i]) / (d[i+1] - d[i])
    
    
    for s in slopes:
        
        fig, ax = plt.subplots()
        
        F_aero  = drag_force(vehicle, speed)
        F_roll  = rolling_res_force(vehicle, speed, s)
        F_grade = grading_force(vehicle, s) # returns a scalar
        F_grade = np.full_like(speed, F_grade)
        F_tot   = total_force(vehicle, speed, s)
    

        ax.plot(speed, F_aero,  color="tab:green",  label="Aerodynamic drag")
        ax.plot(speed, F_roll,  color="tab:orange",    label="Rolling resistance")
        ax.plot(speed, F_grade, color="tab:pink", label=f"Grading resistance ({s:.0f} %)")
        ax.plot(speed, F_tot,   color="tab:blue",   label="Total resistance")
    
        # V1 : drag = rolling ; V2 : drag = grading (si dans la plage de vitesse)
        V1 = crossing(speed, F_aero, F_roll)
        if V1 is not None:
            ax.plot(V1, np.interp(V1, speed, F_roll), "o", color="black",
                    markersize=15, zorder=5, label=rf"$V_1$ = {V1:.1f} km/h")
        if s > 0:
            V2 = crossing(speed, F_aero, F_grade)
            if V2 is not None:
                ax.plot(V2, np.interp(V2, speed, F_grade), "o", color="tab:brown",
                        markersize=15, zorder=5, label=rf"$V_2$ = {V2:.1f} km/h")
    
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Resistance force (N)")
        ax.set_xlim(0, vehicle.max_speed)
        ax.set_ylim(bottom=-4)
        
        legend_kw = dict(loc="upper left", framealpha=0.9)
        ax.legend(**legend_kw)
        
        ax.grid(True)
        
        figs.append(fig)
        
        if export_path:
            fig.savefig(export_path + f"_{s}percent.pdf", dpi=300)
        
    return figs


def plot_resistive_power(vehicle, slopes, export_path=None):
    """
    Plot the  total resistive power as a function of the speed of the vehicle 
    and slope.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    slopes : np.array
        Slopes to consider, in (%).
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    
    fig, ax = plt.subplots()
        
    for s in slopes:
        
        P = total_power(vehicle, speed, s)
        label = f"slope {s:.0f} %"
        ax.plot(speed, P / 1000, label=label)     
    
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel(r"$P_{road}$ (kW)")
    ax.set_xlim(0, vehicle.max_speed)
    ax.set_ylim(bottom=0)
    
    ax.grid(True)
    
    legend_kw = dict(loc="upper left", framealpha=0.9)
    ax.legend(**legend_kw)
        
    if export_path:
        fig.savefig(export_path, dpi=300)
    
    return fig


def plot_torque_power(rpm, C_peak, P_peak, k_oc, highlight="peak", export_path=None):
    """
    Power and torque curves (peak + nominal)
    
    highlight puts one or the other in evidence
    """
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nom" else 0.3

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


def plot_netforce(vehicle, motor, speed, slopes, highlight="peak", export_path=True):
    """
    Plots the net force as a function of the speed an the slopes.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    speed : np.ndarray
        The speeds to consider (in km/h).
    slopes : np.ndarray
        The slopes for the grading force (in %)
    highlight : string, optional
        Which power mode to put in evidence. The default is "peak".
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nom" else 0.3
    
    fig, ax = plt.subplots()
    
    for s in slopes:
        
        F_net_peak = net_force(vehicle, motor, speed, s, power="peak")
        F_net_nom = net_force(vehicle, motor, speed, s, power="nom")
        
        ax.plot(speed, F_net_peak, label=f"slope {s} %" if highlight=="peak" else None, alpha=alpha_peak)
        ax.plot(speed, F_net_nom, label=f"slope {s} %" if highlight=="nom" else None, alpha=alpha_nom)
    
    ax.set_ylabel(r"$F_{net}$ (N)")
    
    rpm_base = motor.base_speed_rpm
    R_e = vehicle.tire_effective_radius_rear
    i_red = vehicle.gear_ratio
    V_base_kmh = rpm_base * 2*np.pi/60 * R_e / i_red * 3.6
    
    ax.axvline(V_base_kmh, color="grey", linestyle="--", linewidth=3)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Speed (km/h)")
    ax.set_xlim(0, vehicle.max_speed)
    ax.set_ylim(bottom=-1000)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    
    if export_path:
        fig.savefig(export_path, dpi=300)
    
    return fig


def plot_acc(vehicle, motor, speed, slopes, highlight="peak", export_path=True):
    """
    Plots the acceleration as a function of the speed an the slopes.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    speed : np.ndarray
        The speeds to consider (in km/h).
    slopes : np.ndarray
        The slopes for the grading force (in %)
    highlight : string, optional
        Which power mode to put in evidence. The default is "peak".
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nom" else 0.3
    
    m_et = vehicle.total_effective_mass(motor)
    
    fig, ax = plt.subplots()
    
    for s in slopes:
        
        a_x_peak = net_force(vehicle, motor, speed, s, power="peak") / m_et
        a_x_nom = net_force(vehicle, motor, speed, s, power="nom") / m_et
        
        ax.plot(speed, a_x_peak, label=f"slope {s} %" if highlight=="peak" else None, alpha=alpha_peak)
        ax.plot(speed, a_x_nom, label=f"slope {s} %" if highlight=="nom" else None, alpha=alpha_nom)
    
    ax.set_ylabel(r"$a_x$ (m/s$^2$)")
    
    rpm_base = motor.base_speed_rpm
    R_e = vehicle.tire_effective_radius_rear
    i_red = vehicle.gear_ratio
    V_base_kmh = rpm_base * 2*np.pi/60 * R_e / i_red * 3.6
    
    ax.axvline(V_base_kmh, color="grey", linestyle="--", linewidth=3)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Speed (km/h)")
    ax.set_xlim(0, vehicle.max_speed)
    ax.set_ylim(bottom=-1)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    
    if export_path:
        fig.savefig(export_path, dpi=300)
    
    return fig


def plot_acc_capa(vehicle, motor, speed, slopes, highlight="peak", export_path=True):
    """
    Plots the acceleration capability as a function of the speed an the slopes.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    speed : np.ndarray
        The speeds to consider (in km/h).
    slopes : np.ndarray
        The slopes for the grading force (in %)
    highlight : string, optional
        Which power mode to put in evidence. The default is "peak".
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nom" else 0.3
    
    m_et = vehicle.total_effective_mass(motor)
    
    fig, ax = plt.subplots()
    
    for s in slopes:
        
        a_x_peak = net_force(vehicle, motor, speed, s, power="peak") / m_et
        a_x_nom = net_force(vehicle, motor, speed, s, power="nom") / m_et
        
        # Mask when it goes to infty
        a_x_peak = np.where(a_x_peak > 0, a_x_peak, np.nan)
        a_x_nom  = np.where(a_x_nom  > 0, a_x_nom,  np.nan)
        
        acc_capa_peak = 1/a_x_peak
        acc_capa_nom = 1/a_x_nom
        
        ax.plot(speed, acc_capa_peak, label=f"slope {s} %" if highlight=="peak" else None, alpha=alpha_peak)
        ax.plot(speed, acc_capa_nom, label=f"slope {s} %" if highlight=="nom" else None, alpha=alpha_nom)
    
    ax.set_ylabel(r"$1/a_x$ (s$^2$/m)")
    
    rpm_base = motor.base_speed_rpm
    R_e = vehicle.tire_effective_radius_rear
    i_red = vehicle.gear_ratio
    V_base_kmh = rpm_base * 2*np.pi/60 * R_e / i_red * 3.6
    
    ax.axvline(V_base_kmh, color="grey", linestyle="--", linewidth=3)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Speed (km/h)")
    ax.set_xlim(0, vehicle.max_speed)
    ax.set_ylim(0, 2)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    
    if export_path:
        fig.savefig(export_path, dpi=300)
    
    return fig


def plot_topspeed(speed, eta_tr, P_max, wheel_power, resistive_power, under_redline, k_oc, v_max, highlight, export_path=None):
    
    alpha_peak = 1.0 if highlight == "peak" else 0.3
    alpha_nom = 1.0 if highlight == "nom" else 0.3
    
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

    
    # v_max
    y_limb, y_limt = ax.get_ylim()
    ax.axvline(v_max, ymax=(eta_tr * P_max * 1e-3)/y_limt, ls="dotted", color="red", 
               label=r"$v_\mathsf{{max}}$" if highlight=="peak" else None, alpha=alpha_peak)
    ax.axvline(v_max, ymax=(eta_tr * P_max * 1e-3 / k_oc)/y_limt, ls="dotted", color="red", 
               label=r"$v_\mathsf{{max}}$" if highlight=="nom" else None, alpha=alpha_nom)
    ax.scatter(v_max, eta_tr * P_max * 1e-3, color="red", alpha=alpha_peak, zorder=4)
    ax.scatter(v_max, eta_tr * P_max * 1e-3 / k_oc, color="red", alpha=alpha_nom, zorder=4)
            
    # v_maxmax (peak)
    level = eta_tr * P_max
    i = np.argmin(np.abs(resistive_power - level))
    v_maxmax_peak = speed[i]
    p_maxmax = resistive_power[i] * 1e-3
    ax.scatter(v_maxmax_peak, p_maxmax, color="purple", alpha=alpha_peak, zorder=4)
    ax.axvline(v_maxmax_peak, ls="dotted", color="purple", 
               label=r"$v_\mathsf{{max}}^\mathsf{{max}}$ (peak)",
               alpha=alpha_peak, ymax=p_maxmax/y_limt)
    
    # v_maxmax (nom)
    level = eta_tr * P_max / k_oc
    i = np.argmin(np.abs(resistive_power - level))
    v_maxmax_nom = speed[i]
    p_maxmax = resistive_power[i] * 1e-3
    ax.scatter(v_maxmax_nom, p_maxmax, color="purple", alpha=alpha_nom, zorder=4)
    ax.axvline(v_maxmax_nom, ls="dotted", color="purple", 
               label=r"$v_\mathsf{{max}}^\mathsf{{max}}$ (nom)",
               alpha=alpha_nom, ymax=p_maxmax/y_limt)

    xmax = speed.max()
    v_maxs = [v_max, v_maxmax_peak, v_maxmax_nom]
    
    base = [t for t in ax.get_xticks() if 0 <= t <= xmax]
    xtk = sorted(set(base + v_maxs))
    
    is_special = lambda t: np.isclose(t, v_maxs).any()
    is_near = lambda t: any(abs(t - v) <= 0.1 * xmax for v in v_maxs)
    
    labels = []
    for t in xtk:
        if is_special(t):
            labels.append(f"{t:.1f}")
        elif is_near(t):
            labels.append("")
        else:
            labels.append(f"{t:.0f}")
    
    ax.set_xticks(xtk)
    ax.set_xticklabels(labels)
    
    lw = 6
    
    for tick, t in zip(ax.xaxis.get_major_ticks(), xtk):
        if np.isclose(t, v_max):
            color, alpha = "red", 1.0
        elif np.isclose(t, v_maxmax_peak):
            color, alpha = "purple", alpha_peak
        elif np.isclose(t, v_maxmax_nom):
            color, alpha = "purple", alpha_nom
        else:
            continue
    
        tick.label1.set_color(color)
        tick.label1.set_fontweight("bold")
        tick.label1.set_alpha(alpha)
    
        tick.tick1line.set_marker("o")
        tick.tick1line.set_markersize(lw)
        tick.tick1line.set_markeredgewidth(0)
        tick.tick1line.set_markerfacecolor(color)
        tick.tick1line.set_markeredgecolor(color)
        tick.tick1line.set_alpha(alpha)
                    
    ax.set_xlim(0, speed.max())
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Power (kW)")
                    
    # We want the axes to have the same size no matter the legend size so 
    # this is required for a little trick after drawing the legend
    w, h = fig.get_size_inches()
    
    fig.legend(loc="outside lower center", ncol=2)
    fig.get_layout_engine().set(h_pad=0.2)
    
    # Now we apply the box aspect given by the style dict
    ax.set_box_aspect(h / w)
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_max_slopes(data, powers, export_path=None):
    """
    Max slope for different mu and both power regimes
    (this plot code is generated by Claude since it is not the focus of the course
     to handle matplotlib...)
    of course the related computations in the modules are handmade!
    """

    rc = plt.rcParams
    base_fs = float(rc["font.size"])
    tick_fs = float(rc["xtick.labelsize"])
    fs_bar = 0.6 * base_fs
    fs_value = 0.65 * base_fs

    names = {"wet02": "Wet (0.2 mm)", "wet2": "Wet (2 mm)"}
    colors = dict(zip(powers, ["tab:purple", "black"]))

    w_gray = 0.7
    w_thin = 0.08
    label_offset = 0.4
    spacing_minmax = 3.2
    gap_group = 1.6

    def pretty(name):
        return names.get(name, name.replace("_", " ").title())

    ymax = max(
        max(max(a for _, _, adhs in groups for a in adhs), max(trac.values()))
        for trac, groups in data.values()
    ) * 1.15

    n = len(powers)
    gap = (w_gray - n * w_thin) / (n + 1)

    figs = []

    for s, (trac, groups) in data.items():
        fig, ax = plt.subplots()

        x = 0.0
        ticks, tick_labels, group_centers = [], [], []

        for name, values, adhs in groups:
            xs = []
            for j, (mu, adh) in enumerate(zip(values, adhs)):
                ax.bar(x, adh, w_gray, color="lightgray", edgecolor="gray", zorder=2)
                ax.text(x, adh, f"{adh:.1f}%", ha="center", va="bottom",
                        fontsize=fs_value, fontweight="bold", color="dimgray")

                for k, p in enumerate(powers):
                    eff = min(adh, trac[p])
                    lim = "adh" if adh < trac[p] else "trac"
                    c = colors[p]
                    xb = x - w_gray / 2 + gap * (k + 1) + w_thin * (k + 0.5)
                    ax.bar(xb, eff, w_thin, color=c, zorder=3)

                    if k == 0:
                        ax.text(x - w_gray / 2 - label_offset, eff / 2, f"{eff:.1f}%\n({lim})",
                                ha="right", va="center", fontsize=fs_bar,
                                color=c, fontweight="bold", zorder = 10)
                    else:
                        ax.text(x + w_gray / 2 + label_offset, eff / 2, f"{eff:.1f}%\n({lim})",
                                ha="left", va="center", fontsize=fs_bar,
                                color=c, fontweight="bold", zorder = 10)

                mu_txt = rf"$\mu$ = {mu}"
                if len(values) == 1:
                    tick_labels.append(mu_txt)
                else:
                    tick_labels.append(("min" if j == 0 else "max") + "\n" + mu_txt)
                ticks.append(x)
                xs.append(x)
                x += spacing_minmax
            x -= spacing_minmax
            group_centers.append((name, np.mean(xs)))
            x += spacing_minmax + gap_group

        for p in powers:
            ax.axhline(trac[p], color=colors[p], ls="--", zorder=4)

        ax.set_xlim(ticks[0] - 2.0, ticks[-1] + 2.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        
        sec = ax.secondary_xaxis("bottom")
        sec.set_xticks([xc for _, xc in group_centers])
        sec.set_xticklabels([pretty(name) for name, _ in group_centers], fontweight="bold")
        sec.tick_params(length=0, pad=rc["xtick.major.size"] + rc["xtick.major.pad"] + 2.4 * tick_fs)
        sec.spines["bottom"].set_visible(False)

        ax.set_xlabel(" \n ", labelpad=rc["axes.labelpad"])

        ax.set_ylim(0, ymax)
        
        lw = plt.rcParams["lines.linewidth"]
        base = [t for t in ax.get_yticks() if 0 <= t <= ymax]
        ytk = sorted(set(base + list(trac.values())))
        
        is_special = lambda t: np.isclose(t, list(trac.values())).any()
        is_near = lambda t: any(abs(t - v) <= 0.05 * ymax for v in trac.values())
        
        labels = []
        for t in ytk:
            if is_special(t):
                labels.append(f"{t:.1f}")
            elif is_near(t):
                labels.append("")
            else:
                labels.append(f"{t:.1f}")
        
        ax.set_yticks(ytk)
        ax.set_yticklabels(labels)
        
        for tick, t in zip(ax.yaxis.get_major_ticks(), ytk):
            for p, v in trac.items():
                if np.isclose(t, v):
                    tick.label1.set_color(colors[p])
                    tick.label1.set_fontweight("bold")
                    tick.tick1line.set_markeredgecolor(colors[p])
                    tick.tick1line.set_markeredgewidth(lw)

        ax.set_ylabel("Max slope (%)")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        handles = [Patch(facecolor="lightgray", edgecolor="gray",
                         label=r"Max slope limited by adherence $\theta_\mathsf{adh}$")]
        handles += [Patch(facecolor=colors[p], label=rf"Effective max slope $\theta_\mathsf{{max}}$ ({p})")
                    for p in powers]
        handles += [Line2D([0], [0], color=colors[p], ls="--",
                           label=rf"Max slope limited by traction $\theta_\mathsf{{trac}}$ ({p})")
                    for p in powers]
        handles += [Line2D([], [], linestyle="", label="(adh) or (trac) = effective limiting factor")]
        
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
        fig.get_layout_engine().set(h_pad=0.2)

        if export_path:
            path = Path(export_path)
            fig.savefig(path.with_name(f"{path.stem}_{s}kmh{path.suffix}"))

        figs.append(fig)

    plt.show()
    return figs


def plot_traction(vehicle, motor, slopes, export_path=None):
    """
    Plots tractive forces, resistive forces against the speed and 
    their possible intersections. 

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    slopes : np.ndarray
        The slopes for the grading force (in %)
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    i = vehicle.gear_ratio
    R_er = vehicle.tire_effective_radius_rear
    k_oc = motor.overcharging_factor
            
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    rpm = (speed * KMH_TO_MS) * (i/R_er) * RADS_TO_RPM # rpm
    
    F_t = vehicle.tractive_force(motor, rpm)
    
    fig, ax = plt.subplots()
    
    ax.set_ylim(0, np.max(np.abs(F_t)) * 1.1)
    
    powers = ["peak", "nom"]
    
    for p in powers:
    
        if p == "nom":
            F_t /= k_oc
                
        # Tractive forces plot
        color = "tab:purple" if p == "peak" else "tab:cyan"
        p_string = "peak" if p == "peak" else "nominal"
        ax.plot(speed, F_t, color=color, label= f"Tractive force ({p_string} power)")
    
    
    # Resistive forces plot
    ls = {slopes[0]: "-", slopes[1]: "--", slopes[2]: ":"}
    
    for slope in slopes:
        
        F_res = total_force(vehicle, speed, slope)
        
        ax.plot(speed, F_res, color="black", ls=ls[slope],
                label=f"Total road resistive force ({slope} % slope)")
        
        # Intersection
        tol = 1e-2 * np.max(np.abs(F_t))
        gap = np.abs(F_t - F_res)
        k = np.argmin(gap)
        
        if gap[k] <= tol:
            ax.scatter(speed[k], F_t[k], zorder=3, color="red",
                       label = rf"$V_{{\mathsf{{lim}},\ {slope}\,\%\ \mathsf{{slope}}}}$ = {speed[k]:.1f} (km/h)")
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Force (N)")
    ax.grid()
    
    # We want the axes to have the same size no matter the legend size so 
    # this is required for a little trick after drawing the legend
    w, h = fig.get_size_inches()
    
    fs = plt.rcParams["legend.fontsize"] - 2
    fig.legend(loc="outside lower center", ncol=2, fontsize=fs)
    fig.get_layout_engine().set(h_pad=0.2)
    
    # Now we apply the box aspect given by the style dict
    ax.set_box_aspect(h / w)
    
    ax.set_xlim(0, vehicle.max_speed)
    ticks = np.arange(0, vehicle.max_speed+1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_power(vehicle, motor, slopes, export_path=None):
    """
    Plots tractive power, resistive power against the speed and 
    their possible intersections. 

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    slopes : np.ndarray
        The slopes for the grading force (in %)
    export_path : string, optional
        If export, specify the path. The default is None.

    Returns
    -------
    plt.fig
        The plot!
    """
    
    i = vehicle.gear_ratio
    R_er = vehicle.tire_effective_radius_rear
    k_oc = motor.overcharging_factor
            
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    rpm = (speed * KMH_TO_MS) * (i/R_er) * RADS_TO_RPM # rpm
    
    P_w = vehicle.tractive_power(motor, rpm) * 1e-3 # kW
    
    fig, ax = plt.subplots()
    
                
    # Tractive power plot
    
    powers = ["peak", "nom"]
    
    for p in powers:
    
        if p == "nom":
            P_w /= k_oc
                
        # Tractive forces plot
        color = "tab:purple" if p == "peak" else "tab:cyan"
        p_string = "peak" if p == "peak" else "nominal"
        ax.plot(speed, P_w, color=color, label= f"Tractive power ({p_string} power)")
    
    
    # Resistive forces plot
    ls = {slopes[0]: "-", slopes[1]: "--", slopes[2]: ":"}
    
    for slope in slopes:
        
        P_res = total_power(vehicle, speed, slope) * 1e-3 # kW
        
        ax.plot(speed, P_res, color="black", ls=ls[slope],
                label=f"Total road resistive power ({slope} % slope)")
        
        # Intersection (note: need to exclude speeds around 0 where there is an intersection)
        tol = 1e-2 * np.max(np.abs(P_w))
        v_tol = 10 # km/h
        gap = np.abs(P_w - P_res)
        
        # mask = where the speed is above v_tol
        mask = speed > v_tol
        
        # set artificially infinite gap where speed is under v_tol (~ = boolean inverse) 
        # so that these speeds aren't taken into account in argmin 
        gap[~mask] = np.inf
        
        k = np.argmin(gap)
        
        if gap[k] <= tol:
            ax.scatter(speed[k], P_w[k], zorder=3, color="red",
                       label = rf"$V_{{\mathsf{{lim}},\ {slope}\,\%\ \mathsf{{slope}}}}$ = {speed[k]:.1f} (km/h)")
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Power ($kW$)")
    ax.grid()
    
    # We want the axes to have the same size no matter the legend size so 
    # this is required for a little trick after drawing the legend
    w, h = fig.get_size_inches()
    
    fs = plt.rcParams["legend.fontsize"] - 2
    fig.legend(loc="outside lower center", ncol=2, fontsize=fs)
    fig.get_layout_engine().set(h_pad=0.2)
    
    # Now we apply the box aspect given by the style dict
    ax.set_box_aspect(h / w)
    
    ax.set_ylim(0, 150)
    ticks = np.arange(0, 151, 25)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    
    ax.set_xlim(0, vehicle.max_speed)
    ticks = np.arange(0, vehicle.max_speed+1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_traction_opti(vehicle, motor, slopes, i_dict, export_path=None):
    """
    Plots tractive forces, resistive forces against the speed and 
    their possible intersections, with different gear ratios
    for peak and nominal power regimes.

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    slopes : np.ndarray
        The slopes for the grading force (in %)
    export_path : string, optional
        If export, specify the path. The default is None.
    i_dict : dict
        The gear ratios to consider. Order is (peak, nominal).

    Returns
    -------
    plt.fig
        The plot!
    """
    
    R_er = vehicle.tire_effective_radius_rear
    k_oc = motor.overcharging_factor
            
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
            
    fig, ax = plt.subplots()
    
    powers = ["peak", "nom"]
    
    for p in powers:
        
        i = i_dict[p]
        rpm = (speed * KMH_TO_MS) * (i/R_er) * RADS_TO_RPM # rpm
        
        F_t = vehicle.tractive_force(motor, rpm)
    
        if p == "nom":
            F_t /= k_oc
        
        if p == "peak":
            ax.set_ylim(0, np.max(np.abs(F_t)) * 1.1)
        
        color = "tab:pink" if p == "peak" else "royalblue"
        p_string = "peak" if p == "peak" else "nominal"
        ax.plot(speed, F_t, color=color, label= f"Tractive force ({p_string} power)")
        
        
    
    # Resistive forces plot
    ls = {slopes[0]: "-", slopes[1]: "--", slopes[2]: ":"}
    
    for slope in slopes:
        
        F_res = total_force(vehicle, speed, slope)
        
        ax.plot(speed, F_res, color="black", ls=ls[slope],
                label=f"Total road resistive force ({slope} % slope)")
        
        # Intersection
        tol = 1e-2 * np.max(np.abs(F_t))
        gap = np.abs(F_t - F_res)
        k = np.argmin(gap)
        
        if gap[k] <= tol:
            ax.scatter(speed[k], F_t[k], zorder=3, color="red",
                       label = rf"$V_{{\mathsf{{lim}},\ {slope}\,\%\ \mathsf{{slope}}}}$ = {speed[k]:.1f} (km/h)")
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Force (N)")
    ax.grid()
    
    # We want the axes to have the same size no matter the legend size so 
    # this is required for a little trick after drawing the legend
    w, h = fig.get_size_inches()
    
    fs = plt.rcParams["legend.fontsize"] - 2
    fig.legend(loc="outside lower center", ncol=2, fontsize=fs)
    fig.get_layout_engine().set(h_pad=0.2)
    
    # Now we apply the box aspect given by the style dict
    ax.set_box_aspect(h / w)
    
    ax.set_xlim(0, vehicle.max_speed)
    ticks = np.arange(0, vehicle.max_speed+1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_power_opti(vehicle, motor, slopes, i_dict, export_path=None):
    """
    Plots tractive power, resistive power against the speed and 
    their possible intersections. 

    Parameters
    ----------
    vehicle : Vehicle()
        The vehicle to take into account.
    motor : Motor()
        The motor in the vehicle.
    slopes : np.ndarray
        The slopes for the grading force (in %)
    export_path : string, optional
        If export, specify the path. The default is None.
    i_dict : dict
        The gear ratios to consider. Order is (peak, nominal).

    Returns
    -------
    plt.fig
        The plot!
    """
    
    R_er = vehicle.tire_effective_radius_rear
    k_oc = motor.overcharging_factor
    
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    
    fig, ax = plt.subplots()
    
    # Tractive power plot
    
    powers = ["peak", "nom"]
    
    for p in powers:
        
        i = i_dict[p]
                    
        rpm = (speed * KMH_TO_MS) * (i/R_er) * RADS_TO_RPM # rpm
        
        P_w = vehicle.tractive_power(motor, rpm) * 1e-3 # kW
    
        if p == "nom":
            P_w /= k_oc
                
        # Tractive forces plot (ideal gear ratio)
        color = "tab:pink" if p == "peak" else "royalblue"
        p_string = "peak" if p == "peak" else "nominal"
        ax.plot(speed, P_w, color=color, label= f"Tractive power ({p_string} power, optimal $i$)")
        
        
        i = vehicle.gear_ratio
        
        rpm = (speed * KMH_TO_MS) * (i/R_er) * RADS_TO_RPM # rpm
        
        P_w = vehicle.tractive_power(motor, rpm) * 1e-3 # kW
    
        if p == "nom":
            P_w /= k_oc
                
        # Tractive forces plot (actual gear ratio)
        color = "tab:purple" if p == "peak" else "tab:cyan"
        p_string = "peak" if p == "peak" else "nominal"
        ax.plot(speed, P_w, color=color,
                label= f"Tractive power ({p_string} power, actual $i$)",
                alpha=0.3)
    
    
    # Resistive forces plot
    ls = {slopes[0]: "-", slopes[1]: "--", slopes[2]: ":"}
    
    for slope in slopes:
        
        P_res = total_power(vehicle, speed, slope) * 1e-3 # kW
        
        ax.plot(speed, P_res, color="black", ls=ls[slope],
                label=f"Total road resistive power ({slope} % slope)")
        
        # Intersection (note: need to exclude speeds around 0 where there is an intersection)
        tol = 1e-2 * np.max(np.abs(P_w))
        v_tol = 10 # km/h
        gap = np.abs(P_w - P_res)
        
        # mask = where the speed is above v_tol
        mask = speed > v_tol
        
        # set artificially infinite gap where speed is under v_tol (~ = boolean inverse) 
        # so that these speeds aren't taken into account in argmin 
        gap[~mask] = np.inf
        
        k = np.argmin(gap)
        
        if gap[k] <= tol:
            ax.scatter(speed[k], P_w[k], zorder=3, color="red",
                       label = rf"$V_{{\mathsf{{lim}},\ {slope}\,\%\ \mathsf{{slope}}}}$ = {speed[k]:.1f} (km/h)")
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Power ($kW$)")
    ax.grid()
    
    # We want the axes to have the same size no matter the legend size so 
    # this is required for a little trick after drawing the legend
    w, h = fig.get_size_inches()
    
    fs = plt.rcParams["legend.fontsize"] - 2
    fig.legend(loc="outside lower center", ncol=2, fontsize=fs)
    fig.get_layout_engine().set(h_pad=0.2)
    
    # Now we apply the box aspect given by the style dict
    ax.set_box_aspect(h / w)
    
    ax.set_ylim(0, 150)
    ticks = np.arange(0, 151, 25)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    
    ax.set_xlim(0, vehicle.max_speed)
    ticks = np.arange(0, vehicle.max_speed+1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig 


def plot_consumption(speed, consumptions, slopes, points=None, export_path=None):
    """
    Plots the energy consumption at constant speed against the speed,
    for different slopes.
 
    Parameters
    ----------
    speed : np.ndarray
        The speeds (in km/h)
    consumptions : list of np.ndarray
        The energy consumption curves (in kWh/km), one per slope
    slopes : list
        The slopes (in %) corresponding to the curves
    points : np.ndarray, optional
        Specific speeds (in km/h) to highlight on each curve. The default is None.
    export_path : string, optional
        If export, specify the path. The default is None.
 
    Returns
    -------
    plt.fig
        The plot!
    """
    
    fig, ax = plt.subplots()
    
    colors = ["tab:purple", "tab:cyan", "tab:orange"]
    
    for E, slope, color in zip(consumptions, slopes, colors):
        
        ax.plot(speed, E, color=color, label=f"{slope} % slope")
        
        if points is not None:
            ax.scatter(points, np.interp(points, speed, E), zorder=3, color="red",
                       label="30, 50, 90 and 120 km/h" if slope == slopes[-1] else None)
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Energy consumption (kWh/km)")
    ax.grid()
    
    # Same trick as the other plots to keep the axes size
    w, h = fig.get_size_inches()
    
    fig.legend(loc="outside lower center", ncol=2)
    fig.get_layout_engine().set(h_pad=0.2)
    
    ax.set_box_aspect(h / w)
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, speed.max())
    ticks = np.arange(0, speed.max() + 1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig



def plot_cycle(t, speed, P_bat, phase=None, export_path=None):
    """
    Plots a driving cycle in two separate figures: 
    vehicle speed against time, and battery power against time.
    t in (s), speed in (km/h), P_bat in (W)
    (P_bat > 0: drawn from the battery, < 0: recovered by braking)
    
    If export_path is given (e.g. "Graphs/wltp.pdf"), the figures are saved
    as "Graphs/wltp_speed.pdf" and "Graphs/wltp_power.pdf".
    """
    
    lw = plt.rcParams["lines.linewidth"] / 2
    
    # Figure 1: speed
    fig1, ax1 = plt.subplots()
    ax1.plot(t, speed, color="tab:purple",lw=lw)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(MultipleLocator(300))
    ax1.grid()
    if phase is not None:
    
        ax1.set_ylim(top=speed.max() * 1.15) 
        names, first = np.unique(phase, return_index=True)  # names of the phases, in the order of the cycle
        names = names[np.argsort(first)]
    
    for name in names:
        t_phase = t[phase == name]
        if t_phase.min() > t.min():
            ax1.axvline(t_phase.min(), color="black", ls="--", lw=lw)
        ax1.text(t_phase.mean(), speed.max() * 1.07, name, 
                 ha="center", va="center", fontweight="bold",
                 fontsize=plt.rcParams["legend.fontsize"])
    
    # Figure 2: battery power
    P_kW = P_bat * 1e-3
    
    fig2, ax2 = plt.subplots()
    ax2.plot(t, np.where(P_kW > 0, P_kW, 0), color="tab:purple", lw=lw,
             label=r"$\mathcal{P}_\mathsf{bat}^\mathsf{out}$")
    ax2.plot(t, np.where(P_kW < 0, P_kW, 0), color="tab:cyan", lw=lw,
             label=r"$\mathcal{P}_\mathsf{bat}^\mathsf{in}$")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Battery power (kW)")
    ax2.set_xlim(t.min(), t.max())
    ax2.xaxis.set_major_locator(MultipleLocator(300))
    ax2.grid()
    ax2.set_ylim(np.floor(P_kW.min() / 10) * 10, np.ceil(P_kW.max() / 10) * 10)
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.legend(loc="upper left")
    
    if export_path:
        path = Path(export_path)
        fig1.savefig(path.with_name(f"{path.stem}_speed{path.suffix}"), 
                     dpi=300, bbox_inches="tight")
        fig2.savefig(path.with_name(f"{path.stem}_power{path.suffix}"), 
                     dpi=300, bbox_inches="tight")
    
    return fig1, fig2


def plot_cycle_alpha(t, P_bats, alphas, xlim=None, t_mark=None, export_path=None):
    """
    Plots the battery power along a driving cycle for different braking 
    energy recovery coefficients alpha.
    t in (s), P_bats: list of battery powers (W), one per alpha
    xlim: time window to show (t_min, t_max) in (s), None = whole cycle
    t_mark: time (s) at which the recovered powers are marked with a dot
    """
    
    fig, ax = plt.subplots()
    
    # 1800 points: thinner lines than in the other plots
    lw = plt.rcParams["lines.linewidth"] / 3
    
    # Power drawn from the battery (same for all alpha)
    P_kW = P_bats[0] * 1e-3
    ax.plot(t, np.where(P_kW > 0, P_kW, 0), color="tab:purple", lw=lw,
            label=r"$\mathcal{P}_\mathsf{bat}^\mathsf{out}$")
    
    # Recovered power: largest alpha first, so that all the curves stay visible
    colors = ["tab:red", "tab:orange", "tab:cyan"]
    
    for P_bat, alpha, color in list(zip(P_bats, alphas, colors))[::-1]:
        P_kW = P_bat * 1e-3
        ax.plot(t, np.where(P_kW < 0, P_kW, 0), color=color, lw=lw,
                label=rf"$\mathcal{{P}}_\mathsf{{bat}}^\mathsf{{in}}$: $\alpha$ = {alpha}")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Battery power (kW)")
    
    # Time window, and the part of the powers that is visible in it
    # Time window, and the part of the powers that is visible in it
    if xlim is None:
        xlim = (t.min(), t.max())
        ax.xaxis.set_major_locator(MultipleLocator(300))
    else:
        # graduations every 50 s for a short window, every 100 s for a longer one
        step = 50 if xlim[1] - xlim[0] <= 300 else 100
        ax.xaxis.set_major_locator(MultipleLocator(step))
    
    ax.set_xlim(xlim)
    visible = (t >= xlim[0]) & (t <= xlim[1])
    P_visible = np.array(P_bats)[:, visible] * 1e-3
    
    # y axis: just above the highest visible power, and a small margin below 
    # the lowest one (graduations every 10 kW: the last one written is -10)
    y_top = np.ceil(P_visible.max() / 10) * 10
    y_bottom = P_visible.min() - 2
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid()
    
    # legend: P_out, then the recovered powers by increasing alpha
    handles, labels = ax.get_legend_handles_labels()
    order = [0] + list(range(len(handles) - 1, 0, -1))
    handles = [handles[k] for k in order]
    labels = [labels[k] for k in order]
    
    # Recovered powers at t_mark: one coloured dot per curve, 
    # with its value in the second column of the legend
    if t_mark is not None:
        k = np.argmin(np.abs(t - t_mark))
        
        # title of the second column (no symbol in front of it)
        handles.append(Line2D([], [], linestyle=""))
        labels.append(f"At t = {t[k]:.0f} s:")
        
        for P_bat, color in zip(P_bats, colors):
            dot = ax.scatter(t[k], P_bat[k] * 1e-3, color=color, zorder=4,
                             s=(plt.rcParams["lines.markersize"] / 2.5) ** 2)
            handles.append(dot)
            labels.append(f"{P_bat[k] * 1e-3:.1f} kW")
    
    # Same trick as the other plots to keep the axes size
    w, h = fig.get_size_inches()
    
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.get_layout_engine().set(h_pad=0.2)
    ax.set_box_aspect(h / w)
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig




def plot_consumption_weather(speed, consumptions, labels, points=None, ymax=None, export_path=None):
    """
    Plots the energy consumption at constant speed against the speed,
    for different weather conditions (accessories power).
    speed in (km/h), consumptions: list of curves in (kWh/km), one per label
    points: speeds (km/h) to highlight on each curve
    ymax: upper limit of the consumption axis (None = automatic)
    """
    
    fig, ax = plt.subplots()
    
    colors = ["tab:purple", "tab:cyan", "tab:orange"]
    
    for E, label, color in zip(consumptions, labels, colors):
        
        ax.plot(speed, E, color=color, label=label)
        
        if points is not None:
            ax.scatter(points, np.interp(points, speed, E), zorder=3, color="red",
                       label="30, 50, 90 and 120 km/h" if label == labels[-1] else None)
    
    # Style
    ax.set_xlabel(r"Speed (km/h)")
    ax.set_ylabel(r"Energy consumption (kWh/km)")
    ax.grid()
    
    # Same trick as the other plots to keep the axes size
    w, h = fig.get_size_inches()
    
    fig.legend(loc="outside lower center", ncol=2)
    fig.get_layout_engine().set(h_pad=0.2)
    
    ax.set_box_aspect(h / w)
    
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, speed.max())
    ticks = np.arange(0, speed.max() + 1, 15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    
    if export_path:
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
    
    return fig




