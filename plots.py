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


def plot_topspeed(speed, eta_tr, P_max, wheel_power, resistive_power, under_redline, k_oc, v_max, highlight, export_path=None):
    
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

    
    # v_max
    y_limb, y_limt = ax.get_ylim()
    ax.axvline(v_max, ymax=(eta_tr * P_max * 1e-3)/y_limt, ls="dotted", color="red", 
               label=r"$v_\mathsf{{max}}$" if highlight=="peak" else None, alpha=alpha_peak)
    ax.axvline(v_max, ymax=(eta_tr * P_max * 1e-3 / k_oc)/y_limt, ls="dotted", color="red", 
               label=r"$v_\mathsf{{max}}$" if highlight=="nominal" else None, alpha=alpha_nom)
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
    
    base = [t for t in ax.get_xticks()
            if 0 <= t <= xmax and all((abs(t - v) > 0.05 * xmax) for v in v_maxs)]
    xtk = sorted(base + v_maxs)
    
    is_special = lambda t: np.isclose(t, v_maxs).any()
    
    ax.set_xticks(xtk)
    ax.set_xticklabels([f"{t:.1f}" if is_special(t) else f"{t:.0f}" for t in xtk])
    
    for tl, t in zip(ax.get_xticklabels(), xtk):
        if np.isclose(t, v_max):
            tl.set_color("red")
            tl.set_fontweight("bold")
        elif np.isclose(t, v_maxmax_peak):
            tl.set_color("purple")
            tl.set_fontweight("bold")
            tl.set_alpha(alpha_peak)
        elif np.isclose(t, v_maxmax_nom):
            tl.set_color("purple")
            tl.set_fontweight("bold")
            tl.set_alpha(alpha_nom)
                    
    ax.set_xlim(0, speed.max())
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Power (kW)")
                    
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    
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
        base = [t for t in ax.get_yticks()
                if 0 <= t <= ymax and all(abs(t - v) > 0.05 * ymax for v in trac.values())]
        ytk = sorted(base + list(trac.values()))
        ax.set_yticks(ytk)
        ax.set_yticklabels([f"{t:.1f}" for t in ytk])
        for tl, t in zip(ax.get_yticklabels(), ytk):
            for p, v in trac.items():
                if np.isclose(t, v):
                    tl.set_color(colors[p])
                    tl.set_fontweight("bold")

        ax.set_ylabel("Max slope (%)")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        handles = [Patch(facecolor="lightgray", edgecolor="gray",
                         label=r"Max slope limited by adherence $\theta_\mathsf{{max}}^{(a)}$")]
        handles += [Patch(facecolor=colors[p], label=rf"Effective max slope $\theta_\mathsf{{max}}$ ({p})")
                    for p in powers]
        handles += [Line2D([0], [0], color=colors[p], ls="--",
                           label=rf"Max slope limited by traction $\theta_\mathsf{{max}}^{{(t)}}$ ({p})")
                    for p in powers]
        handles += [Line2D([], [], linestyle="", label="(adh) or (trac) = effective limiting factor")]

        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=True)

        if export_path:
            path = Path(export_path)
            fig.savefig(path.with_name(f"{path.stem}_{s}kmh{path.suffix}"),
                        bbox_inches="tight")

        figs.append(fig)

    plt.show()
    return figs