#!/usr/bin/env python3
"""
run_all.py — regenerate ROI masks (if needed) and all four paper-figure
replications (Figs 1, 3, 4, 6) in one shot.

Usage
-----
    python -m scripts.edema_figures.run_all
"""

from . import roi_space, flair_space, fig1_slices, fig3_boxplots, fig4_methods, fig6_bars


def main():
    print("== resampling ROI masks ==")
    roi_space.resample_all()
    print("== resampling FLAIR ==")
    flair_space.resample_all()
    print("== figure 1 ==")
    fig1_slices.main()
    print("== figure 3 ==")
    fig3_boxplots.main()
    print("== figure 4 ==")
    fig4_methods.main()
    print("== figure 6 ==")
    fig6_bars.main()


if __name__ == "__main__":
    main()
