"""
MADI-GPU — Metabolic Activity Diffusion Imaging (GPU-accelerated)
=================================================================
Numba CUDA Monte Carlo simulation of water diffusion in contracted
Voronoi cell ensembles with semipermeable membranes.

Tailored to multi-delta SDE acquisitions (e.g. preclinical 9.4T).

Reference:
    Springer et al., NMR in Biomedicine 2023;36:e4781
"""

from . import config, ensemble, walker_gpu, signal, plotting
