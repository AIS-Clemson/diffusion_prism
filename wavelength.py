# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 04:33:38 2023

@author: MaxGr
"""

import colour
import numpy as np

RGB_f = np.array([245.0, 122.0, 155.0]) / 255

# Using the individual definitions:
RGB_l = colour.models.eotf_sRGB(RGB_f)
XYZ = colour.RGB_to_XYZ(
    RGB_l,
    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
    colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
    colour.models.RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ,
)
xy = colour.XYZ_to_xy(XYZ)
wl, xy_1, xy_2 = colour.dominant_wavelength(
    xy, colour.models.RGB_COLOURSPACE_sRGB.whitepoint
)

# Using the automatic colour conversion graph:
wl, xy_1, xy_2 = colour.convert(RGB_f, "Output-Referred RGB", "Dominant Wavelength")

print(wl, xy_1, xy_2)

colour.plotting.colour_style()
figure, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(
    diagram_opacity=0.15, standalone=False
)

xy_i = np.vstack([xy_1, xy, colour.models.RGB_COLOURSPACE_sRGB.whitepoint, xy_2])
axes.plot(xy_i[..., 0], xy_i[..., 1], "-o")
colour.plotting.render()
























