'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

Module containing fixed constants for the tessilator

These are the pixel size, the typical full-width half maximum of the pixel
response function, the TESS zeropoints and the latest TESS sector available
for download.
'''

pixel_size=21.0
'''This is the pixel size for TESS (in arcseconds).

This is held constant. Do not change.
'''


exprf=0.65
'''This is the full-width half maximum of a TESS pixel.

This is held constant. Do not change.
'''


Zpt, eZpt = 20.44, 0.05
'''This is the zero-point TESS magnitude calculated in Vanderspek et al. 2018.

This is held constant. Do not change.
'''

sec_max = 74
'''The maximum sector number to be acquisitioned when looking for TESS data

This will change over time as more data is collected.
'''
