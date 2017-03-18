"""
Build CSV file to correlate Megamorph objects with Galfit objects
"""

import csv
from astropy.io import fits
from glob import glob
from tqdm import tqdm

WriteFile = open("CorrelateResults.csv", 'w')
WW = csv.writer(WriteFile)
WW.write('FileID', 'RA', 'DEC')

for File in tqdm(glob("/Users/samzimmerman/Documents/Capstone/"
                      "Sam_Jeyhan_Sample/**/*_gf_changed_content.fits")):
    DataHDUL = fits.open(File)
    RA = (DATAHduList[0].header['CRVAL1'] -
          DATAHduList[0].header['CRPIX1']*DATAHduList[0].header['CDELT1'])
    DEC = (DATAHduList[0].header['CRVAL2'] -
           DATAHduList[0].header['CRPIX2']*DATAHduList[0].header['CDELT2'])
    WW.write(File[57:-24], RA, DEC)

WriteFile.close()