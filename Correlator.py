"""
Build CSV file to correlate Megamorph objects with Galfit objects
"""

import csv
from astropy.io import fits
from glob import glob
from tqdm import tqdm

WriteFile = open("CorrelateResults.csv", 'w')
WW = csv.writer(WriteFile)
WW.writerow(['FileID', 'RA', 'DEC'])

for File in tqdm(glob("/Users/samzimmerman/Documents/Capstone/"
                      "Sam_Jeyhan_Sample/**/*_gf_changed_content.fits")):
    DATAHduList = fits.open(File)
    Hdr = DATAHduList[0].header
    RA = (Hdr['CRVAL1'] - (Hdr['CRPIX1'] -
          float(Hdr['NAXIS1'])/2)*Hdr['CDELT1'])
    DEC = (Hdr['CRVAL2'] - (Hdr['CRPIX2'] -
           float(Hdr['NAXIS2'])/2)*Hdr['CDELT2'])
    WW.writerow([File[57:-24], RA, DEC])
    DATAHduList.close()

WriteFile.close()
