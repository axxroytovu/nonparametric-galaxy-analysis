from astropy.io import fits
import pandas as pd
import numpy as np
import csv
import statmorph as stat
from glob import glob
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt


def Clean(S):
    return float(S.translate(None, "[]*"))

AllData = pd.read_csv("/Users/samzimmerman/Documents/Capstone/Data/"
                      "COSAllGalfit/AllCOSData.csv", index_col=0)

Results = {'h': [], 'v': [], 'z': [], 'j': []}
# format: Dc, Dn, C
# Dc = distance between component centers / mean RE
# Dn = difference between sersic indices
# C = classification "color"

ResFile = "Results.csv"
WriteFile = open(ResFile, 'w')
fieldnames = ['GalName', 'FileIndex', 'Dc', 'Dn', 'Band', 'Class']
W = csv.writer(WriteFile)
W.writerow(fieldnames)

for File in tqdm(glob("/Users/samzimmerman/Documents/Capstone/"
                      "Sam_Jeyhan_Sample/**/*_gf_changed_content.fits")):
    DATAHduList = fits.open(File)
    RA = (DATAHduList[0].header["CRVAL1"] -
          DATAHduList[0].header['CRPIX1']*DATAHduList[0].header['CDELT1'])
    DEC = (DATAHduList[0].header["CRVAL2"] -
           DATAHduList[0].header['CRPIX2']*DATAHduList[0].header['CDELT2'])
    # print RA, DEC
    VAL = (np.sqrt((AllData['RA_2']-RA)**2 + (AllData['DEC_2']-DEC)**2))
    # print VAL
    df_closerow = AllData.ix[VAL.abs().argsort()[0]]
    # print df_closerow['Index']
    # input()
    # Order: V:4, Z:5, J:6, H:7
    if df_closerow["HbandPhotometry"]:
        InfoHDU = DATAHduList[7]
        C = 'h'
    elif df_closerow["VbandPhotometry"]:
        InfoHDU = DATAHduList[4]
        C = 'v'
    elif df_closerow["ZbandPhotometry"]:
        InfoHDU = DATAHduList[5]
        C = 'z'
    else:
        InfoHDU = DATAHduList[6]
        C = 'j'
    Info = []  # Mag, Xcoord, Ycoord, N, RE
    for i in count(1):
        if "COMP_{:d}".format(i) in InfoHDU.header:
            # print True
            if InfoHDU.header["COMP_{:d}".format(i)] == 'sersic':
                # print True
                Info.append([Clean(InfoHDU.header["{:d}_MAG_0".format(i)]),
                             Clean(InfoHDU.header["{:d}_XC_0".format(i)]),
                             Clean(InfoHDU.header["{:d}_YC_0".format(i)]),
                             Clean(InfoHDU.header["{:d}_N_0".format(i)]),
                             Clean(InfoHDU.header["{:d}_RE_0".format(i)])])
        else:
            break
    Info.sort()
    if len(Info) > 1:
        Dc = (np.sqrt((Info[0][1]-Info[1][1])**2 +
              (Info[0][2]-Info[1][2])**2) / (min(Info[0][4], Info[1][4])))
        Dn = abs(Info[1][3] - Info[0][3])
        Class = [df_closerow["f_Spheroid"] > 0.6, df_closerow["f_Disk"] > 0.6,
                 df_closerow["f_Irr"] > 0.6, df_closerow["f_any"] > 0.5]
        W.writerow([df_closerow.name, File[57:-24], Dc, Dn, C,
                    ''.join(np.where(Class, ['s', 'd', 'i', 'm'], ''))])
        if Class == [1, 1, 1, 1]:
            Class == [0, 0, 0, 1]
        Results[C].append([Dc, Dn, Class])
    DATAHduList.close()
    # print Results
plt.figure(1)
RotResults = zip(*Results["h"])
plt.scatter(RotResults[0], RotResults[1], c=RotResults[2], s=50)
plt.title('H band data')
# plt.figure(2)
RotResults = zip(*Results["v"])
plt.scatter(RotResults[0], RotResults[1], c=RotResults[2], s=50)
plt.title('V band data')
# plt.figure(3)
RotResults = zip(*Results["z"])
plt.scatter(RotResults[0], RotResults[1], c=RotResults[2], s=50)
plt.title('Z band data')
# plt.figure(4)
RotResults = zip(*Results["j"])
plt.scatter(RotResults[0], RotResults[1], c=RotResults[2], s=50)
plt.title('J band data')
for idx in [1]:
    plt.figure(idx)
    n = plt.plot([], [], 'o', c=[0, 0, 0], label='all')
    i = plt.plot([], [], 'o', c=[0, 0, 1], label='irr')
    d = plt.plot([], [], 'o', c=[0, 1, 0], label='disk')
    s = plt.plot([], [], 'o', c=[1, 0, 0], label='sphr')
    sd = plt.plot([], [], 'o', c=[1, 1, 0], label='sphr/disk')
    si = plt.plot([], [], 'o', c=[1, 0, 1], label='sphr/irr')
    di = plt.plot([], [], 'o', c=[0, 1, 1], label='disk/irr')
    sdi = plt.plot([], [], 'o', c=[1, 1, 1], label='none/no int')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Normalized center distance")
    plt.ylabel("Sersic Index difference")
WriteFile.close()
plt.show()
