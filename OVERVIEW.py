from astropy.io import fits
import pandas as pd
import numpy as np
import csv
import statmorph as stat
from glob import glob
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.neighbors import NearestNeighbors
import sys
import os


def Clean(S):
    return float(S.translate(None, "[]*"))


def CropImg(Img, Shape, Scale1, Scale2):
    x, y = Img.shape
    xmin, xmax = (x-Shape[0])/2, (x+Shape[0])/2
    ymin, ymax = (y-Shape[1])/2, (y+Shape[0])/2
    return Img[xmin:xmax, ymin:ymax]


class YNbutton(object):
    keep = False
    def YES(self, event):
        plt.close()  
        self.keep = True
    def NO(self, event):
        plt.close()

AllData = pd.read_csv("/Users/samzimmerman/Documents/Capstone/Data/"
                      "COSAllGalfit/AllCOSData.csv", index_col=0)
SourceExtractor = pd.read_csv("/Users/samzimmerman/Documents/SourceTree/"
                              "SourceExtractor.csv", index_col=None)

Results = {'h': [], 'v': [], 'z': [], 'j': []}
# format: Dc, Dn, C
# Dc = distance between component centers / mean RE
# Dn = difference between sersic indices
# C = classification "color"

ResFile = "Results.csv"
WriteFile = open(ResFile, 'w')
fieldnames = ['GalName', 'FileIndex', 'Band', 'Class', 'Mag', 'G_r', 'RPcirc1', 'RPcirc2', 'SN', 'RPellip', 'Asym1', 'Asym2', 'XC1', 'YC1', 'XC2', 'YC2', 'Gini', 'M20', 'Success']
W = csv.writer(WriteFile)
W.writerow(fieldnames)

MatchedData = pd.read_csv("/Users/samzimmerman/Documents/SourceTree/MatchedPairs.csv")

for id, Row in tqdm(list(MatchedData.iterrows())):
    DATAHduList = fits.open("/Users/samzimmerman/Documents/Capstone/"
                      "Sam_Jeyhan_Sample/"+Row['FileID']+"_gf_changed_content.fits")
    # print VAL

    if Row['VbandPhotometry']:
        DatHDR = DATAHduList[0].header
        Res = DATAHduList[8]  # 8
        Band = 'V'
    elif Row['ZbandPhotometry']:
        DatHDR = DATAHduList[1].header
        Res = DATAHduList[9]  # 9
        Band = 'Z'
    elif Row['JbandPhotometry']:
        DatHDR = DATAHduList[2].header
        Res = DATAHduList[10]  # 10
        Band = 'J'
    else:
        DatHDR = DATAHduList[3].header
        Res = DATAHduList[11]  # 11
        Band = 'H'
    classify = []
    if Row['f_Spheroid'] > 0.6:
        classify.append('s')
    if Row['f_Disk'] > 0.6:
        classify.append('d')
    if Row['f_Irr'] > 0.6:
        classify.append('i')
    if (Row['f_merger'] + Row['f_Int1'] + Row['f_Int2'])> 0.6:
        classify.append('m')
    classify = ''.join(classify)
    DatXX, DatYY = np.meshgrid(DatHDR['CRVAL1'] - (DatHDR['CRPIX1'] -
                               np.arange(0, DatHDR['NAXIS1'])) *
                               DatHDR['CDELT1'],
                               DatHDR['CRVAL2'] - (DatHDR['CRPIX2'] -
                               np.arange(0, DatHDR['NAXIS2'])) *
                               DatHDR['CDELT2'])
    try:
        SegHDUlist = fits.open("/Volumes/Untitled/Capstone Data/UpdatedSegmaps/"+Row['ID']+"_segmap.fits")
        SegHDU = SegHDUlist[0]
        if SegHDU.data.shape != Res.data.shape:
            raise Exception()
    except:
        SegHDUlist = fits.open("/Volumes/Untitled/Capstone Data/COSMOS/"+Row['ID']+"_segmap.fits")
        SegHDR = SegHDUlist[0].header
        NewSegmap = np.zeros(Res.data.shape)
        SegXX, SegYY = np.meshgrid(SegHDR['CRVAL1'] - (SegHDR['CRPIX1'] -
                                   np.arange(0,SegHDR['NAXIS1'])) *
                                   SegHDR['CD1_1'],
                                   SegHDR['CRVAL2'] - (SegHDR['CRPIX2'] -
                                   np.arange(0,SegHDR['NAXIS2'])) *
                                   SegHDR['CD2_2'])
        SegXF = np.reshape(SegXX, SegXX.size)
        SegYF = np.reshape(SegYY, SegYY.size)
        SegDT = np.reshape(SegHDUlist[0].data, SegHDUlist[0].data.size)
        Data1 = np.vstack([SegXF, SegYF])
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm='ball_tree').fit(Data1.T)
        nbrz = lambda x, y: SegDT[nbrs.kneighbors([[x,y]])[1][0]]
        vnbrz = np.vectorize(nbrz)
        Nseg = []
        for R1, R2 in tqdm(zip(DatXX, DatYY)):
            Nseg.append(vnbrz(R1, R2))
        NewSegmap = np.array(Nseg)
        SegHDU = fits.PrimaryHDU(NewSegmap)
        SegHDU.writeto("/Volumes/Untitled/Capstone Data/UpdatedSegmaps/"+Row['ID']+"_segmap.fits", clobber=True)
        

    '''
    plt.imshow(NewSegmap)
    plt.figure()
    plt.imshow(SegHDUlist[0].data)
    plt.show()
    '''
    Number = int(Row['ID'][4:])
    SECATALOG = SourceExtractor.ix[Number-1]
    SECATALOG['CD1_1'] = DatHDR['CD1_1']
    #print Row['FileID']
    #print SegHDU.header
    sys.stdout = open(os.devnull, "w")
    #if np.sum(Res.data) < 0:
    #Res.data = np.abs(Res.data)*(SegHDU.data == SECATALOG['NUMBER']) + (SegHDU.data != SECATALOG['NUMBER'])*Res.data
    #plt.imshow(Res.data)
    #plt.show()
    morph_hdu, rpa_seg_hdu, gdOBJ = stat.morph_from_stamp(Res, SegHDU, SECATALOG)
    sys.stdout = sys.__stdout__
    Data = [Row['ID'], Row['FileID'], Band, classify, SECATALOG['MAG_AUTO']]
    Success = True
    for attribute in ['residual_gini', 'rp_circ_1', 'rp_circ_2', 'snpix_init', 'rp_ellip', 'asym1', 'asym2', 'xcen_a1', 'ycen_a1', 'xcen_a2', 'ycen_a2', 'gini', 'm20']:
        try:
            Data.append(getattr(gdOBJ, attribute))
        except AttributeError:
            Data.append(None)
            Success = False
    Data.append(Success)
    W.writerow(Data)
    DATAHduList.close()
    SegHDUlist.close()
    #break
    
WriteFile.close()
