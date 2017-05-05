import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
from numpy.random import normal
import numpy.ma as ma
from scipy.signal import convolve2d
import pandas as pd

def reject_outliers(L, m=2):
    d = np.abs(L - np.median(L))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return L[s<m]

Log = "Log.log"

#"/Volumes/TOSHIBA EXT/CapstoneData/ERSTotal/ERS_*_?.fits" 43 4:-7
#"/Users/samzimmerman/Documents/Capstone/Data/Goods-South-Deep10/GDS_deep10_*_?.fits" 63 11:-7

AllData = pd.read_csv("/Users/samzimmerman/GitHub/GalMorphology/Results.csv")

for image in tqdm(glob.glob("/Volumes/Untitled/Capstone Data/COSMOS/COS_*_?.fits")):
    try:
        if image[39:-7] not in list(AllData.GalName):
            continue
        #print image[39:]
        INAME = image[39:]
        with open(Log, 'a') as LogFile:
            LogFile.write("Starting Image "+INAME)
        hdulist = fits.open(image)
        Data = hdulist[0].data
        #N = hdulist[0].header['NDRIZIM']
        '''
        fig, ax = plt.subplots(1,1)
        n, bins, _ = ax.hist(Data.flatten(), bins=100)
        SKY = bins[np.argmax(n)]
        plt.close()
        '''
        segmap = fits.open("/Volumes/Untitled/Capstone Data/COSMOS/"+INAME[:-7]+"_segmap.fits")
        GalNum = int(INAME[4:-7])
        SegmData = segmap[0].data
        if SegmData.shape != Data.shape:
            while SegmData.shape[0] < Data.shape[0]:
                SegmData = np.pad(SegmData, [(1,1), (0,0)], "constant", constant_values=0)
            while SegmData.shape[1] < Data.shape[1]:
                SegmData = np.pad(SegmData, [(0,0), (1,1)], "constant", constant_values=0)
        Exterior = ma.compressed(ma.MaskedArray(Data, mask=(SegmData==GalNum)))
        Exterior = reject_outliers(Exterior)
        SKY = np.mean(Exterior)
        STD = 0#np.std(Exterior)
        SegData1 = np.where(SegmData == GalNum, Data, 
            SKY*np.ones(Data.shape)
        )
        SegData2 = convolve2d(SegData1, np.ones((10, 10))/100., 'same')
        SegData = np.where(SegmData == GalNum, Data, SegData2)
        segmap.close()
        '''
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(Data, cmap="gray_r")
        C = ax2.imshow(SegData, cmap="gray_r")
        Ax = f.add_axes([0.1, 0.1, 0.8, 0.04])
        f.colorbar(C, Ax, orientation='horizontal')
        plt.savefig("image/"+INAME[:-5]+".png", format = 'png', dpi = 300)
        plt.close()
        '''
        hdulist[0].data = SegData
        hdulist.writeto("/Users/samzimmerman/Documents/Capstone/Data/COSAllGalfit/Cleaned/C_"+INAME, clobber = True, output_verify="silentfix")
        hdulist.close()
        with open(Log, 'a') as LogFile:
            LogFile.write("; Finished\n")
    except:
        try:
            with open(Log, 'a') as LogFile:
                LogFile.write("; **FAILED**\n")
            hdulist.close()
            segmap.close()
        except:
            pass
    