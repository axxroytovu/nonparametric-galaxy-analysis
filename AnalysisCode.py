from astropy.io import fits
from astropy.table import Table
from ftplib import FTP
import sys
import traceback
import aplpy
import numpy as np
import numpy.ma as ma
from numpy.random import normal
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools as itr
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.stats import chisquare, mode, pearsonr, gaussian_kde
from matplotlib.colors import LogNorm


def normalize(L):
    """
    Simple function to normalize the list L.
    Scales the list to be between 0 and 1.
    """
    return (L-np.min(L))/(np.max(L)-np.min(L))


def reject_outliers(L, m=2):
    """
    Function to reject outliers beyond a specific value.
    Takes in a list L and removes all values that are more than m median
    deviations away from the median.
    
    The median is used to try and remove as many outliers as possible.
    """
    d = np.abs(L - np.median(L))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return L[s < m]


def gini(L, graph=False):
    """
    Calculates the Gini coefficient of a masked array L.
    Calculates a sky value from the rest of the image.
    graph allows you to plot the result of the analysis in a nice way.
    """
    D = L.copy()
    D.mask = ma.nomask
    SKY = mode(D, axis=None)[0][0]
    L = ma.compressed(L)
    SKY = np.min(L)
    L -= SKY
    L.sort()
    XbarABS = np.sum(np.abs(L))
    Xbar = np.mean(L)
    n = len(L)
    SUMMER = L * (2*(np.array(range(n)))-n-1)
    SUMMERCORR = np.abs(L) * (2*(np.array(range(n)))-n-1)
    if graph:
        plt.figure()
        xx = [float(n)/(len(L)-1) for n in xrange(len(L))]
        plt.fill_between(xx, 0, np.cumsum(np.abs(L))/np.sum(np.abs(L)),
                         facecolor='g')
        plt.fill_between(xx, xx, np.cumsum(np.abs(L))/np.sum(np.abs(L)),
                         facecolor='r')
        plt.ylim(0, 1)
        plt.grid()
        ax = plt.gca()
        plt.xlabel('Population Percentage')
        plt.ylabel('Cumulative Intensity Percentage')
        plt.title('Gini Coefficient')
        ax.set_aspect('equal')
        plt.savefig('GiniExample.png', format='png', dpi=300)
        plt.close()
    return np.sum(SUMMERCORR)/(abs(XbarABS)*n*(n-1))


def M20(D, graph=False):
    """
    Calculates the 20% moment of area of the masked array D.
    graph plots the image with the 20% brightest pixels highlighted.
    
    More correlation work needs to be done.
    """
    Nx, Ny = D.shape
    XX, YY = np.meshgrid(range(Nx), range(Ny))
    XX = np.compress(D.mask.flatten() == 0, XX)
    YY = np.compress(D.mask.flatten() == 0, YY)
    L = ma.compressed(D)
    F = zip(L, XX.flatten(), YY.flatten())
    F.sort()
    F = np.array(F[::-1])
    # print F[:10]
    Xc = Nx/2.
    Yc = Ny/2.
    SumVal = []
    for f, x, y in F:
        SumVal.append(f*((x-Xc)**2 + (y-Yc)**2))
    Mtot = np.sum(SumVal)
    CSf = np.cumsum([x[0] for x in F])
    CSf /= CSf[-1]
    Mag = 0
    i = 0
    List = []
    while CSf[i] < .2:
        Mag += F[i][0]*((F[i][1]-Xc)**2 + (F[i][2]-Yc)**2)
        List.append(F[i][1:])
        i += 1
    if graph:
        plt.figure()
        IMG = np.zeros(D.shape)
        for x, y in List:
            IMG[y, x] = 1.
        plt.imshow(ma.MaskedArray(np.ones(IMG.shape), mask=1.-IMG),
                   cmap='seismic', interpolation='nearest', vmin=-2, vmax=2)
        plt.imshow(D.data, interpolation='nearest', cmap='gray_r', alpha=0.8)
        plt.xlim(20, Nx-20)
        plt.ylim(20, Ny-20)
        plt.xticks([])
        plt.title('M20')
        plt.yticks([])
        plt.savefig('M20Example.png', format='png', dpi=300)
        plt.close()
    return np.log(Mag/Mtot)/np.log(10)


def Asymmetry(D, rp, graph=False):
    """
    Calculates the asymmetry of the image.
    D is a masked array of the galaxy.
    rp is the Petrosian radius (circular for now)
    graph plots the difference between the image and the 180 degree rotation
    along with a circle indicating 1.5 petrosian radii.
    """
    Nx, Ny = D.shape
    Size = min(Nx, Ny)
    if Nx != Ny:
        D = D[:Size, :Size]
    XX, YY = [X.astype('float') for X in np.meshgrid(range(Size), range(Size))]
    XX -= Size/2.
    YY -= Size/2.
    Distance = np.sqrt(XX**2 + YY**2)
    MASK = D.mask
    Bdat = D.data
    Background = ma.filled(ma.MaskedArray(Bdat, mask=Distance < rp*1.5), 0.)
    Exterior = reject_outliers(ma.compressed(Background))
    SKY = np.mean(Exterior)
    STD = np.std(Exterior)
    # print SKY, STD
    L = D.data  # np.where(1-D.mask, D.data, normal(SKY, STD, D.shape))
    BG180 = 0
    # or (np.sum(abs(Background - np.rot90(Background,
    #     k=2))))/np.sum(abs(Background))
    BG90 = 0
    # or (np.sum(abs(Background -
    #     np.rot90(Background))))/np.sum(abs(Background))
    Mag = np.sum(abs(L))
    R2 = np.rot90(L, k=2)
    Roll = [-int(Size/2), -int(Size/2)]
    R2 = np.roll(R2, Roll[0], axis=0)
    R2 = np.roll(R2, Roll[1], axis=1)
    Asym = np.zeros([Size, Size, 3])
    for x in xrange(Size):
        for y in xrange(Size):
            UU = ma.MaskedArray(abs(R2-L), mask=(Distance > rp*1.5))
            Asym[x, y, 0] = (np.sum(UU))/Mag-BG180
            Asym[x, y, 1] = Roll[0]/2.
            Asym[x, y, 2] = Roll[1]/2.
            Roll[1] += 1
            R2 = np.roll(R2, 1, axis=1)
        Roll[0] += 1
        Roll[1] = -int(Size/2)
        R2 = np.roll(R2, 1, axis=0)
    if graph:
        plt.figure()
        plt.imshow(np.abs(Asym[:, :, 0]), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig('AysmmetryDiscuss.png', format='png', dpi=300)
        plt.close()
    UU = list(Asym.reshape(-1, 3))
    A180, Cx, Cy = sorted(UU, key=lambda x: x[0])[0]
    R2 = np.rot90(L)
    Roll = [-int(Size/2), -int(Size/2)]
    R2 = np.roll(R2, Roll[0], axis=0)
    R2 = np.roll(R2, Roll[1], axis=1)
    Asym = np.zeros([Size, Size, 3])
    for x in xrange(Size):
        for y in xrange(Size):
            UU = ma.MaskedArray(abs(R2-L), mask=(Distance > rp*1.5))
            Asym[x, y, 0] = (np.sum(abs(R2-L)))/Mag-BG90
            Asym[x, y, 1] = Roll[0]/2.
            Asym[x, y, 2] = Roll[1]/2.
            Roll[1] += 1
            R2 = np.roll(R2, 1, axis=1)
        Roll[0] += 1
        Roll[1] = -int(Size/2)
        R2 = np.roll(R2, 1, axis=0)
    UU = list(Asym.reshape(-1, 3))
    A90, _, _ = sorted(UU, key=lambda x: x[0])[0]
    if graph:
        plt.figure()
        U = ma.MaskedArray(D.data, mask=ma.nomask)
        Roll = np.roll(np.roll(np.rot90(U, k=2), int(Cx*2), axis=0),
                       int(Cy*2), axis=1)
        plt.imshow(np.abs(U - Roll), cmap='gray_r', interpolation='nearest')
        Circle = plt.Circle((Size/2., Size/2.), rp*1.5, color='r', fill=False)
        ax = plt.gca()
        ax.add_artist(Circle)
        plt.xlim(20, Nx-20)
        plt.ylim(20, Ny-20)
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.title('Asymmetry')  # '+str(np.sum(np.abs(L-Roll))))
        plt.savefig("AsymmetryExample.png", format='png', dpi=300)
        plt.close()
    return A90, A180


def PetrosianRad(D, graph=False):
    """
    Calculates the Petrosian radius of the masked array D
    graph does nothing, it's only there for compatibility purposes.
    """
    L = ma.filled(D, 0)
    Ny, Nx = L.shape
    XX, YY = np.meshgrid(range(Nx), range(Ny))
    Distance = np.sqrt(np.square(XX-(Nx-1)/2.) + np.square(YY-(Ny-1)/2.))
    AllRad = np.unique(Distance)
    for R in AllRad:
        u = np.mean(ma.masked_array(L, Distance != R))
        ubar = np.mean(ma.masked_array(L, Distance >= R))
        if ubar and u/ubar < 0.2:
            return R
    return R


def Concentration(L, rp, graph=False):
    """
    Calculates the Concentration from the masked array L and Petrosian
    radius rp.
    grpah plots the original image with two circles indicating the 20% and 80%
    radii.
    """
    Ny, Nx = L.shape
    XX, YY = np.meshgrid(range(Nx), range(Ny))
    Distance = np.sqrt(np.square(XX-(Nx-1)/2.) + np.square(YY-(Ny-1)/2.))
    AllRad = np.unique(Distance)
    TotFlux = np.sum(ma.masked_array(L, Distance >= rp*1.5))
    rad20 = 0
    rad80 = 0
    for R in AllRad:
        Flux = np.sum(ma.masked_array(L, Distance >= R))
        if Flux and (not rad20) and Flux/TotFlux >= 0.2:
            rad20 = R
        if Flux and (not rad80) and Flux/TotFlux >= 0.8:
            rad80 = R
        if rad80 and rad20:
            if graph:
                plt.figure()
                C80 = plt.Circle((Nx/2., Ny/2.), rad80, color='r', fill=False)
                C20 = plt.Circle((Nx/2., Ny/2.), rad20, color='g', fill=False)
                plt.imshow(L.data, cmap='gray_r', interpolation='nearest')
                ax = plt.gca()
                ax.add_artist(C80)
                ax.add_artist(C20)
                plt.xlim(20, Nx-20)
                plt.ylim(20, Ny-20)
                plt.yticks([])
                plt.xticks([])
                plt.legend([C20, C80], ['20% Radius', '80% Radius'])
                plt.title('Concentration')
                plt.savefig('ConcentrationExample.png', format='png', dpi=300)
                plt.close()
            return 5*np.log10(rad80/rad20)


def Clumpiness(D, rp, graph=False):
    """
    Calculates the clumpiness of the masked array D based on the Petrosian
    radius rp.
    Smoothing is done using a square boxcar filter of size rp/4.
    graph plots the difference between the smoothed image and the normal image,
    with circles indicating the rp/4 and 1.5*rp regions of integration.
    """
    L = D.copy()
    MASK = np.array(L.mask, copy=True)
    Ny, Nx = L.shape
    XX, YY = np.meshgrid(range(Nx), range(Ny))
    Distance = np.sqrt(np.square(XX-(Nx-1)/2.) + np.square(YY-(Ny-1)/2.))
    L.mask = ma.nomask
    Csize = int(rp/4)
    Boxcar = np.ones((Csize, Csize))/(Csize**2)
    Smoothed = convolve2d(L, Boxcar, 'same')
    MASK = (Distance < rp/4) + (Distance > rp*1.5)
    L = ma.masked_array(L, MASK)
    Smoothed = ma.masked_array(Smoothed, MASK)
    if graph:
        plt.figure()
        C1 = plt.Circle((Nx/2., Ny/2.), rp*0.25, color='g', fill=False)
        C2 = plt.Circle((Nx/2., Ny/2.), rp*1.5, color='g', fill=False)
        plt.imshow(L.data-Smoothed.data, cmap='gray_r',
                   interpolation='nearest')
        ax = plt.gca()
        ax.add_artist(C1)
        ax.add_artist(C2)
        plt.xlim(20, Nx-20)
        plt.ylim(20, Ny-20)
        plt.yticks([])
        plt.xticks([])
        plt.title('Clumpiness')
        plt.savefig('ClumpinessExample.png', format='png', dpi=300)
        plt.close()
    return np.sum(abs(L-Smoothed))/np.sum(abs(L))


def Multimode(D, graph=False):
    """
    Calculates the multimode of the masked array D. 
    graph plots the image with the differently colored regions on top of it.
    It's not super optimized, it makes a new plot every time it finds a new
    maximum multimode value.
    """
    N = D.count()
    Ny, Nx = D.shape
    RawData = ma.compressed(D)
    QuartileArray = [
        np.percentile(RawData, p) for p in np.linspace(1, 100, 201)
    ]
    L = ma.filled(D, 0.)
    Rmax = 0
    for PCT in QuartileArray:
        IMG = L >= PCT
        Labels = label(IMG, output='int',
                       structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])[0]
        Sizes = np.sort(np.bincount(Labels.flatten()))
        if len(Sizes) > 2:
            Rmax = max(Rmax, Sizes[-3]**2/float(Sizes[-2]))
            if graph and (Rmax == Sizes[-3]**2/float(Sizes[-2])):
                plt.figure()
                plt.imshow(D.data, cmap='gray_r', interpolation='nearest')
                plt.imshow(Labels, interpolation='nearest', alpha=0.3,
                           cmap='cubehelix_r')
                plt.xlim(20, Nx-20)
                plt.ylim(20, Ny-20)
                plt.yticks([])
                plt.xticks([])
                plt.title('Multimode')
                plt.savefig('MultimodeExample.png', format='png', dpi=300)
                plt.close()
    return Rmax


def Intensity(L, graph=False):
    """
    Finds the intensity value for a masked array L. 
    graph plots the results of the watershed segmentation on top of
    the raw image. 
    Need to do work to generate a zero-gradient analysis instead of watershed
    segmentation.
    """
    Ny, Nx = L.shape
    threshold = abs(min(ma.compressed(L)))/2.
    RawData = ma.filled(L, 0)
    # plt.imshow(RawData)
    # plt.show()
    Filtered = gaussian_filter(RawData, 1)
    # plt.imshow(-Filtered, cmap = plt.cm.jet)
    # plt.show()
    data_max = peak_local_max(Filtered, indices=False, min_distance=2,
                              threshold_abs=threshold)
    DataIndices = peak_local_max(Filtered, indices=True, min_distance=2,
                                 threshold_abs=threshold)
    markers = ndi.label(data_max)[0]
    # plt.imshow(markers)
    # plt.show()
    Regions = watershed(-Filtered, markers, mask=np.logical_not(L.mask))
    Intensities = []
    if len(np.unique(Regions)) == 2:
        return DataIndices[0], 0
    if graph:
        plt.figure()
        plt.imshow(L.data, cmap='gray_r', interpolation='nearest')
        plt.imshow(ma.MaskedArray(Regions, mask=L.mask), cmap='cubehelix_r',
                   interpolation='nearest', alpha=0.3, vmin=0)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(20, Nx-20)
        plt.ylim(20, Ny-20)
        plt.title('Intensity')
        plt.savefig('IntensityExample.png', format='png', dpi=300)
        plt.close()
    for J in np.unique(Regions)[1:]:
        Med = ma.masked_array(L, Regions != J)
        Intensities.append(np.sum(Med))
    Intensity, Index = zip(*sorted(zip(Intensities,
                                       range(1, len(np.unique(Regions))))))

    return DataIndices[Index[-1]-1], Intensity[-2]/Intensity[-1]


def Deviation(L, MaxCoord, graph=False):
    """
    Calculates the Deviation based on the masked array L and the
    coordinates of the largest maxima from the Intensity statistic.
    MaxCoord is he direct output from Intensity.
    Graph simply indicates the distance from the center of light to the
    maxima given by MaxCoord on top of the raw image data.
    """
    XX, YY = np.meshgrid(range(L.shape[1]), range(L.shape[0]))
    N = L.count()
    Ny, Nx = L.shape
    I = np.sum(L)
    CX = np.sum(XX * L)/I
    CY = np.sum(YY * L)/I
    if graph:
        plt.figure()
        plt.imshow(L.data, cmap='gray_r', interpolation='nearest')
        plt.plot([CX, MaxCoord[1]], [CY, MaxCoord[0]], 'ro-')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(20, Nx-20)
        plt.ylim(20, Ny-20)
        plt.title('Deviation')
        plt.savefig('DeviationExample.png', format='png', dpi=300)
        plt.close()
    return np.sqrt(np.pi/N)*np.sqrt((CX-MaxCoord[1])**2 + (CY-MaxCoord[0])**2)
