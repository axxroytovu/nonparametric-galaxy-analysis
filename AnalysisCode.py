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
from terminaltables import AsciiTable
from matplotlib.colors import LogNorm
import pandas as pd
from tqdm import tqdm
import glob
from datetime import datetime
from random import sample

def normalize(L):
	return L/np.max(L)

def reject_outliers(L, m=2):
	d = np.abs(L - np.median(L))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return L[s<m]

def gini(L, graph=False):
	D = L.copy()
	D.mask = ma.nomask
	SKY = mode(D, axis = None)[0][0]
	L = ma.compressed(L)
	SKY = np.min(L)
	L -= SKY
	L.sort()
	XbarABS = np.mean(np.abs(L))
	Xbar = np.mean(L)
	n = len(L)
	SUMMER = L * (2*(np.array(range(n))+1)-n-1)
	SUMMERCORR = np.abs(L) * (2*(np.array(range(n))+1)-n-1)
	if graph:
		plt.figure()
		xx = [float(n)/(len(L)-1) for n in xrange(len(L))]
		plt.fill_between(xx, 0, np.cumsum(np.abs(L))/np.sum(np.abs(L)), facecolor = 'g')
		plt.fill_between(xx, xx, np.cumsum(np.abs(L))/np.sum(np.abs(L)), facecolor = 'r')
		plt.ylim(0,1)
		plt.grid()
		ax = plt.gca()
		plt.xlabel('Population Percentage')
		plt.ylabel('Cumulative Intensity Percentage')
		plt.title('Gini Coefficient')
		ax.set_aspect('equal')
		plt.savefig('GiniExample.png', format='png', dpi = 300)
		plt.close()
	return np.sum(SUMMERCORR)/(abs(XbarABS)*n*(n-1))

def M20(D, graph=False):
	Nx, Ny = D.shape
	XX, YY = np.meshgrid(range(Nx), range(Ny))
	XX = np.compress(D.mask.flatten()==False, XX)
	YY = np.compress(D.mask.flatten()==False, YY)
	L = ma.compressed(D)
	F = zip(L, XX.flatten(), YY.flatten())
	F.sort()
	F = np.array(F[::-1])
	#print F[:10]
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
			IMG[y,x] = 1.
		plt.imshow(ma.MaskedArray(np.ones(IMG.shape), mask=1.-IMG), cmap='seismic', interpolation='nearest', vmin=-2, vmax=2)
		plt.imshow(D.data, interpolation = 'nearest', cmap='gray_r', alpha=0.8)
		plt.xlim(20, Nx-20)
		plt.ylim(20, Ny-20)
		plt.xticks([])
		plt.title('M20')
		plt.yticks([])
		plt.savefig('M20Example.png', format='png', dpi=300)
		plt.close()
	return np.log(Mag/Mtot)/np.log(10)

def Asymmetry(D, rp, graph=False):
	Nx, Ny = D.shape
	Size = min(Nx, Ny)
	if Nx != Ny:
		D = D[:Size,:Size]
	XX, YY = [X.astype('float') for X in np.meshgrid(range(Size), range(Size))]
	XX -= Size/2.
	YY -= Size/2.
	Distance = np.sqrt(XX**2 + YY**2)
	MASK = D.mask
	Bdat = D.data
	Background = ma.filled(ma.MaskedArray(Bdat, mask = Distance < rp*1.5), 0.)
	Exterior = reject_outliers(ma.compressed(Background))
	SKY = np.mean(Exterior)
	STD = np.std(Exterior)
	#print SKY, STD
	L = D.data #np.where(1-D.mask, D.data, normal(SKY, STD, D.shape))
	BG180 = 0#(np.sum(abs(Background - np.rot90(Background, k=2))))/np.sum(abs(Background))
	BG90 = 0#(np.sum(abs(Background - np.rot90(Background))))/np.sum(abs(Background))
	Mag = np.sum(abs(L))
	R2 = np.rot90(L, k=2)
	Roll = [-int(Size/2), -int(Size/2)]
	R2 = np.roll(R2, Roll[0], axis = 0)
	R2 = np.roll(R2, Roll[1], axis = 1)
	Asym = np.zeros([Size, Size, 3])
	for x in xrange(Size):
		for y in xrange(Size):
			UU = ma.MaskedArray(abs(R2-L), mask=(Distance > rp*1.5))
			Asym[x,y,0] = (np.sum(UU))/Mag-BG180
			Asym[x,y,1] = Roll[0]/2.
			Asym[x,y,2] = Roll[1]/2.
			Roll[1] += 1
			R2 = np.roll(R2, 1, axis = 1)
		Roll[0] += 1
		Roll[1] = -int(Size/2)
		R2 = np.roll(R2, 1, axis = 0)
	if graph:
		plt.figure()
		plt.imshow(Asym[:,:,0], vmin = 0, vmax = 2)
		plt.colorbar()
		plt.savefig('AysmmetryDiscuss.png', format='png', dpi=300)
		plt.close()
	UU = list(Asym.reshape(-1, 3))
	A180, Cx, Cy = sorted(UU, key = lambda x: x[0])[0]
	R2 = np.rot90(L)
	Roll = [-int(Size/2), -int(Size/2)]
	R2 = np.roll(R2, Roll[0], axis = 0)
	R2 = np.roll(R2, Roll[1], axis = 1)
	Asym = np.zeros([Size, Size, 3])
	for x in xrange(Size):
		for y in xrange(Size):
			UU = ma.MaskedArray(abs(R2-L), mask=(Distance > rp*1.5))
			Asym[x,y,0] = (np.sum(abs(R2-L)))/Mag-BG90
			Asym[x,y,1] = Roll[0]/2.
			Asym[x,y,2] = Roll[1]/2.
			Roll[1] += 1
			R2 = np.roll(R2, 1, axis = 1)
		Roll[0] += 1
		Roll[1] = -int(Size/2)
		R2 = np.roll(R2, 1, axis = 0)
	UU = list(Asym.reshape(-1, 3))
	A90, _, _ = sorted(UU, key = lambda x: x[0])[0]
	if graph:
		plt.figure()
		U = ma.MaskedArray(D.data, mask=ma.nomask)
		Roll = np.roll(np.roll(np.rot90(U, k=2), int(Cx*2), axis=0), int(Cy*2), axis=1)
		plt.imshow(U - Roll, cmap = 'gray_r', interpolation = 'nearest')
		Circle = plt.Circle((Size/2., Size/2.), rp*1.5, color='r', fill=False)
		ax = plt.gca()
		ax.add_artist(Circle)
		plt.xlim(20, Nx-20)
		plt.ylim(20, Ny-20)
		plt.xticks([])
		plt.yticks([])
		#plt.colorbar()
		plt.title('Asymmetry')# '+str(np.sum(np.abs(L-Roll))))
		plt.savefig("AsymmetryExample.png", format='png', dpi= 300)
		plt.close()
	return A90, A180
	
def PetrosianRad(D, graph=False):
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
	L = D.copy()
	MASK = np.array(L.mask, copy=True)
	Ny, Nx = L.shape
	XX, YY = np.meshgrid(range(Nx), range(Ny))
	Distance = np.sqrt(np.square(XX-(Nx-1)/2.) + np.square(YY-(Ny-1)/2.))
	L.mask = ma.nomask
	Csize = int(rp/4)
	Boxcar = np.ones((Csize,Csize))/(Csize**2)
	Smoothed = convolve2d(L, Boxcar, 'same')
	MASK = (Distance < rp/4) + (Distance > rp*1.5)
	L = ma.masked_array(L, MASK)
	Smoothed = ma.masked_array(Smoothed, MASK)
	if graph:
		plt.figure()
		C1 = plt.Circle((Nx/2., Ny/2.), rp*0.25, color='g', fill=False)
		C2 = plt.Circle((Nx/2., Ny/2.), rp*1.5, color='g', fill=False)
		plt.imshow(L.data-Smoothed.data, cmap='gray_r', interpolation='nearest')
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
	N = D.count()
	Ny, Nx = D.shape
	RawData = ma.compressed(D)
	QuartileArray = [np.percentile(RawData, p) for p in np.linspace(1,100,201)]
	L = ma.filled(D, 0.)
	Rmax = 0
	for PCT in QuartileArray:
		IMG = L >= PCT
		Labels = label(IMG, output = 'int', structure = [[1,1,1],[1,1,1],[1,1,1]])[0]
		Sizes = np.sort(np.bincount(Labels.flatten()))
		if len(Sizes) > 2:
			Rmax = max(Rmax, Sizes[-3]**2/float(Sizes[-2]))
			if graph and (Rmax == Sizes[-3]**2/float(Sizes[-2])):
				plt.figure()
				plt.imshow(D.data, cmap='gray_r', interpolation='nearest')
				plt.imshow(Labels, interpolation='nearest', alpha =0.3, cmap='cubehelix_r')
				plt.xlim(20, Nx-20)
				plt.ylim(20, Ny-20)
				plt.yticks([])
				plt.xticks([])
				plt.title('Multimode')
				plt.savefig('MultimodeExample.png', format='png', dpi=300)
				plt.close()
	return Rmax

def Intensity(L, graph=False):
	Ny, Nx = L.shape
	threshold = abs(min(ma.compressed(L)))/2.
	RawData = ma.filled(L,0)
	#plt.imshow(RawData)
	#plt.show()
	Filtered = gaussian_filter(RawData, 1)
	#plt.imshow(-Filtered, cmap = plt.cm.jet)
	#plt.show()
	data_max = peak_local_max(Filtered, indices=False, min_distance = 2, threshold_abs=threshold)
	DataIndices = peak_local_max(Filtered, indices=True, min_distance = 2, threshold_abs=threshold)
	markers = ndi.label(data_max)[0]
	#plt.imshow(markers)
	#plt.show()
	Regions = watershed(-Filtered, markers, mask = np.logical_not(L.mask))
	Intensities = []
	if len(np.unique(Regions)) == 2:
		return DataIndices[0], 0
	if graph:
		plt.figure()
		plt.imshow(L.data, cmap='gray_r', interpolation='nearest')
		plt.imshow(ma.MaskedArray(Regions, mask=L.mask), cmap = 'cubehelix_r', interpolation ='nearest', alpha=0.3, vmin=0)
		plt.xticks([])
		plt.yticks([])
		plt.xlim(20, Nx-20)
		plt.ylim(20, Ny-20)
		plt.title('Intensity')
		plt.savefig('IntensityExample.png', format='png', dpi=300)
		plt.close()
	for J in np.unique(Regions)[1:]:
		Med = ma.masked_array(L, Regions != J)
		Intensities.append( np.sum(Med) )
	Intensity, Index = zip(*sorted(zip(Intensities, range(1,len(np.unique(Regions))))))
	
	return DataIndices[Index[-1]-1], Intensity[-2]/Intensity[-1]
	
def Deviation(L, MaxCoord, graph=False):
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

def SingleGalaxy(GalaxyNumber):
	DataDirectory = r'/Users/samzimmerman/Documents/Capstone/Data/GDS_deep10_13/'


	filenames = ['GDS_deep10_'+str(GalaxyNumber)+'_h.fits',
		'GDS_deep10_'+str(GalaxyNumber)+'_j.fits',
		'GDS_deep10_'+str(GalaxyNumber)+'_v.fits',
		'GDS_deep10_'+str(GalaxyNumber)+'_z.fits']
	
	Segmap = 'GDS_deep10_'+str(GalaxyNumber)+'_segmap.fits'

	hdulist = fits.open(DataDirectory+Segmap)
	segarray = np.array(hdulist[0].data)
	MASK = ()


	hdulist = fits.open(DataDirectory + filenames[0])
	CImage1 = np.array(hdulist[0].data)
	hdulist.close()
	hdulist = fits.open(DataDirectory + filenames[1])
	CImage2 = np.array(hdulist[0].data)
	hdulist.close()
	hdulist = fits.open(DataDirectory + filenames[2])
	CImage3 = np.array(hdulist[0].data)
	hdulist.close()
	CImage = np.dstack([CImage1, CImage2, CImage3])
	Vmin = np.amin(CImage)
	Vmax = np.amax(CImage)
	#SCALE = lambda arr: (arr-Vmin)/(Vmax-Vmin)
	SCALE = lambda arr: np.log((np.exp(1)-1)*(arr-Vmin)/(Vmax-Vmin) + 1)
	CImage = SCALE(CImage)
	#plt.imshow(CImage, interpolation = 'nearest')
	#plt.show()
	
	plt.figure()
	Ny, Nx = CImage1.shape
	plt.imshow(CImage1, interpolation='nearest', cmap='gray_r')
	plt.xlim(20, Nx-20)
	plt.ylim(20, Ny-20)
	plt.xticks([])
	plt.yticks([])
	plt.title("GOODS South Deep 10: "+str(GalaxyNumber))
	plt.savefig('RawImage.png', format='png', dpi=300)
	plt.close()

	Data = [["Filename", "Petrosian Radius", "Gini Coefficient", "M20", "Asymmetry 90", "Asymmetry 180", "Concentration", "Clumpiness", "Multimode", "Intensity", "Deviation"]]
	Graph = True
	for f in filenames:
		hdulist = fits.open(DataDirectory+f)
		#print hdulist.info()
		AllDat = ma.masked_array(np.array(hdulist[0].data), segarray != GalaxyNumber)
		#print AllDat[0:10]
		rp = PetrosianRad(AllDat)
		MaxCoord, I = Intensity(AllDat, graph=Graph)
		A1, A2 = Asymmetry(AllDat, rp, graph=Graph)
		Data.append([ f, str(rp), str(gini(AllDat, graph=Graph)), str(M20(AllDat, graph=Graph)), str(A1), str(A2), str(Concentration(AllDat, rp, graph=Graph)), str(Clumpiness(AllDat, rp, graph=Graph)), str(Multimode(AllDat, graph=Graph)), str(I), str(Deviation(AllDat, MaxCoord, graph=Graph))])
		hdulist.close()
		Graph = False
	print AsciiTable(Data).table

def BuildDatabase(subset = False):
	DataDirectory = r'/Users/samzimmerman/Documents/Capstone/Data/GDS_deep10_13/'
	
	AllData = pd.read_csv('/Users/samzimmerman/Documents/Capstone/Data/CombinedData.csv')

	AllData['Size'] = pd.Series(AllData['SN_H']*5/.06, index = AllData.index)
	
	CombDat = pd.DataFrame(columns = ('ID_2', 'Size', 'f_Spheroid', 'f_Disk', 'f_Irr', 'f_PS', 'Rpet_cir_H', 'C_H', 'A_H', 'S_H', 'G_H', 'M20_H', 'M_H', 'I_H', 'D_H', 'Rpet_cir_J', 'C_J', 'A_J', 'S_J', 'G_J', 'M20_J', 'M_J', 'I_J', 'D_J', 'Rph', 'Ch', 'A90h', 'A180h', 'Sh', 'Gh', 'M20h', 'Mh', 'Ih', 'Dh'))#, 'Rpj', 'Cj', 'A90j', 'A180j', 'Sj', 'Gj', 'M20j', 'Mj', 'Ij', 'Dj', 'Rpv', 'Cv', 'A90v', 'A180v', 'Sv', 'Gv', 'M20v', 'Mv', 'Iv', 'Dv', 'Rpz', 'Cz', 'A90z', 'A180z', 'Sz', 'Gz', 'M20z', 'Mz', 'Iz', 'Dz'))
	
	
	AllData.set_index(["ID_2"], inplace = True, drop=False)
	Files = glob.glob(DataDirectory + "*_segmap.fits")
	if subset:
		Files = sample(Files, 200)
	for GN in tqdm(Files):
		GalName = GN[len(DataDirectory):-12]
		if GalName not in AllData.index or (AllData.loc[GalName]['SN_H'] < 3):
			with open('log.log', 'a') as logfile:
				logfile.write("NO DATA: "+GalName+"\n")
			continue
		BaseFile = GN[:-12]
		try:
			Data = AllData.loc[GalName]
			hdulist = fits.open(GN)
			StartDat = [Data[k] for k in CombDat.columns[:24]]
			segarray = np.array(hdulist[0].data)
			hdulist.close()
			
			hdulist = fits.open(BaseFile + '_h.fits')
			HData = ma.masked_array(np.array(hdulist[0].data), segarray != int(GalName[11:16]))
			#plt.imshow(HData)
			#plt.show()
			RPet = PetrosianRad(HData)
			StartDat.append(RPet*0.06)
			StartDat.append(Concentration(HData, RPet))
			A90h, A180h = Asymmetry(HData, RPet)
			StartDat.append(A90h)
			StartDat.append(A180h)
			StartDat.append(Clumpiness(HData, RPet))
			StartDat.append(gini(HData))
			StartDat.append(M20(HData))
			StartDat.append(Multimode(HData))
			HData = ma.masked_array(np.array(hdulist[0].data), segarray != int(GalName[11:16]))
			MCH, Ih = Intensity(HData)
			StartDat.append(Ih)
			StartDat.append(Deviation(HData, MCH))
			#print len(StartDat)
			CombDat.loc[GalName] = StartDat
		except Exception as ERR:
			_, _, tb = sys.exc_info()
			print traceback.format_list(traceback.extract_tb(tb)[-1:])[-1]
			with open('log.log', 'a') as logfile:
				logfile.write(str(ERR))
			try:
				hdulist.close()
			except:
				pass
			print ERR
			pass
	
	Spheroids = CombDat[CombDat['f_Spheroid'] > 0.6]
	Disks = CombDat[CombDat['f_Disk'] > 0.6]
	Irregular = CombDat[CombDat['f_Irr'] > 0.6]
	PointSource = CombDat[CombDat['f_PS'] > 0.6]
	
	header = fits.Header()
	header['DATETIME'] = str(datetime.now())
	header['NAME'] = 'Sam Zimmerman'
	header['MORE'] = 'More Comments'
	
	hdu = fits.PrimaryHDU(header=header)
	FullHDU = fits.table_to_hdu(Table.from_pandas(CombDat))
	SphereHDU = fits.table_to_hdu(Table.from_pandas(Spheroids))
	DiskHDU = fits.table_to_hdu(Table.from_pandas(Disks))
	IrrHDU = fits.table_to_hdu(Table.from_pandas(Irregular))
	PSHDU = fits.table_to_hdu(Table.from_pandas(PointSource))
	hdulist = fits.HDUList([hdu, FullHDU, SphereHDU, DiskHDU, IrrHDU, PSHDU])
	
	hdulist.writeto('Full.fits', clobber=True)
	hdulist.close()
	
def CompareResults():
	
	CombDat = Table.read('Full.fits', format='fits', hdu=1).to_pandas()
	Spheroids = Table.read('Full.fits', format='fits', hdu=2).to_pandas()
	Disks = Table.read('Full.fits', format='fits', hdu=3).to_pandas()
	Irregular = Table.read('Full.fits', format='fits', hdu=4).to_pandas()
	PointSource = Table.read('Full.fits', format='fits', hdu=5).to_pandas()
	
	plt.scatter(Spheroids['Rpet_cir_H'], Spheroids['Rph'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['Rpet_cir_H'], Disks['Rph'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['Rpet_cir_H'], Irregular['Rph'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['Rpet_cir_H'], PointSource['Rph'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Rph'], CombDat['Rpet_cir_H'])
	plt.title("H band Petrosian Radius")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc='best')
	plt.savefig("PetrosianRadComparison.png", format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['C_H'][Spheroids['C_H'] != -99], Spheroids['Ch'][Spheroids['C_H'] >= -90], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['C_H'][Disks['C_H'] >= -90], Disks['Ch'][Disks['C_H'] != -99], c = 'g', label = 'Disks')
	plt.scatter(Irregular['C_H'][Irregular['C_H'] >= -90], Irregular['Ch'][Irregular['C_H'] >= -90], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['C_H'][PointSource['C_H'] >= -90], PointSource['Ch'][PointSource['C_H'] >= -90], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Ch'], CombDat['C_H'])
	plt.title("H band Concentration")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.legend(loc = 'best')
	plt.ylabel('Calculated')
	plt.savefig('ConcentrationComparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['A_H'][Spheroids['A_H'] >= -90], Spheroids['A180h'][Spheroids['A_H'] >= -90], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['A_H'][Disks['A_H'] >= -90], Disks['A180h'][Disks['A_H'] >= -90], c = 'g', label = 'Disks')
	plt.scatter(Irregular['A_H'][Irregular['A_H'] >= -90], Irregular['A180h'][Irregular['A_H'] >= -90], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['A_H'][PointSource['A_H'] >= -90], PointSource['A180h'][PointSource['A_H'] >= -90], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['A180h'], CombDat['A_H'])
	plt.title("H band Asymmetry")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.legend(loc='best')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.savefig('AsymmetryComparison.png', format='png', dpi=300)
	plt.close()
	
	
	plt.scatter(Spheroids['S_H'][Spheroids['S_H'] >= -90], Spheroids['Sh'][Spheroids['S_H'] >= -90], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['S_H'][Disks['S_H'] >= -90], Disks['Sh'][Disks['S_H'] >= -90], c = 'g', label = 'Disks')
	plt.scatter(Irregular['S_H'][Irregular['S_H'] >= -90], Irregular['Sh'][Irregular['S_H'] >= -90], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['S_H'][PointSource['S_H'] >= -90], PointSource['Sh'][PointSource['S_H'] >= -90], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Sh'], CombDat['S_H'])
	plt.title("H band Clumpiness")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('ClumpinessComparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['G_H'], Spheroids['Gh'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['G_H'], Disks['Gh'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['G_H'], Irregular['Gh'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['G_H'], PointSource['Gh'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Gh'], CombDat['G_H'])
	plt.title("H band Gini Coefficient")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('GiniComparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['M20_H'], Spheroids['M20h'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['M20_H'], Disks['M20h'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['M20_H'], Irregular['M20h'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['M20_H'], PointSource['M20h'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['M20h'], CombDat['M20_H'])
	plt.title("H band M20")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('M20Comparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['I_H'], Spheroids['Ih'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['I_H'], Disks['Ih'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['I_H'], Irregular['Ih'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['I_H'], PointSource['Ih'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Ih'], CombDat['I_H'])
	plt.title("H band Intensity")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('IntensityComparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['M_H'], Spheroids['Mh'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['M_H'], Disks['Mh'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['M_H'], Irregular['Mh'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['M_H'], PointSource['Mh'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Mh'], CombDat['M_H'])
	plt.title("H band Multimode")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('MultimodeComparison.png', format='png', dpi=300)
	plt.close()
	
	plt.scatter(Spheroids['D_H'], Spheroids['Dh'], c = 'b', label = 'Spheroids')
	plt.scatter(Disks['D_H'], Disks['Dh'], c = 'g', label = 'Disks')
	plt.scatter(Irregular['D_H'], Irregular['Dh'], c = 'r', label = 'Irregulars')
	plt.scatter(PointSource['D_H'], PointSource['Dh'], c = 'k', label = 'Point Sources')
	plt.grid()
	C, P = pearsonr(CombDat['Dh'], CombDat['D_H'])
	plt.title("H band Deviation")
	plt.axis('equal')
	limx = plt.xlim()
	limy = plt.ylim()
	plt.plot([-100,100],[-100,100], 'g--')
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel('Nominal')
	plt.ylabel('Calculated')
	plt.legend(loc = 'best')
	plt.savefig('DeviationComparison.png', format='png', dpi=300)
	plt.close()
	
def Compare2():
	
	CombDat = Table.read('Full.fits', format='fits', hdu=1).to_pandas()
	Spheroids = Table.read('Full.fits', format='fits', hdu=2).to_pandas()
	Disks = Table.read('Full.fits', format='fits', hdu=3).to_pandas()
	Irregular = Table.read('Full.fits', format='fits', hdu=4).to_pandas()
	PointSource = Table.read('Full.fits', format='fits', hdu=5).to_pandas()
	
	f, (a1, a2)  = plt.subplots(2,1)
	
	XX, YY = np.mgrid[-3.:0.:100j, 0.:1.:100j]
	positions = np.vstack([XX.ravel(), YY.ravel()])
	SpherGM = gaussian_kde(np.vstack([Spheroids['M20_H'], Spheroids['I_H']]))
	DiskGM = gaussian_kde(np.vstack([Disks['M20_H'], Disks['I_H']]))
	IrrGM = gaussian_kde(np.vstack([Irregular['M20_H'], Irregular['I_H']]))
	
	Extent = [np.min(Spheroids['M20_H']), np.max(Spheroids['M20_H']), np.min(Spheroids['I_H']), np.max(Spheroids['I_H'])]
	
	Sdata = normalize(np.reshape(SpherGM(positions).T, XX.shape))
	Ddata = normalize(np.reshape(DiskGM(positions).T, XX.shape))
	Idata = normalize(np.reshape(IrrGM(positions).T, XX.shape))
	COLOR = np.dstack([np.rot90(Idata), np.rot90(Ddata), np.rot90(Sdata)])
	a1.imshow(COLOR, extent = Extent)
	a1.set_title('their stuff')
	SpherGM = gaussian_kde(np.vstack([Spheroids['M20h'], Spheroids['Ih']]))
	DiskGM = gaussian_kde(np.vstack([Disks['M20h'], Disks['Ih']]))
	IrrGM = gaussian_kde(np.vstack([Irregular['M20h'], Irregular['Ih']]))
	
	Sdata = normalize(np.reshape(SpherGM(positions).T, XX.shape))
	Ddata = normalize(np.reshape(DiskGM(positions).T, XX.shape))
	Idata = normalize(np.reshape(IrrGM(positions).T, XX.shape))
	
	COLOR = np.dstack([np.rot90(Idata), np.rot90(Ddata), np.rot90(Sdata)])
	
	a2.imshow(COLOR, extent = Extent)
	a2.set_title('My stuff')
	#plt.scatter(Spheroids['M20_H'], Spheroids['G_H'], c = 'b', label = 'Spheroids')
	#plt.scatter(Disks['M20_H'], Disks['G_H'], c = 'g', label = 'Spheroids')
	#plt.scatter(Irregular['M20_H'], Irregular['G_H'], c = 'r', label = 'Spheroids')
	plt.show()
	
def Compare2():
	
	CombDat = Table.read('Full.fits', format='fits', hdu=1).to_pandas()
	Spheroids = Table.read('Full.fits', format='fits', hdu=2).to_pandas()
	Disks = Table.read('Full.fits', format='fits', hdu=3).to_pandas()
	Irregular = Table.read('Full.fits', format='fits', hdu=4).to_pandas()
	PointSource = Table.read('Full.fits', format='fits', hdu=5).to_pandas()
	
	f, ((a1, a2), (a3, a4))  = plt.subplots(2,2)
	
	XX, YY = np.mgrid[-3.:0.:100j, 0.:1.:100j]
	positions = np.vstack([XX.ravel(), YY.ravel()])
	SpherGM = gaussian_kde(np.vstack([Spheroids['M20_H'], Spheroids['G_H']]))
	DiskGM = gaussian_kde(np.vstack([Disks['M20_H'], Disks['G_H']]))
	IrrGM = gaussian_kde(np.vstack([Irregular['M20_H'], Irregular['G_H']]))
	
	Extent = [np.min(Spheroids['M20_H']), np.max(Spheroids['M20_H']), np.min(Spheroids['G_H']), np.max(Spheroids['G_H'])]
	
	Sdata = normalize(np.reshape(SpherGM(positions).T, XX.shape))
	Ddata = normalize(np.reshape(DiskGM(positions).T, XX.shape))
	Idata = normalize(np.reshape(IrrGM(positions).T, XX.shape))
	COLOR = np.dstack([np.rot90(Idata), np.rot90(Ddata), np.rot90(Sdata)])
	a1.imshow(COLOR, extent = Extent)
	a1.set_title('their stuff')
	SpherGM = gaussian_kde(np.vstack([Spheroids['M20h'], Spheroids['Gh']]))
	DiskGM = gaussian_kde(np.vstack([Disks['M20h'], Disks['Gh']]))
	IrrGM = gaussian_kde(np.vstack([Irregular['M20h'], Irregular['Gh']]))
	
	Sdata = normalize(np.reshape(SpherGM(positions).T, XX.shape))
	Ddata = normalize(np.reshape(DiskGM(positions).T, XX.shape))
	Idata = normalize(np.reshape(IrrGM(positions).T, XX.shape))
	
	COLOR = np.dstack([np.rot90(Idata), np.rot90(Ddata), np.rot90(Sdata)])
	
	a2.imshow(COLOR, extent = Extent)
	a2.set_title('My stuff')
	#plt.scatter(Spheroids['M20_H'], Spheroids['G_H'], c = 'b', label = 'Spheroids')
	#plt.scatter(Disks['M20_H'], Disks['G_H'], c = 'g', label = 'Spheroids')
	#plt.scatter(Irregular['M20_H'], Irregular['G_H'], c = 'r', label = 'Spheroids')
	plt.show()
	
	
	
SingleGalaxy(11647)#12228
#BuildDatabase(subset = False)
#CompareResults()
#Compare2()