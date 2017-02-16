import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import glob
from datetime import datetime
from random import sample
import numpy as np
import numpy.ma as ma
from AnalysisCode import *
from terminaltables import AsciiTable


def SingleGalaxy(GalaxyNumber, DataDirectory):

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
	
def Compare2(DataFile):
	
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


DataDirectory = r'/Users/samzimmerman/Documents/Capstone/Data/Goods-South-Deep10/'
SingleGalaxy(11647, DataDirectory)#12228
#BuildDatabase(subset = False)
#CompareResults()
#Compare2()