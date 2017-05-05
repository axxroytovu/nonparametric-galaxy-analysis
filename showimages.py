import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
from tqdm import tqdm
from textwrap import wrap

for image in tqdm(glob("*.fits")):
	hdu = fits.open(image)
	fig, (a1, a2, a3) = plt.subplots(1,3)
	cax = fig.add_axes([0.2, 0.2, 0.6, 0.04])
	Max = np.max(hdu[1].data)
	U = a1.imshow(hdu[1].data, vmin = 0, vmax = Max*0.5, cmap='gray_r')
	a2.imshow(hdu[2].data, vmin = 0, vmax = Max*0.5, cmap='gray_r')
	a3.imshow(np.absolute(hdu[3].data), vmin = 0, vmax = Max*0.5, cmap='gray_r')
	a1.tick_params(
    	axis='both',          # changes apply to the x-axis
    	which='both',      # both major and minor ticks are affected
    	bottom='off',      # ticks along the bottom edge are off
    	top='off',         # ticks along the top edge are off
    	left='off',
    	right='off',
    	labelleft='off',
    	labelbottom='off') # labels along the bottom edge are off
	a2.tick_params(
    	axis='both',          # changes apply to the x-axis
    	which='both',      # both major and minor ticks are affected
    	bottom='off',      # ticks along the bottom edge are off
    	top='off',         # ticks along the top edge are off
    	left='off',
    	right='off',
    	labelleft='off',
    	labelbottom='off') # labels along the bottom edge are off
	a3.tick_params(
    	axis='both',          # changes apply to the x-axis
    	which='both',      # both major and minor ticks are affected
    	bottom='off',      # ticks along the bottom edge are off
    	top='off',         # ticks along the top edge are off
    	left='off',
    	right='off',
    	labelleft='off',
    	labelbottom='off') # labels along the bottom edge are off
	u = False
	n = 0
	items = []
	for x, y in hdu[2].header.iteritems():
		if u:
			if x[0] == n:
			    if x[-1] == 'N':
				    items[-1] = items[-1] + "{0}: {1}".format(x, y)
			else:
				u = False
				n = 0
		if y == 'sersic':
			u = True
			n = x[-1]
			items.append('')
	a2.set_title(image+"\n"+", ".join(items))
	cbar = fig.colorbar(U, cax, orientation='horizontal')  # horizontal colorbar
	plt.savefig(image[:-4]+'png', format='png', dpi=300)
	plt.close()
	hdu.close()