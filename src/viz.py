import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  


def viz_raster_plot(act, ax, args, nframefirst=True):
	# act: [nrep, nframe, ncell] or [nrep, ncell, nframe] => nrep*ncell, nframe

	if nframefirst:
		nrep, nframe, ncell = act.shape
		im = np.moveaxis(act, -1, 0).reshape((-1, nframe))
	else:
		nrep, ncell, nframe = act.shape
		im = np.array([act[:, ci, :]for ci in range(ncell)]).reshape((-1, nframe))

	ax.imshow(1-im, cmap='gray', vmax=1, vmin=0, aspect='auto')
	ax.set_xlabel('# frame')
	ax.set_ylabel(args['ylabel'])
	ax.set_title(args['title'])


def viz_psth_multiplots(nrows,ncols, psth, t, c, lbl, neurons, ax):
	for ri in range(nrows):
		for ci in range(ncols):
			n = ri*ncols+ci
			
			ax[ri][ci].plot(psth[n], color=c, label=lbl, linewidth=1)
			ax[ri][ci].axvline(x=t)

			ax[ri][ci].set_title('neuron%d'%(neurons[n]))


def mansavefig(avgbatchlist, evallist, step, savename_trainingcurve):
	plt.ioff()
	
	# minindex = 0
	minindex = np.argmin(np.asarray(evallist))
	xrange = max(step)
    
	textx = 100 + xrange/8
	if (textx<step[minindex]) & (step[minindex]<xrange/2):
		textx = xrange/12 + step[minindex]
	ymax = 0.3
	ymin = 0.1
	deltay = ymax-ymin
	figA, axA = plt.subplots()

	axA.plot(np.asarray(step), np.asarray(avgbatchlist), 'r', label='train loss')
	axA.set_ylim([ymin,ymax])
	axA.set_xlabel('step')
  	## Make the y-axis label, ticks and tick labels match the line color.
	axA.axvline(x=step[minindex], ymin=-.1, ymax = 2, linewidth=2, color='r')
	axA.set_ylabel('cross entropy loss', color='r')
	axA.tick_params('y', colors='r')
	axA.annotate('Step: %d, train: %.4f, eval: %.4f' % (step[minindex], avgbatchlist[minindex], evallist[minindex]), xy=(textx, 1.0*deltay+ymin))
	plt.legend(loc=3)
	axB = axA.twinx()
	axB.plot(np.asarray(step), np.asarray(evallist), 'k--', label='validation loss')
	axB.set_ylim([0.1,0.3])
	axB.set_ylabel('eval loss', color='k')
	axB.tick_params('y', colors='k')
	plt.legend(loc=4)
	# figA.tight_layout()
	plt.savefig(savename_trainingcurve)
	plt.close(figA)
	
