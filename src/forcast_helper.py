from src.buildnet import *
import matplotlib.pyplot as plt 


def recover_rnn_and_cnn(mansave_dir, savenetworkname,
		ncell, imagex, imagey, FLAGS,
		plot_self_postspike_filters=False):

	learnable_weights = np.load(savenetworkname+'.npy', allow_pickle = True)
	inputnet, rnn = build_model(ncell, imagex, imagey, FLAGS) # thr=0.4 if no buffer
	rnn.cell.input_weights = learnable_weights[0]
	rnn.cell.recurrent_weights = learnable_weights[1]
	rnn.cell.bias = learnable_weights[2]
	inputnet.conv2a.set_weights([learnable_weights[3], learnable_weights[4]])
	inputnet.conv2b.set_weights([learnable_weights[5], learnable_weights[6]])

	if plot_self_postspike_filters:
		nrow, ncol = 5,10
		fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True)
		for ci in range(nrow):
			for cj in range(ncol):
				c = ci*ncol+cj
				axes[ci][cj].plot(learnable_weights[1][c,c,:])

		plt.savefig(os.path.join(mansave_dir, 'self_postspike_filter.png'))
		plt.close()

	return inputnet, rnn


def forecast_on_gt(data, FLAGS,
		inputnet, rnn,
		train=True, eval=False,
		plot_hidden_neuron_spikes=True):
	
	input_delay, spike_delay = data.numconsframes, FLAGS.spike_delay
	# nrep = data.train_numrep
	ncell = data.numcell
	numtevolvetrain, numtevolveeval = (data.train_numimg-input_delay+1), (data.eval_numimg-input_delay+1)
	input_delay, spike_delay = data.numconsframes, FLAGS.spike_delay

	if train:
		x = data.xstimuli()
		nrep = data.train_numrep
		predact = np.zeros((nrep, ncell, numtevolvetrain), dtype='bool')
		predfr = np.zeros((nrep, ncell, numtevolvetrain))
		gtact =  np.zeros((nrep, ncell, numtevolvetrain), dtype='bool')
	else:
		x = data.xstimuli_eval()
		nrep = data.eval_numrep
		predact = np.zeros((nrep, ncell, numtevolveeval), dtype='bool')
		predfr = np.zeros((nrep, ncell, numtevolveeval))
		gtact =  np.zeros((nrep, ncell, numtevolveeval), dtype='bool')


	if train:
		y_batch, pasty_batch = data.yresponse_and_spikehistory__train(nrep, input_delay, spike_delay)
	else: 
		y_batch, pasty_batch = data.yresponse_and_spikehistory__eval(nrep, input_delay, spike_delay)

	input_for_rnn = tf.repeat(tf.expand_dims(inputnet(x),axis=0),nrep,axis=0) # batsize, nframe, input_latent_size
	z, v, v_scaled = rnn(tf.concat((input_for_rnn, pasty_batch), axis=2))


	# compute the classification loss
	class_loss, spike_probability, _ = classification_loss(tf.slice(v_scaled,(0,0,0), (nrep,v_scaled.shape[1],ncell)), 
												y_batch, FLAGS.temperature_parameter)

	gtact[:, :, :] = np.moveaxis(y_batch,1,-1).astype('bool')
	predfr[:,:,:] = np.moveaxis(spike_probability[:,:,:ncell],-1,1)
	predact[:,:,:] = np.moveaxis(z[:,:,:ncell],-1,1)


	print(class_loss)
	
	return gtact, predact, predfr, class_loss



def raster_plot(ax,spikes,dt=0.04,time_line=None,linewidth=0.8,**kwargs):
    n_t,n_n = spikes.shape
    event_time_ids,event_ids = np.where(spikes)
    max_spike = 100000
    event_time_ids = event_time_ids[:max_spike]
    event_ids = event_ids[:max_spike]
    for n,t_id in zip(event_ids,event_time_ids):
        t = t_id if time_line is None else time_line[t_id]
        t = t*dt
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)
    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t*0.04])
    ax.set_yticks([0, n_n])
