import numpy as np
import os, pdb
from skimage.transform import resize






class DataLoader(object):

	def get_psth(self, spike_trains, nrep):
		# spike_trains: 
		#	x: frames*repetitions
		# 	y: neurons
		psth = np.array([spike_trains[:,ci].reshape((nrep, -1)).mean(0) for ci in range(spike_trains.shape[1])])
		return psth

	def __init__(self, data_dir, train_filename, eval_filename, FLAGS=None):	
		
		train_activity = np.load(os.path.join(data_dir, train_filename[1]))  # [#rep * #frames, #neurons]
		train_image = np.load(os.path.join(data_dir, train_filename[0]))	 # [#y, #x, #frames]
		
		eval_activity = np.load(os.path.join(data_dir, eval_filename[1]))	# [#rep * #frames, #neurons]
		eval_image = np.load(os.path.join(data_dir, eval_filename[0]))		# [#y, #x, #frames]


		## parameters of data  
		numconsframes = max(FLAGS.numconsframes, FLAGS.spike_delay+1)
		numpixx, numpixy, train_numimg= train_image.shape # x is y on the image, y is x on the image
		_, _, eval_numimg = eval_image.shape
		numrep = int(train_activity.shape[0]/train_numimg)    
		eval_numrep = int(eval_activity.shape[0]/eval_numimg)  
		assert numrep == FLAGS.numrep
		numcell = train_activity.shape[1]


		numtrain = (train_numimg-numconsframes+1)*numrep
		numeval = (eval_numimg-numconsframes+1)*numrep

		## set the outputs
		self.numcell = numcell
		self.numpixx = numpixx
		self.numpixy = numpixy
		self.numtrain = numtrain
		self.numeval = numeval

		self.numconsframes = numconsframes
		self.train_numrep = numrep
		self.eval_numrep = eval_numrep
		self.train_numimg = train_numimg
		self.eval_numimg = eval_numimg

		self.train_activity = train_activity # (nrep*nframes) * numcells
		self.train_image = train_image
		self.train_psth = self.get_psth(train_activity, self.train_numrep)
		self.eval_activity = eval_activity
		self.eval_image = eval_image
		self.eval_psth = self.get_psth(eval_activity, self.eval_numimg)




	def xstimuli(self):
		IDbatch = np.array(range(self.numconsframes, self.train_numimg+1))
		xbatch = np.array([self.train_image[:,:,fi-self.numconsframes:fi] for fi in IDbatch])
		return xbatch

	def xstimuli_eval(self):
		IDbatch = np.array(range(self.numconsframes, self.eval_numimg+1))
		xbatch = np.array([self.eval_image[:,:,fi-self.numconsframes:fi] for fi in IDbatch])
		return xbatch

	def yresponse_and_spikehistory__train(self, numbatch_rep, input_delay, spike_delay):

		y_train__whole = self._yresponse_nbatch_train__whole(numbatch_rep) # batch_rep_size, ncell, nframe

		cut = max(input_delay-1, spike_delay)
		y_batchrep_train = np.array([y_train__whole[ri,:,cut:].T for ri in range(numbatch_rep)], dtype='float32') # [batch_rep_size, nframe, ncell]

		pasty_batchrep_train = np.array([y_train__whole[:, :, fi-spike_delay:fi].reshape((numbatch_rep, -1)) for fi in range(cut, self.train_numimg)]) # [nframe, batch_rep_size, ncell*spike_delay]
		pasty_batchrep_train = np.moveaxis(pasty_batchrep_train, 1, 0) # [batch_rep_size, nframe, ncell*spike_delay]
		pasty_batchrep_train = pasty_batchrep_train.astype('float32') # [batch_rep_size, nframe, ncell*spike_delay]

		return y_batchrep_train, pasty_batchrep_train

	def yresponse_and_spikehistory__eval(self, numbatch_rep, input_delay, spike_delay):
		y_eval__whole = self._yresponse_nbatch_eval__whole(numbatch_rep) # batch_rep_size, ncell, nframe

		cut = max(input_delay-1, spike_delay)
		y_batchrep_eval = np.array([y_eval__whole[ri,:,cut:].T for ri in range(numbatch_rep)], dtype='float32') # [batch_rep_size, nframe, ncell]

		pasty_batchrep_eval = np.array([y_eval__whole[:, :, fi-spike_delay:fi].reshape((numbatch_rep, -1)) for fi in range(cut, self.train_numimg)]) # [nframe, batch_rep_size, ncell*spike_delay]
		pasty_batchrep_eval = np.moveaxis(pasty_batchrep_eval, 1, 0) # [batch_rep_size, nframe, ncell*spike_delay]
		pasty_batchrep_eval = pasty_batchrep_eval.astype('float32') # [batch_rep_size, nframe, ncell*spike_delay]

		return y_batchrep_eval, pasty_batchrep_eval







	def _yresponse_nbatch_train__whole(self, nbatch):
		if (not hasattr(self, 'rnd_idx__train')) or (self.rnd_idx__train is None):
			assert nbatch <= self.train_numrep
			self._generate_permuted_train_idx(self.train_numrep, train=True)


		if (self.batch_i+nbatch > self.train_numrep): # not enough -> repermute the repetitions
			self.rnd_idx__train = None
			return self._yresponse_nbatch_train__whole(nbatch)
		else:

			batchid = self.rnd_idx__train[self.batch_i:self.batch_i+nbatch]
			self.batch_i += nbatch

			return self._yresponse_repi_train_whole(batchid)

	def _yresponse_nbatch_eval__whole(self, nbatch):
		if (not hasattr(self, 'rnd_idx__eval')) or (self.rnd_idx__eval is None):
			assert nbatch <= self.eval_numrep
			self._generate_permuted_train_idx(self.eval_numrep, train=False)


		if (self.batch_i+nbatch > self.eval_numrep):
			self.rnd_idx__eval = None
			return self._yresponse_nbatch_eval__whole(nbatch)
		else:

			batchid = self.rnd_idx__eval[self.batch_i:self.batch_i+nbatch]
			self.batch_i += nbatch

			return self._yresponse_repi_eval_whole(batchid)

	def _yresponse_repi_train_whole(self, batchid):
		return np.array([self.train_activity[ri*self.train_numimg:(ri+1)*self.train_numimg,:].T for ri in batchid])

	def _yresponse_repi_eval_whole(self, batchid):
		return np.array([self.eval_activity[ri*self.eval_numimg:(ri+1)*self.eval_numimg,:].T for ri in batchid])


	def _generate_permuted_train_idx(self, max_rep, train=True):
		if train:
			self.rnd_idx__train = np.random.permutation(max_rep)
		else:
			self.rnd_idx__eval = np.random.permutation(max_rep)
		self.batch_i = 0

