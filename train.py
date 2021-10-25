
import tensorflow as tf
import numpy as np
import random
from scipy.stats import pearsonr

import os, sys, time, pickle
import pdb


from src.utils import DataLoader
from src.parser import gen_parser
from src.buildnet import *
from src.viz import mansavefig
FLAGS = None
parser = gen_parser()
FLAGS, unparsed = parser.parse_known_args()

SEED = FLAGS.SEED
# Function to initialize seeds for all libraries which might have stochastic behavior
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
# Activate Tensorflow deterministic behavior
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


set_global_determinism(seed=SEED)



import matplotlib.pyplot as plt
import seaborn as sns



if FLAGS.GPU == 'yes':
	from tensorflow.compat.v1 import ConfigProto
	from tensorflow.compat.v1 import InteractiveSession

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
else:
	try:
	    # Disable all GPUS
	    tf.config.set_visible_devices([], 'GPU')
	    visible_devices = tf.config.get_visible_devices()
	    for device in visible_devices:
	        assert device.device_type != 'GPU'
	except:
	    # Invalid device or cannot modify virtual devices once initialized.
	    # pass
	    print('not able to disable GPU')


def run_training():

	
	'''
	setting up saving and file IDs
	'''
	IDtag = str(int(time.time())) # use a unix time ID for each training
	mansave_dir = os.path.join(FLAGS.save_dir, str(IDtag)+f'_{FLAGS.ground_truth}_{FLAGS.SEED}')  # the directory to save to
	print("manual save dir: " + mansave_dir)
	savename_network = mansave_dir + '/network_manualsave_' + IDtag # saving network parameters
	savename_traininglog = mansave_dir + '/training_' + IDtag # saving network parameters
	savename_trainingcurve = mansave_dir + '/training_plot_' + IDtag # saving traing figure

	#making a folder to save to
	if (tf.io.gfile.exists(mansave_dir) == 0):
		tf.io.gfile.makedirs(mansave_dir)


	'''
	Load Data
	'''
	# list of filenames for data. 
	data_dir = FLAGS.data_dir
	train_filename = ['M_train(last)_natural.npy', 'S_1_train(last)_natural_first80.npy']  # movie, spikes
	eval_filename = ['M_train(last)_natural.npy', 'S_1_train(last)_natural_last10.npy']

	data = DataLoader(data_dir, train_filename, eval_filename, FLAGS) 

	'''
	initiate the model
	'''
	inputnet, rnn = build_model(data.numcell, data.numpixx, data.numpixy, FLAGS) # thr=0.4 if no buffer
	variables = rnn.cell.trainable_variables + inputnet.trainable_variables



	'''
	optimizer
	'''
	optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)
	mse = tf.keras.losses.MeanSquaredError()


	
	x= data.xstimuli()  # (641, 27, 27, 10)
	xeval = data.xstimuli_eval()


	'''
	training log
	'''
	step = 0; steplist = []
	# train batch 
	lossbatchlist, loss_avgbatch, lossavgbatchlist = [], 0, []
	lossavgbatchmin = 1e8
	# validation
	loss_eval, lossevallist = 0, []
	lossevalmin = 1e8

	for i_iteration in range(FLAGS.maxiter):
		step += 1


		y_batchrep_train, pasty_batchrep_train = data.yresponse_and_spikehistory__train(FLAGS.batch_size, FLAGS.numconsframes, FLAGS.spike_delay)


		with tf.GradientTape() as tape:
			# We say which variables are tracked with back-prop through time (BPTT)
			[tape.watch(v) for v in variables]
			
			# get the CNN-processed input for the model
			input_for_rnn = tf.repeat(tf.expand_dims(inputnet(x),axis=0),FLAGS.batch_size,axis=0) 
								# rep, # step, input_latent_size

			# forward pass
			lossavgbatch, loss_gt, loss_psth, loss_NC, loss_fr = RSNN.forward_pass(rnn, input_for_rnn, y_batchrep_train, pasty_batchrep_train,
                    																FLAGS.ground_truth=='yes', FLAGS.weight_psth, FLAGS.weight_NC, FLAGS.weight_GT,
                    																FLAGS.weight_fr, hidden_output_neuron=FLAGS.hidden_output_neuron)

	
		# log
		lossbatchlist.append(lossavgbatch)
		loss_avgbatch = np.mean(lossbatchlist[-int(data.train_numrep/FLAGS.batch_size):])



		# a better model?
		if loss_avgbatch < lossavgbatchmin: 

			lossavgbatchmin = loss_avgbatch


			# run on the whole evaluation dataset
			input_for_rnn_eval = tf.repeat(tf.expand_dims(inputnet(xeval),axis=0),data.eval_numrep,axis=0) # batsize, nframe, input_latent_size
			
			y_batchrep_eval, pasty_batchrep_eval = data.yresponse_and_spikehistory__eval(data.eval_numrep, FLAGS.numconsframes, FLAGS.spike_delay)


			loss_eval, loss_eval_gt, loss_eval_NC, loss_eval_psth, loss_eval_fr = RSNN.forward_pass(rnn, input_for_rnn_eval, y_batchrep_eval, pasty_batchrep_eval,
	                																FLAGS.ground_truth=='yes', FLAGS.weight_psth, FLAGS.weight_NC, FLAGS.weight_GT,
	                																FLAGS.weight_fr, hidden_output_neuron=FLAGS.hidden_output_neuron)

			if loss_eval < lossevalmin:

				lossevalmin =  loss_eval 


				print('new save %d'%step)
				network_save(variables, savename_network) #save the parameters of network		


		# backward pass
		grads = tape.gradient(lossavgbatch, variables)
		assert grads ,"No gradients have been computed, grads is an empty list: {}, variables are {}".format(grads,variables)
		optimizer.apply_gradients([(g,v) for g,v in zip(grads, variables)])

		
		## log
		print('Step %d: loss = %.8f; loss_eval = %.8f' % (step, loss_avgbatch, loss_eval))
		steplist.append(step)
		lossavgbatchlist.append(loss_avgbatch)
		lossevallist.append(loss_eval)

		mansavefig(lossavgbatchlist, lossevallist, steplist, savename_trainingcurve)
		pickle.dump({'step': steplist, 'train_avgloss': lossavgbatchlist, 'eval_loss': lossevallist},
					open(savename_traininglog, 'wb'))





def network_save(variables, savename_network):
	learnable_weights = [v.numpy() for v in variables]
	np.save(savename_network, learnable_weights)




def main(_):

  run_training()

# run main
if __name__ == '__main__':
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
