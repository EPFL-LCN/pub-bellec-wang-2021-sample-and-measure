
import tensorflow as tf
import numpy as np
import random
from scipy.stats import pearsonr

import os, sys, time
import pdb

from src.utils import DataLoader
from src.parser import gen_parser
from src.forcast_helper import *
from src.viz import viz_raster_plot

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


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



FLAGS = None
parser = gen_parser()
FLAGS, unparsed = parser.parse_known_args()


'''
Load Data
'''
data_dir = FLAGS.data_dir
train_filename = ['M_train(last)_natural.npy', 'S_1_train(last)_natural_first80.npy']
eval_filename = ['M_train(last)_natural.npy', 'S_1_train(last)_natural_mid30.npy']

data = DataLoader(data_dir, train_filename, eval_filename, FLAGS) # loads and formats all the data



IDtag = FLAGS.IDtag
mansave_dir = os.path.join(FLAGS.save_dir, str(IDtag)+f'_{FLAGS.ground_truth}_{FLAGS.SEED}/')  # the directory to save to
print("manual save dir: " + mansave_dir)
savename_network = mansave_dir + 'network_manualsave_' + IDtag # saving network parameters
savename_nogttrainresults = mansave_dir + 'training_results_nogt' + IDtag
savename_nogtevalresults = mansave_dir + 'val_results_nogt' + IDtag


inputnet, rnn = recover_rnn_and_cnn(mansave_dir, savename_network, data.numcell, data.numpixx, data.numpixy, FLAGS)
rnn.cell.ground_truth = False #FLAGS.ground_truth == 'yes'
'''
simulate on the training set
'''
gtact_train, predact_train, predfr_train, _ = forecast_on_gt(data, FLAGS, inputnet, rnn, train=True, eval=False)  # nrep, ncell, nframe

np.save(savename_nogttrainresults, [predact_train, predfr_train])


'''
simulate on the test set
'''
gtact_eval, predact_eval, predfr_eval, _ = forecast_on_gt(data, FLAGS, inputnet, rnn, train=False, eval=True)  # nrep, ncell, nframe

np.save(savename_nogtevalresults, [predact_eval, predfr_eval])

fig, ax_list = plt.subplots(2,2, figsize=(12,6), sharex=True, sharey=True)
select_first = lambda x : x[0].T[None, ...]
viz_raster_plot(select_first(gtact_train), ax_list[0,0], {'ylabel':"neuron id", "title": "example raster from the training set"})
viz_raster_plot(select_first(predact_train), ax_list[1,0], {'ylabel':"neuron id", "title": "example raster from the model"})

viz_raster_plot(select_first(gtact_eval), ax_list[0,1], {'ylabel':"neuron id", "title": "example raster from the testing set"})
viz_raster_plot(select_first(predact_eval), ax_list[1,1], {'ylabel':"neuron id", "title": "other from the model"})
plt.tight_layout()
fig.savefig(mansave_dir + "rasters.jpg")
plt.show()