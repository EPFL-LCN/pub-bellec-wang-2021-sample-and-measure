import os
import argparse

def gen_parser():
    parser = argparse.ArgumentParser()

    saving = parser.add_argument_group('saving params')
    saving.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getcwd(),'data','v1_natural'),
    )
    saving.add_argument(
      '--save_dir',
      type=str,
      default=os.path.join(os.getcwd(),'manualsave'),
      help='Directory to save outputs'
    )


    
    # Data
    data = parser.add_argument_group('data params')
    data.add_argument(
      '--numrep',
      type=int,
      default=80,
    )
    data.add_argument(
      '--dt',
      default=40,
    )




    opt = parser.add_argument_group('optimizer params')
    opt.add_argument(
      '--GPU',
      type=str,
      help='whether use GPU: yes/ no',
      default='no',
    )
    opt.add_argument(
      '--batch_size', 
      type=int,
      default=20,
    )
    
    opt.add_argument(
      '--maxiter',
      type=int,
      default=2000,
    )
    opt.add_argument(
      '--learning_rate',
      type=float,
      default=1e-3,
    )
    opt.add_argument(
      '--clipnorm',
      type=float,
      default=1.,
    )
    opt.add_argument(
      '--temperature_parameter',
      type=float,
      default=1.,
    )

    loss = parser.add_argument_group('loss params')
    loss.add_argument(
      '--ground_truth',
      type=str,
      help='whether conditioned on ground truth: yes/ no',
      default = 'no',
    )
    loss.add_argument(
      '--weight_psth', 
      default=0.1,
    )
    loss.add_argument(
      '--weight_NC', 
      default=50,
    )
    loss.add_argument(
      '--weight_GT', 
      default=0.4,
    )
    loss.add_argument(
      '--weight_fr', 
      help='weight for SM-h if hidden neurons are included in the SNN',
      default=0.001,
    )




    other = parser.add_argument_group('other params')
    other.add_argument(
      '--SEED',
      type=int,
      default=0,
    )
    other.add_argument(
      '--IDtag',
      type=str,
      help='no need to set during training; set when forcast',
      default='',
    )


    # Define SNN:
    snn = parser.add_argument_group('SNN params')
    snn.add_argument(
      '--hidden_output_neuron',
      type=int,
      default=0
    )
    data.add_argument(
      '--numconsframes',
      type=int,
      default=10,
    )
    data.add_argument(
      '--spike_delay',
      type=int,
      default=9,
    )
    snn.add_argument(
      '--thr',
      type=float,
      default=.4,
    )
    snn.add_argument(
      '--dampening_factor',
      type=float,
      default=.3,
    )
    

    # Define Convolutional parameters
    cnn = parser.add_argument_group('CNN params')
    cnn.add_argument(
      '--conv1',
      type=int,
      default=16,
      help='Number of filters in conv 1.'
    )
    cnn.add_argument(
      '--conv2',
      type=int,
      default=32,
      help='Number of filters in conv 2.'
    )
    cnn.add_argument(
      '--conv1size',
      type=int,
      default=7,
      help='Size (linear) of convolution kernel larer 1.'
    )
    cnn.add_argument(
      '--nk1',
      type=int,
      default=3,
      help='Size of max pool kernel layer 1.'
    )
    cnn.add_argument(
      '--nstride1',
      type=int,
      default=2,
      help='Size of max pool stride layer 1.'
    )
    cnn.add_argument(
      '--conv2size',
      type=int,
      default=7,
      help='Size (linear) of convolution kernel larer 2.'
    )
    cnn.add_argument(
      '--nk2',
      type=int,
      default=3,
      help='Size of max pool kernel layer 2.'
    )
    cnn.add_argument(
      '--nstride2',
      type=int,
      default=2,
      help='Size of max pool stride.'
    )
    cnn.add_argument(
      '--numconvlayer',
      type=int,
      default=2,
      help='number of convolutional layers'
    )
    cnn.add_argument(
      '--input_latent_size',
      type=int,
      default=1568 # 23104/6080 allen # 1568 v1
    )

    return parser

def default_flags():
    FLAGS,_ = gen_parser().parse_known_args()

    return FLAGS
