 # %tensorflow_version 2.x
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import io
import os
import pdb

def pseudo_derivative(v_scaled, dampening_factor):
  return dampening_factor * tf.maximum(0.,1 - tf.abs(v_scaled))

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor, temperature_parameter):

    z_ = tf.math.greater(
            tf.sigmoid(temperature_parameter*v_scaled), 
            tf.random.uniform(shape=v_scaled.shape, minval=0., maxval=1.)
        )
    z_ = tf.cast(z_, dtype=tf.float32)


    def grad(dy):
        # This is where we overwrite the gradient
        # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor), tf.zeros_like(temperature_parameter)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.function
def classification_loss(v_scaled, spike_labels, temperature_parameter):
    # temperature_parameter is either trained or tuned by hand
    spike_logits = temperature_parameter * v_scaled
    spike_probability = tf.sigmoid(spike_logits)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(spike_labels,spike_logits)
    loss_reduced = tf.reduce_mean(loss)


    return loss_reduced, spike_probability, loss

@tf.function
def psth_loss(v_scaled, spike_labels, temperature_parameter):
    # temperature_parameter is either trained or tuned by hand
    spike_logits = temperature_parameter * v_scaled
    spike_probability = tf.sigmoid(spike_logits)

    pi_gt = tf.reduce_mean(spike_labels, axis=0)
    pi_tc = tf.reduce_mean(spike_probability, axis=0)


    loss = 0-tf.math.multiply(pi_gt, tf.math.log(pi_tc))-tf.math.multiply(1-pi_gt, tf.math.log(1-pi_tc))

    loss_reduced = tf.reduce_mean(loss)
    return loss_reduced, spike_probability, loss


@tf.function
def psth_loss_MSE(v_scaled, spike_labels, temperature_parameter):
    # temperature_parameter is either trained or tuned by hand
    spike_logits = temperature_parameter * v_scaled
    spike_probability = tf.sigmoid(spike_logits)

    pi_gt = tf.reduce_mean(spike_labels, axis=0)
    pi_tc = tf.reduce_mean(spike_probability, axis=0)


    mse = tf.keras.losses.MeanSquaredError()
    loss_reduced = mse(pi_gt, pi_tc)

    return loss_reduced, spike_probability, None


@tf.function
def get_delayed_covariance(n_l, delays):
    # n_l: nbatch, nframe, ncell
    nbatch, nframe, ncell = n_l.shape
    Cs = [tf.reshape(tf.einsum("bti,btj->ij", n_l, tf.roll(n_l, shift=delay, axis=1)),[-1]) / (nbatch * nframe)\
            for delay in delays]

    return tf.stack(Cs, axis=0)

@tf.function
def NoiseCorr_loss_MSE(v_scaled, z_model, spike_labels, temperature_parameter, delays=[0]):
    nbatch, nframe, ncell = spike_labels.shape 

    # temperature_parameter is either trained or tuned by hand
    spike_logits = temperature_parameter * v_scaled
    spike_probability = tf.sigmoid(spike_logits)

    # check spike_labels.dtype: float32
    # check loss_reduced
    spikes_corrected = spike_labels - tf.math.reduce_mean(spike_labels, axis=0, keepdims=True)

    # M_ij_D = tf.einsum("bti,btj->ij", spikes_corrected, spikes_corrected) / (nbatch * nframe)
    M_ij_D = get_delayed_covariance(spikes_corrected, delays)

    spike_probability_corrected = spike_probability - tf.math.reduce_mean(spike_probability, axis=0, keepdims=True)
    # M_ij_est = tf.einsum("bti,btj->ij", spike_probability_corrected, spike_probability_corrected) / (nbatch * nframe)
    M_ij_est = get_delayed_covariance(spike_probability_corrected, delays)

    mse = tf.keras.losses.MeanSquaredError()
    loss_reduced = mse(M_ij_D, M_ij_est) 

    return loss_reduced, spike_probability, None

@tf.function
def HiddenNeuron_FRloss_MSE(z_model, target_fr):
    mse = tf.keras.losses.MeanSquaredError()
    loss_fr = mse(tf.math.reduce_mean(z_model,axis=[0,1]), target_fr)

    return loss_fr


class RSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, FLAGS):
        super().__init__()

        self.dampening_factor = FLAGS.dampening_factor
        self.thr = FLAGS.thr
        self.temperature_parameter = FLAGS.temperature_parameter

        self.num_neurons = num_neurons
        self.hidden_output_neuron = FLAGS.hidden_output_neuron
        self.spike_delay = FLAGS.spike_delay
        self.input_weights = None
        self.recurrent_weights = None

        self.state_size = (num_neurons+self.hidden_output_neuron, 
                            (num_neurons+self.hidden_output_neuron)*self.spike_delay, 
                        ) 

        self.ground_truth = FLAGS.ground_truth
        self.call_framei = 0
        
    def build(self, input_shape):


        n_in = input_shape[-1]
            
        n = self.num_neurons + self.hidden_output_neuron
        self.num_inputs = n_in
 
        rand_init = tf.keras.initializers.RandomNormal
        const_init = tf.keras.initializers.Constant

        self.input_weights = self.add_weight(
            shape=(n_in,n),
            trainable=True,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=1. / np.sqrt(float(n_in))),
            name='input_weights')
        
        # define the recurrent weight variable
        self.recurrent_weights = self.add_weight(
            shape=(n, n, self.spike_delay),
            trainable=True,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=1. / np.sqrt(n *self.spike_delay)),
            name='recurrent_weights')

        self.bias = self.add_weight(
            shape=(n), 
            initializer='zeros', dtype=tf.float32, 
            name='neuron_bias')

        super().build(input_shape)

    def get_recurrent_weights(self): # get the coupled weights
        return self.recurrent_weights

    def get_input_weights(self):
        return self.input_weights

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        initialized_state = (tf.zeros((batch_size,self.num_neurons+self.hidden_output_neuron)), 
                             tf.zeros((batch_size,self.num_neurons+self.hidden_output_neuron, self.spike_delay)),
                            )
        self.call_framei = 0
        return initialized_state

    def call(self, inputs, states, constants=None):


        def get_spike_from_inputs(): 
            return tf.concat([
                        tf.reshape(inputs[:,self.num_inputs:], (-1, self.num_neurons, self.spike_delay)), # nbatch, ncell, spike_delay
                        states[1][:,self.num_neurons:, :], # nbatch, hidden_output_neuron, spike_delay
                    ], 1) # first output neurons, then hidden neurons
        def get_spike_from_states(): 
            return states[1]
        
        spike_buffer = tf.cond( self.ground_truth, 
                            get_spike_from_inputs, 
                            get_spike_from_states)


        w_rec = self.get_recurrent_weights()
        i_from_spike_buffer = tf.einsum("bit,jit->bj",spike_buffer, w_rec)

        video_inputs = inputs[:,:self.num_inputs] # nbatch * num_inputs

        w_in = self.get_input_weights()
        i_from_in_buffer = tf.einsum("bi,ij->bj", video_inputs, w_in)

        new_v = tf.nn.bias_add(tf.add(i_from_spike_buffer, i_from_in_buffer), self.bias)

        new_v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(new_v_scaled, self.dampening_factor, self.temperature_parameter)
        new_spike_buffer = tf.concat([spike_buffer[:,:,1:], tf.expand_dims(new_z, 2)], axis=2)
        new_state = (new_v, new_spike_buffer)


        self.call_framei += 1
        
        return (new_z, new_v, new_v_scaled), new_state


    @staticmethod
    def forward_pass(rnn, input_for_rnn, y_batchrep, pasty_batchrep,
                    ground_truth, weight_psth, weight_NC, weight_GT, weight_fr, hidden_output_neuron=0):

        numbatch_rep,numconsinferenceframes, ncell = y_batchrep.shape

        loss_fr = 0.; target_fr = 0.17

        if not ground_truth:
            rnn.cell.ground_truth = False
            z_psth, v, v_scaled_psth = rnn(tf.concat((input_for_rnn, pasty_batchrep), axis=2))
            # [nrep, nframe, ncell]

            loss_psth, spike_probability, _ = psth_loss(tf.slice(v_scaled_psth,(0,0,0), (numbatch_rep,numconsinferenceframes,ncell)), 
                                                                    tf.slice(y_batchrep,(0,0,0), (numbatch_rep,numconsinferenceframes,ncell)), 
                                                                    rnn.cell.temperature_parameter)
            loss_NC, _, _ = NoiseCorr_loss_MSE(tf.slice(v_scaled_psth,(0,0,0), (numbatch_rep,numconsinferenceframes,ncell)), 
                                                                    tf.slice(z_psth,(0,0,0), (numbatch_rep,numconsinferenceframes,ncell)), 
                                                                    tf.slice(y_batchrep,(0,0,0), (numbatch_rep,numconsinferenceframes,ncell)), 
                                                                    rnn.cell.temperature_parameter,
                                                                    delays=list(range(rnn.cell.spike_delay))
                                                                    )
            if hidden_output_neuron:
                loss_fr = HiddenNeuron_FRloss_MSE(tf.slice(z_psth,(0,0,ncell), (numbatch_rep,numconsinferenceframes,hidden_output_neuron)), target_fr)
                print('hidden neuron mean fr: %f'%(np.median(z_psth.numpy()[:,:,ncell:].mean(0).mean(0))))
        else:
            loss_psth = 0
            loss_NC = 0

        if True:
            rnn.cell.ground_truth = True
            z_gt, v, v_scaled_gt = rnn(tf.concat((input_for_rnn, pasty_batchrep), axis=2)) # [nrep, nframe, ncell]
            loss_llh, spike_probability, _ = classification_loss(tf.slice(v_scaled_gt,(0,0,0), (numbatch_rep, numconsinferenceframes,ncell)), 
                                                                    tf.slice(y_batchrep,(0,0,0), (numbatch_rep, numconsinferenceframes,ncell)), 
                                                                    rnn.cell.temperature_parameter)
            if hidden_output_neuron and ground_truth:
                loss_fr = HiddenNeuron_FRloss_MSE(tf.slice(z_gt,(0,0,ncell), (numbatch_rep,numconsinferenceframes,hidden_output_neuron)), target_fr)
                print('hidden neuron mean fr: %f'%(np.median(z_gt.numpy()[:,:,ncell:].mean(0).mean(0))))


        if not ground_truth:
            lossavgbatch = loss_psth*weight_psth + loss_NC*weight_NC + \
                    loss_llh*weight_GT + weight_fr*loss_fr
        else:   
            lossavgbatch = loss_llh + weight_fr*loss_fr



        return lossavgbatch, loss_llh, loss_psth, loss_NC, loss_fr


class conv2(tf.keras.Model):
    def __init__(self, FLAGS):
        super(conv2, self).__init__(name='')
        assert(FLAGS.numconvlayer == 2)

        self.conv2a = tf.keras.layers.Conv2D(FLAGS.conv1, FLAGS.conv1size, strides=1, padding='same',
            activation='relu',
            )
        self.maxpool2a = tf.keras.layers.MaxPool2D(FLAGS.nk1, strides=FLAGS.nstride1, padding='same')

        self.conv2b = tf.keras.layers.Conv2D(FLAGS.conv2, FLAGS.conv2size, strides=1, padding='same',
            activation='relu',
            )
        self.maxpool2b = tf.keras.layers.MaxPool2D(FLAGS.nk2, strides=FLAGS.nstride2, padding='same')

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = self.maxpool2a(x)

        x = self.conv2b(x)
        x = self.maxpool2b(x)

        return self.flatten(x)
        
def build_model(ncell, imagex, imagey, FLAGS):
    inputnet = conv2(FLAGS)
    inputnet(np.random.rand(FLAGS.batch_size, imagex, imagey, FLAGS.numconsframes))  # dummy call
    cell = RSNN(ncell, FLAGS)
    cell.build([None,None, FLAGS.input_latent_size])
    rnn = tf.keras.layers.RNN(cell,return_sequences=True)

    return inputnet, rnn
