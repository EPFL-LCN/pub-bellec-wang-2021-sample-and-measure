# Sample-and-measure paper

Fitting summary statistics of neural data with a differentiable spiking network simulator  
Guillaume Bellec*, Shuqi Wang*, Alireza Modirshanechi, Johanni Brea^, Wulfram Gerstner^  
  

Accepted for publication at the NeurIPS conference 2021  
(*= equal contributions, ^ = senior authors)  
[arxiv](https://arxiv.org/abs/2106.10064)  

## Environment

`pip install -r requirements.txt`

## Execution example

`python train.py --SEED=0 --ground_truth=no  --GPU=yes --maxiter=800`


`src.parser.py` specifies arguments used in the experiment.


#### Network architecture

Overall, the network `src.buildnet.py` consists of two parts. One is the CNN `conv2`that pre-processes the video non-linearly. And the other is stochastic recurrent neural network `RSNN` that contains one layer of N GLM neurons that generates spikes.


#### Dataset

`DataLoader` in the `src.utils.py` handles preprocessing and loading data.

To run the code, one will need to prepare:
One video data for training, saved in the form of numpy array with size: #pixel_y, #pixel_x, #frame; 
One spike train recordings for training, saved in the form of numpy array with size: #repetition, #frame, #neuron;
And similarly, one video data + one spike train recordings for validating/ testing.

You may want to tailor the CNN that pre-processes the video according to the input.
Once a different video or CNN is used, one have to specify the `input_latent_size`, it's the size of the flattened output of CNN, also the input size of the recurrent network.


#### Forcasting

Once a model is trained, one can use `forcast.py` to simulate spike train samples. 