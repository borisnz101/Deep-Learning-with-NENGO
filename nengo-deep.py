import nengo
import numpy as np


#We start off with creating a modelwork named "function"
model = nengo.Network(seed=0, label="function")
model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(0, 1)
model.config[nengo.Connection].synapse = None

with model:
    x, y, z = [nengo.Node(nengo.processes.WhiteSignal(0.5, 5, rms=0.3, seed=s))for s in range(3)]
    
with model:
	#First ensemble is composed of 50 neurons of type Rectified linear, taking 2 inputs (x and y) each
	ens1 = nengo.Ensemble(50, 2, neuron_type = nengo.RectifiedLinear())
	#The second and third ensemble are composed of 25 neurons, taking a single input (z).
	ens2 = nengo.Ensemble(25, 1, neuron_type = nengo.RectifiedLinear())
	ens3 = nengo.Ensemble(25, 1, neuron_type = nengo.RectifiedLinear())
    #Notice how ens1 is bi-dimensional, meaning each neuron can take 2 inputs
    
with model:
    nengo.Connection(x, ens1[0]) 
    nengo.Connection(y, ens1[1])
    nengo.Connection(z, ens2)
    
    #We create an output node. ens1 and ens3 will be its inputs, to compute the function
    f = nengo.Node(size_in=1)
    #The connection over ens1 and f computes the firt half of the function
    nengo.Connection(ens1, f, function=lambda x: (2*x[0] + 1) * x[1] ** 2)
    #The connection over ens2 and ens3 computes the cos wave
    nengo.Connection(ens2, ens3, function=np.cos)
    #The connection over ens3 and output f computes the squared cos wave
    nengo.Connection(ens3, f, function=np.square)
    
def target_func(x, y, z):
    return ((2*x + 1)*y ** 2) + np.cos(z) ** 2
    
with model:
    x_p = nengo.Probe(x)
    y_p = nengo.Probe(y)
    z_p = nengo.Probe(z)
    f_p = nengo.Probe(f)
    #Let's define inputs x,y,z
    trainSet = {x: np.random.uniform(-1, 1, size = (1000, 1, 1)),
                y: np.random.uniform(-1, 1, size = (1000, 1, 1)),
                z: np.random.uniform(-1, 1, size = (1000, 1, 1))}
    
    #Let's define the output
    trainSet[f_p] = target_func(trainSet[x], trainSet[y], trainSet[z])
    
    
