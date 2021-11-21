# Axial-MLP
Offical implementation of Axial-MLP in PyTorch

Code for the paper
***
**Axial multi-layer perceptron architecture for automatic segmentation of choroid plexus in multiple sclerosis**  
Marius Schmidt-Mengin, Vito A. G. Ricigliano, Benedetta Bodini, Emanuele Morena, Annalisa Colombi, Mariem Hamzaoui, Arya Yazdan Panah, Bruno Stankoff, Olivier Colliot  
SPIE Medical Imaging 2022
***

# Requirements

- PyTorch
- Numpy
- unfoldNd (https://github.com/f-dangel/unfoldNd) by Felix Dangel as the fold/unfold functions of PyTorch currently do not support volumetric inputs. 

# Usage
```python
from axial_mlp_3d import AxialMLP3d

model = AxialMLP3d(
   patch_size,      # int or 3-tuple
   input_size,      # int or 3-tuple. Spatial shape of the input images. Only fixed input is supported.
   in_channels,     # Number of channels in the input (e.g. 1 for most medical images)
   out_channels,    # Number of output channels. Corresponds to the desired number of classes for segmentation. Note that no sigmoid or softmax is applied.
   num_layers=6,    # Number of Axial-MLP blocks
   filters=8,       # Number of filters in the hidden layers
   dropout_rate=0   # Dropout rate, between 0 and 1.
)

x = torch.zeros(batch_size, in_channels, input_depth, input_height, input_width)
y = model(x) # has shape (batch_size, out_channels, input_depth, input_height, input_width)
```
