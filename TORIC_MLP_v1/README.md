This directory contains scikit-learn surrogate models for the toric neural 
network as well as a conversion script for taking these models and  extracting
the weights and biases so that they can be used with 
[TORAX](https://github.com/google-deepmind/torax).

The `torax_weights_generate.py` script requires the following libraries to run:
- joblib
- numpy
- scikit-learn==1.3.2 (The scikit-learn version these networks were saved with.)

and generates a `toricnn.json` file that can be used to use a ion-cyclotron 
resonance heating source model in TORAX.