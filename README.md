# k3pi-data
Data parsing, cleaning, etc for D->K3pi analysis to get everything into a consistent format

I've chosen to save everything as pickle dumps of pandas dataframes

## Prerequisites
#### python
use a decent version of python

### AmpGen
Amplitude models for $D\rightarrow K+3\pi$.  
The only difference between an amplitude and its conjugate
(e.g. $D^0\rightarrow K^+3\pi$ and $\bar{D}^0\rightarrow K^-3\pi$) should be the direction of the 3-momenta,
so if we want to transform between them we just need to multiply the 3-momenta by -1.  
We can do this by using the `flip_momenta` function [here](lib_data/util.py#L29).

**Used for:** training the efficiency reweighter

### MC
Simulation using the full LHCb software. Generates events according to AmpGen amplitude models

### Particle Gun
Simulation: higher statistics than LHCb MC, but potentially less accurate. Uses the AmpGen models
**Used for:** training the efficiency reweighter

### Real Data
**Used for:** the analysis

