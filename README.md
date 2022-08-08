# k3pi-data
Data parsing, cleaning, etc for D->K3pi analysis to get everything into a consistent format

## Prerequisites
#### python
use a decent version of python

### AmpGen
Amplitude model for $D\rightarrow K3\pi$
**Used for:** training the efficiency reweighter

### MC
Simulation using the full LHCb software. Uses the AmpGen models

### Particle Gun
Simulation: higher statistics than LHCb MC, but potentially less accurate. Uses the AmpGen models
**Used for:** training the efficiency reweighter

### Real Data
**Used for:** the analysis

