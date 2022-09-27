# k3pi-data
Data parsing, cleaning, etc for D->K3pi analysis to get everything into a consistent format

I've chosen to save everything as pickle dumps of pandas dataframes

## Prerequisites
#### python
A the scripts in this repo require a few non-standard libraries (`uproot`, `tqdm`);
a full list is in [`requirements.txt`](requirements.txt)

The install instructions below are for python 3.9 - other versions may work, but this is the one I've been using.


#### lxplus
I don't think there are any pre-built environments on `lxplus.cern.ch` that come with all of the requirements already installed,
so you may want to install your own environment from source.
e.g. with...

#### Miniconda
Basically log in to lxplus and follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

You will probably want to do this in your `/work/` partition on lxplus, since your home directory will get full very quickly

In short:
 - download miniconda
   - `wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh`
 - install it
   - `bash Miniconda3-py38_4.12.0-Linux-x86_64.sh`
 - follow the prompts; you can choose your install location from here as well
 - To activate your miniconda environment:
   - `source $MY_MINICONDA_DIR/bin/activate`
   - where `$MY_MINICONDA_DIR` is the location of your miniconda installation
 - to install requirements:
   - `pip install -r requirements.txt`

#### requirements.txt
This contains all the files required for creating the dataframes in this repo and running the scripts.

It will also contain the requirements for the rest of the analysis (though I haven't got round to this yet)


## Types of data in this analysis
To create the MC, particle gun or real data dataframes, just run the right script on lxplus.
To create the AmpGen dataframes, create the right D->K3pi ROOT files using AmpGen, and point the script at them.

AmpGen: https://github.com/GooFit/AmpGen  
D->K3pi config files: https://zenodo.org/record/3457086  

### AmpGen
Amplitude models for $D\rightarrow K+3\pi$.  
The only difference between an amplitude and its conjugate
(e.g. $D^0\rightarrow K^+3\pi$ and $\bar{D}^0\rightarrow K^-3\pi$) should be the direction of the 3-momenta,
so if we want to transform between them we just need to multiply the 3-momenta by -1.  
We can do this by using the `flip_momenta` function [here](lib_data/util.py#L31).

**Used for:** training the efficiency reweighter

### MC
Simulation using the full LHCb software.
Generates events according to AmpGen amplitude models

**Used for:** training the BDT cuts

### Particle Gun
Simulation: higher statistics than LHCb MC, but potentially less accurate.
Uses the AmpGen models

**Used for:** training the efficiency reweighter

### Real Data
**Used for:** the analysis

## Creating the dataframes
Run the `create_data.py`, `create_mc.py` scripts etc. to create the dataframes.
I haven't set everything up yet so some options aren't yet available - so far I've mostly been working with
2018 magdown data/MC.

The scripts should output an error message if you use them wrong telling you the allowed options.
It takes a __long__ time to parse/dump some of these dataframes - especially the real data - so it's often best to just
cancel the process before it finishes if you just want to test something/run some scripts/do anything except the full
analysis.
The scripts will pick up where they left off if you do this, so won't waste time creating/dumping dataframes that
already exist.

