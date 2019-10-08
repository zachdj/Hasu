# Hasu
Reinforcement learning in the StarCraft II environment.

## Overview
Hasu is a project which seeks to replicate the results obtained by DeepMind on the StarCraft II mini-games using
deep reinforcement learning.  We seek to produce a faithful representation of the architecture described in the paper
using the PyTorch library.

Rather than A3C (Asynchronous Advantage Actor-Critic), this repository uses A2C.  A2C is synchronous and generally
achieves results equivalent to A3C.

The code in this repository should generalize to any SC2 mini-game.  However, we have begun by focusing on the 
"Defeat Roaches" mini-game, and thus "Defeat Roaches" is currently the only map supported by the command-line interface.

## Installation Instructions

You should have the full StarCraft II game installed.  Installation instructions for all platforms can be found 
[here](https://github.com/deepmind/pysc2#get-starcraft-ii).
Make sure to set the `SC2PATH` environment variable to the correct location.

Make sure to download the pysc2 [mini-game maps](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip)
and follow Blizzard's
[provided instructions](https://github.com/Blizzard/s2client-proto#installing-map-and-replay-packs)
to install them.

Other dependencies are managed via [Conda](https://conda.io/docs/).  This repository includes an `environment.yml` 
file that can be used by Conda to create a Python environment with all required dependencies.

### On Linux:

`conda env create -f environment.yml`

This will create a conda environment named `hasu` with the required dependencies.  The environment can be activated with

`source activate hasu`

### On Windows:

`conda env create -f environment.yml`

This will create a conda environment named `hasu` with the required dependencies.  The environment can be activated with

`conda activate hasu`

## Running the Agent

Hasu includes a command-line interface that can be used to run an a2c agent in training mode or "testing" mode.
The agent can be run with any network included in the hasu.networks package.  We have currently implemented the 
AtariNet architecture, and we hope to add the FullyConv and FullyConv LSTM architectures soon.

### Training mode

To run an agent in training mode:

`(hasu) $ python -m hasu train <args>`

To get a full list of arguments and instructions on each argument, run

`(hasu) $ python -m hasu train -h`


### Testing mode

To run an agent in testing mode:

`(hasu) $ python -m hasu run <args>`

To get a full list of arguments and instructions on each argument, run

`(hasu) $ python -m hasu run -h`

### Included networks

We currently include three pre-trained networks in the repository.  Each of the networks were trained using a screen 
resolution of 84 x 84 and a minimap resolution of 64 x 64.
The `trained_nets/not_limited` network can be used without limiting action space or observation space (default args).
This network was trained for 900,000 steps.

The `trained_nets/action_limited` network can be used with the `--limit_action_space` argument.  
This network was trained for 1.75 million steps with a customized selection of the action space.

The `trained_nets/not_limited` network can be used with both the `--limit_observation_space` and `--limit_action_space` arguments.
This network was trained for 2 million steps with a customized selection of the observation space and a 
customized selection of the action space.


## Contributing

If you'd like to contribute to this project, 
please contact [Zach Jones](https://github.com/zachdj) for instructions and help getting started.

Following is a list of ideas that we would like to incorporate soon:

- Xavier initialization for the NN weights
- Allow the categorical feature embedder to output many dimensions (see `hasu.utils.preprocess`)
- FullyConv LSTM architecture
- Speed up preprocessing so network can be trained faster
- Add support for multiple GPUS
- Verify implementation of A2C
- Allow map to be chosen via the CLI

# Contributors
- Zach Jones (https://github.com/zachdj)
- Layton Hayes (https://github.com/minimum-LaytonC)
