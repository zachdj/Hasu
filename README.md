# Hasu
Reinforcement learning in the StarCraft II environment.

## Installation Notes

You should have the full StarCraft II game installed.  Installation instructions for all platforms can be found 
[here](https://github.com/deepmind/pysc2#get-starcraft-ii).
Make sure to set the `SC2PATH` environment variable to the correct location.

Make sure to download the 2018 Ladder Maps and the Melee maps from 
[Blizzard](https://github.com/Blizzard/s2client-proto#map-packs) and install them using the 
[provided instructions](https://github.com/Blizzard/s2client-proto#installing-map-and-replay-packs).

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

## Development Notes

Dono where else to put these for now

### Running an agent

`python -m pysc2.bin.agent --map DefeatRoaches --agent hasu.agents.a2c.A2CAgent`

### listing actions

`python -m pysc2.bin.valid_actions`




# Contributors
- Zach Jones (https://github.com/zachdj)
- Layton Hayes (https://github.com/minimum-LaytonC)
