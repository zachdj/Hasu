# Hasu
Reinforcement learning in the StarCraft II environment.

# Installation Notes

You should have the full StarCraft II game installed.  Installation instructions for all platforms can be found [here](https://github.com/deepmind/pysc2#get-starcraft-ii).

Make sure to download the 2018 Ladder Maps and the Melee maps from [Blizzard](https://github.com/Blizzard/s2client-proto#map-packs) and install them using the [provided instructions](https://github.com/Blizzard/s2client-proto#installing-map-and-replay-packs).

Other dependencies are managed via [Conda](https://conda.io/docs/).  This repository includes an `environment.yml` file that can be used by Conda to create a Python environment with all required dependencies.  Just run

`conda env create -f environment.yml`

This will create a conda environment named `hasu` with the required dependencies.  The environment can be activated with

`source activate hasu`

# Contributors
- Zach Jones (https://github.com/zachdj)
- Layton Hayes (https://github.com/minimum-LaytonC)
