# Description

These are the critical files needed when running the experiment for my Master of Science dissertation at The University of Cape Town.

The experiment runs a set number of generations of chromosomes involved in learning the game of Atari Pong via A3C reinfrocement learning. Each chromosome represents the the neural network that is updated as part of policy approximation. The goal is to find neural networks that perform well on some stated fitness metric - in this case, a weighted average of best EMA on the task, noise, and time per episode.

The first generation of chromosomes is created by running genesis.py, thereafter the full experiment is run with experiment_controller.py which runs run_experiment_static.py on each instance.

![Overall design](images/cuningham_2_overallexperiment.png)

# How to run it

genesis

changing important files

CLI syntax

## AMI setup

Insert notes here on dependencies


# Future enhancements



