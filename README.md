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

Getting original OpenAI agent to run was challenging - the instructions given did not work, and there were some extra modules that needed to be installed. The code in this repo has several more dependencies too, and was challenging to get working on a fresh instance. The instructions below worked, but it is possible that there might be a few more tweaks needed.

After creating an instance with the "Ubuntu Deep Learning AMI v20":


1. `source activate tensorflow_p36`
2. `pip install --upgrade pip`
3. `sudo apt-get install -y tmux htop cmake golang libjpeg-dev`
4. `pip install "gym[atari]"``
5. `pip install universe` (seemed to work although 2x docker dependency issues)
6. `pip install six` (req satisfied)
7. `conda install -y -c https://conda.binstar.org/menpo opencv3`
8. `conda install tabulate` (went ahead, although it downgraded a few things)
9. `pip install tflearn`
10. I checked to make sure tensorflow and keras were installed (1.12.0 and 2.2.4 respectively)
11. `mkdir evorl`
12. `cd evorl`
13. `mkdir output`



# Future enhancements
