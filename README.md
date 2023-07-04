## LifeSteps and PPO

[Daniel Bernardi](http://github.com/ancaah)

This project consists in two main parts:
- the creation of a customized Gymnasium environment
- the implementation of a PPO Agent

the final.ipynb notebook contains all the python code used to train and evaluate the PPO agent. Contains also the output from a previous execution.\\
the latex report (report.pdf) describes in a detailed way the steps and results of the project.\\
the agent.py and memory.py files contain the actual implementation of the agent. Many informations about the PPO algorithm are here.\\
the gym_projects folder contains the implementation of the custom Gymnasium environment. lifesteps.py is the actual implementation.\\

## Register the environment to your local installation
If you want to try LifeSteps, you need to register it into your local Gymnasium installation.\\
To do that, go in the command line while into the main folder of this repo and write this command:
```
# pip install -e gym_projects
```
to uninstall:
```
# pip uninstall life-sim
```

If you're interested in running my implementation, follow the final.ipynb notebook.\\
This PPO algorithm can be used on any Gymnasium environment of your liking :\)
Have fun, i did.
