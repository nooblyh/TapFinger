

# Prerequisites
Install tianshou and pyg using the conda environment. Tianshou has been changing so if you need to use the features of the new version of tianshou, you may have to deal with some compatibility issues.

The following command directly creates a conda environment named `gnn_drl` for the project to run.

It requires Python >= 3.6.

`
conda create -n gnn_drl pytorch torchvision torchaudio cudatoolkit=11.3 torchtext torchdata pyg tianshou=0.4.7 pynvml matplotlib -c pytorch -c pyg -c conda-forge
`

# Code Structure
The `custom_gym` directory stores RL environment related files, that is, the environment's `step` function, `reset` function, etc. The code file of the environment needs to be registered in `custom_gym/myenv/__init__.py`, please refer to the code for details.

The `nn` directory contains the neural network definitions of GNN, DRL, and pointer network. We use the pyG library to implement the GNN.

The `policy` directory contains the control logic of the DRL algorithm on the tianshou side. The important ones are the `forward` function and the `learn` function. The `forward` function can be regarded as the general entrance of the entire network, and various neural network modules in the `nn` directory are called at this entrance. `learn` function defines how to update the parameters of the network, such as what loss function to use for the actor and the critic respectively.

The `config` file in the `util` directory saves all environment settings and hyperparameter settings at runtime, such as how many agents there are, how many layers the network has, and so on. `config` file needs to be valid to run correctly. The current method is very inconvenient. A better way would be to save these settings in the form of JSON, then specify which JSON file to import at runtime, and save the JSON file to the result folder to facilitate subsequent reproduction and analysis.

During the iterative process of code development, the config has been modified many times, which may cause the old config file to not match the existing framework and cannot be run directly.

The python files prefixed with `test` in the main directory are running scripts or the main function of training. `multi_agent_inference.py` is the script for inference.

# How to Run
Training: `python test_trace.py`

Inference: `python multi_agent_inference.py`
(There is a conflict between the inference config and the training config, so the inference cannot run directly if the current `config.py` is not correct😟)

# Publications
If our work would have the honor of contributing to your research, you can cite the papers below.

[Li, Yihong, et al. "TapFinger: Task Placement and Fine-Grained Resource Allocation for Edge Machine Learning." IEEE INFOCOM. 2023.](http://i2.cs.hku.hk/~cwu/papers/yhli-infocom23.pdf)

[Li, Yihong, et al. "Task Placement and Resource Allocation for Edge Machine Learning: A GNN-based Multi-Agent Reinforcement Learning Paradigm." arXiv preprint arXiv:2302.00571 (2023).](https://arxiv.org/abs/2302.00571)
