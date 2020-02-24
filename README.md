# Efficiency, Stability and Generalization Analysis for Reinforcement Learning on Robotic Peg-in-hole Assembly
> If you have any question or want to report a bug, please open an issue instead of emailing me directly. 

### Webots install
Follow the officical website : 
```
cd code/webots/atlas_boom/controllers/rl
```
Open the webots world file.
```
./run.sh
```

### Implemented algorithms:
* Deep Deterministic Policy Gradient (DDPG)
* Twined Delayed DDPG (TD3)
* Average-TD3
* [HRL(SAC-AWMP)](https://arxiv.org/abs/2002.02829)

### Dependency
* MacOS 10.12 or Ubuntu 16.04
* PyTorch v1.1.0
* Python 3.6, 3.5
* Core dependencies: `pip install -e .`

**Change the work path to the path of your main.py**

Then run the code:
```
python ./main.py
```

### Our performance 

* DDPG/TD3 evaluation performance.
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/mujoco_eval.png)

# References
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [Feedback Deep Deterministic Policy Gradient with Fuzzy Reward for Robotic Multiple Peg-in-hole Assembly Tasks](https://ieeexplore.ieee.org/abstract/document/8454796)
* [Deep reinforcement learning for high precision assembly tasks](https://ieeexplore.ieee.org/abstract/document/8202244)
* Some hyper-parameters are from [OpenAI Baselines](https://github.com/openai/baselines) and [Deep_RL](https://github.com/ShangtongZhang/DeepRL) 