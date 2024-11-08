# Conditional Neural Expert Processes for Learning Movement Primitives from Demonstration

This repository contains the source code for the paper "Conditional Neural Expert Processes for Learning Movement Primitives from Demonstration" by [Yigit Yildirim](https://www.cmpe.boun.edu.tr/~yigit.yildirim/) and [Emre Ugur](https://www.cmpe.boun.edu.tr/~emre). We both are members of the [CoLoRs Lab, Bogazici University](https://clrslab.wordpress.com/).

CNEP is a novel deep learning architecture for Learning from Demonstration (LfD) in robotics aiming to encode diverse sensorimotor trajectories from demonstrations with varying movements, leveraging a novel gating mechanism, multiple decoders, and an entropy-based loss calculation to promote decoder specialization. This work is published in _IEEE RA-L_. For the full text, please refer to the publisher at [https://ieeexplore.ieee.org/abstract/document/10711283](https://ieeexplore.ieee.org/abstract/document/10711283). You can find the preprint here: [https://arxiv.org/abs/2402.08424](https://arxiv.org/abs/2402.08424) 

You are welcome to use any portion of this study, but in that case, please consider citing:

_Y. Yildirim and E. Ugur, "Conditional Neural Expert Processes for Learning Movement Primitives From Demonstration," in IEEE Robotics and Automation Letters, vol. 9, no. 12, pp. 10732-10739, Dec. 2024, doi: 10.1109/LRA.2024.3477169._

### Architecture overview:

![__over](https://github.com/yildirimyigit/cnep/assets/3774448/4147b4f6-f29c-499a-a119-a5fb31b10aae)

### Some results:
Here are some videos from this work: https://youtube.com/playlist?list=PLXWw0F-8m_ZZD7fpGOKclzVJONXUifDiY
<hr>
<p>
    We assessed the performance of CNEP in comparison to Probabilistic Movement Primitives (ProMP) and Gaussian Mixture Models-Gaussian Mixture Regression (GMM-GMR) on a complex robotic task involving grasping and placing wine glasses onto a dish rack. This task involved high-dimensional sensorimotor trajectories of 1288 dimensions. Each model was trained on 40 expert demonstrations and expected to generate the necessary control commands to achieve successful task completion.  CNEP demonstrated superior performance by successfully completing the task, while ProMP and GMM-GMR were unable to achieve a successful grasp of the glass. A video explaining the experiment is available for further analysis: https://youtu.be/ffnIhrmjwgo
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/572980a1-0f43-404c-8bfa-1811b27af4c0"/>
</p>

<hr>
<p>
    With continuous conditioning on the current configuration of the tabletop, CNEP can adapt to the changes in the environment on the fly and select among multiple experts to properly control the robot. The video showing the online control experiment is here: https://youtu.be/ffnIhrmjwgo
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/00d6ee8b-9f74-432e-b82c-d74cb3926368" width="250px"/>  |  <img src="https://github.com/user-attachments/assets/130c67db-cf7c-4121-9513-2fc642c54c9f" width="250px"/>  |  <img src="https://github.com/user-attachments/assets/1a406ba0-d05e-47d5-9bea-f7f197d58d84" width="250px"/>
</p>


<hr>
<p>
    When there are multiple ways to complete a real-life task, multiple sensorimotor trajectories serve the same goal of achieving that task. If the number of the modalities of the training trajectories increases, modeling them separately, as CNEP does, becomes advantageous. In this comparison, we compared CNEP with ProMP, GMM-GMR, CNMP, and Stable MP (https://github.com/rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning). Also, we included several CNEP variants for a quantitative comparison, which is given on the right. For explanations, please refer to the paper.
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/a7f30709-cf28-487f-9b00-32bd3e5ae96a" height="300px" />  |  <img src="https://github.com/user-attachments/assets/8c6e1729-000c-47ab-adf8-15ec1af9abd4" height="300px" />
</p>

<hr>
<p>
    We trained a CNMP and a CNEP with two experts on two demonstration trajectories. When queried from novel start and end points, CNEP produces trajectories similar to the demonstrations (shown in red and purple). In contrast, other methods (CNMP in this case) produce a mean response (shown in blue). This may lead to suboptimal behavior, as highlighted in the obstacle avoidance tests with a real robot.
</p>
<p align="center">
<img src="https://github.com/yildirimyigit/cnep/blob/master/plots/start.gif" width="300px"/>  |  <img src="https://github.com/yildirimyigit/cnep/blob/master/plots/end.gif" width="300px"/> 
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/2df1623a-7e12-4371-bfe8-355ca9c7fd3d" width="300px"/>  |  <img src="https://github.com/user-attachments/assets/aade4ca4-b74d-4ffd-8c5f-d154e93d1622" width="300px"/>
</p>

<hr>
<p>
    If only a single demonstration trajectory passes through an observation (conditioning) point, both models can successfully synthesize expert-like trajectories. This is the case for the plots in the right column. On the other hand, if there are multiple candidate trajectories passing through the same observation point that the trajectory-producing models are conditioned on, it is reasonable to expect a trajectory close to one of these candidates. Plots on the left show that CNEP successfully picks one of the modes and generates similar trajectories whereas CNMP produces average trajectories. 
</p>
<p align="center">
<img src="https://github.com/yildirimyigit/cnep/assets/3774448/f3bb7363-993e-4b30-97ae-3c52a229f8b0" width="80%" />
</p>

<hr>
<p>
    To demonstrate that CNEP can model MP trajectories of higher dimensions just as easily, we trained a CNEP with a 56-dimensional trajectory of an actual stuntman realizing a cartwheel motion ( [CMU Mocap dataset](http://mocap.cs.cmu.edu/) ). Then, we reproduced the same motion and ran the trajectory on a simulated humanoid.
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/218ae0ba-dbd2-40a3-99c0-fde6064166f3" width="400px"/>
</p>

<hr>
<p>
    Comparison among DMP, Deep DMP and CNEP:
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/5bf3c77f-ae6d-4a2e-aadd-83e923b9128c" width="400px"/>
</p>

## Getting Started
### Requirements
The entire project is developed with
1. Python 3.8
2. Pytorch 2.0.1+cu117

However, most of the code should run cleanly with Python 3.8+ and Pytorch 2+. I tested with Python3.10 and several Pytorch 2+ versions.

### Running
1. Clone the repo
2. Run the training script: for example, _python -u compare_cnp_wta_sine.py_
3. Upon training, run the corresponding test script: _comparison_sine.ipynb_
4. Naming convention: files starting with _compare_ are training scripts whereas files starting with _comparison_ are test scripts.
5. Files with the same name and different extensions:
    1. Files with .ipynb extension are good for inspection & visualization
    2. Files with .py extension are used to run the code on a remote server (an HPC, for example).

If you use the code in your work, kindly consider citing:
Y. Yildirim and E. Ugur, "Conditional Neural Expert Processes for Learning Movement Primitives From Demonstration," in IEEE Robotics and Automation Letters, vol. 9, no. 12, pp. 10732-10739, Dec. 2024, doi: 10.1109/LRA.2024.3477169.
