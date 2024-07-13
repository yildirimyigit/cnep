# Conditional Neural Expert Processes (CNEP)

This repository contains the source code for the paper "Conditional Neural Expert Processes for Learning Movement Primitives from Demonstration" by [Yigit Yildirim](https://www.cmpe.boun.edu.tr/~yigit.yildirim/) and [Emre Ugur](https://www.cmpe.boun.edu.tr/~emre). We both are members of the [CoLoRs Lab, Bogazici University](https://clrslab.wordpress.com/).

CNEP is a novel deep learning architecture for Learning from Demonstration (LfD) in robotics aiming to encode diverse sensorimotor trajectories from demonstrations with varying movements, leveraging a novel gating mechanism, multiple decoders, and an entropy-based loss calculation to promote decoder specialization. This work has been submitted to the _IEEE RA-L_ for possible publication on **July 5, 2024**. You can find the preprint here: [https://arxiv.org/abs/2402.08424](https://arxiv.org/abs/2402.08424)

### Architecture overview:

![__over](https://github.com/yildirimyigit/cnep/assets/3774448/4147b4f6-f29c-499a-a119-a5fb31b10aae)

### Some results:
Here are some videos from this work: https://youtube.com/playlist?list=PLXWw0F-8m_ZZD7fpGOKclzVJONXUifDiY
<hr>
<p>
    We assessed the performance of CNEP in comparison to Probabilistic Movement Primitives (ProMP) and Gaussian Mixture Models-Gaussian Mixture Regression (GMM-GMR) on a complex robotic task involving grasping and placing wine glasses onto a dish rack. This task involved high-dimensional sensorimotor trajectories of 1288 dimensions. Each model was trained on 40 expert demonstrations and expected to generate the necessary control commands to achieve successful task completion.  CNEP demonstrated superior performance by successfully completing the task, while ProMP and GMM-GMR were unable to achieve a successful grasp of the glass. A video demonstrating the results is available for further analysis: https://youtu.be/ffnIhrmjwgo
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/572980a1-0f43-404c-8bfa-1811b27af4c0"/>
</p>
<hr>
<p>
Trained on only two trajectories, CNEP produces trajectories similar to the demonstrations. In contrast, other methods (CNMP in this case) produce a mean response, which may lead to suboptimal behavior, as shown in the obstacle avoidance test.
</p>
<p align="center">
<img src="https://github.com/yildirimyigit/cnep/blob/master/plots/start.gif" width="300px"/>  |  <img src="https://github.com/yildirimyigit/cnep/blob/master/plots/end.gif" width="300px"/> 
</p>
<hr>
<p align="center">
<img src="https://github.com/yildirimyigit/cnep/assets/3774448/f3bb7363-993e-4b30-97ae-3c52a229f8b0" width="80%" />
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

If you use the code in your work, kindly consider citing https://arxiv.org/abs/2402.08424
