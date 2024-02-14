# Conditional Neural Expert Processes (CNEP)

This repository contains the source code for the paper "Conditional Neural Expert Processes for Learning from Demonstration" by [Yigit Yildirim](https://www.cmpe.boun.edu.tr/~yigit.yildirim/) and [Emre Ugur](https://www.cmpe.boun.edu.tr/~emre). We both are members of the [CoLoRs Lab, Bogazici University](https://clrslab.wordpress.com/).

CNEP is a novel deep learning architecture for Learning from Demonstration (LfD) in robotics aiming to encode diverse sensorimotor trajectories from demonstrations with varying movements, leveraging a novel gating mechanism, multiple decoders, and an entropy-based loss calculation to promote decoder specialization. You can find the paper here: https://arxiv.org/abs/2402.08424

##### Architecture overview:

![__over](https://github.com/yildirimyigit/cnep/assets/3774448/4147b4f6-f29c-499a-a119-a5fb31b10aae)

##### Some results:

![comb_example_1](https://github.com/yildirimyigit/cnep/assets/3774448/f3bb7363-993e-4b30-97ae-3c52a229f8b0)


Here are some videos from this work: https://youtube.com/playlist?list=PLXWw0F-8m_ZZD7fpGOKclzVJONXUifDiY&si=vR7OLhx_rigMxdLK

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
5. Files with the same name and different extensions
  1. Files with .ipynb extension are good for inspection & visualization
  2. Files with .py extension are for running the code on a remote server (an HPC, for example).

If you use the code in your work, please cite https://arxiv.org/abs/2402.08424
