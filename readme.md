## Linkage Synthesis Challenge Problem
## test commit

This challenge problem is focused on synthesizing planar linkage mechanisms such that a specific output curve is traced using the mechanism. For this project you are tasked with synthesizing linkage mechanisms to trace 6 different output shapes. Further you are tasked with synthesizing mechanisms such that the total material used for the mechanisms in minimized. 

<img src="https://i.ibb.co/qsPC0gC/2021-09-13-0hl-Kleki.png" alt="Numbered Mechanism" border="0">

## Python Requiremnets
we provide three different requirement files:
- `requirements_CPU.txt`: If you do not have GPU use this.
- `requirements_GPU.txt`: If you want to use CUDA GPU use this. (You will also have to adjust the code since it uses CPU by default)
- `requirements_MAC_M.txt`: If you have an M series mac and want to use the M (Metal) series acceleration use this. This is an experimental package so you may just be stuck with the CPU version.

To setup environments first create a new environmnet in conda/mamba:

```bash
conda create --name CP1 python=3.10
```

Then activate the environment and install packages using pip:

```bash
conda activate CP1
pip install -r requirements_CPU.txt
```
