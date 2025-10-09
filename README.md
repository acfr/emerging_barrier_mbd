# Emerging-Barrier Model-Based Diffusion

Code release for "EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments"

![EB-MBD for UVMS](assets/ebmbd.gif)

# Installation
Follow instructions to install [poetry](https://python-poetry.org/) and run
```
poetry install
```

Test everything works by running on 2D obstacle avoidance problem
```
mkdir results
cd results
poetry run python3 ../scripts/obs2d_trajopt.py comparison
```
