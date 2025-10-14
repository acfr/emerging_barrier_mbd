# Emerging-Barrier Model-Based Diffusion

Code release for ["EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments"](https://arxiv.org/abs/2510.07700)

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

# Cite
Please cite this paper as

```
@misc{mishra2025ebmbd,
      title={EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments}, 
      author={Raghav Mishra and Ian R. Manchester},
      year={2025},
      eprint={2510.07700},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07700}, 
}
```
