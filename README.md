# Error Reduction in Coupling Schemes

Python implementation for my seminar paper on "Methods for Error Reduction in Time-Coupled Partitioned Simulations".
I worked on this as part of the M.Sc. seminar "Partitioned Fluid-Structure Interaction and Multiphysics Simulations" in the winter term 2021/22.

[Look here](https://github.com/valentinaschueller/fsi-seminar-paper) for the final paper and my presentation.

## Notes on the code

### Requirements

- required packages: 
  - SymPy (I used version 1.8)
  - NumPy (I used version 1.21.2)
  - Matplotlib (I used version 3.4.3)
- I use f-strings, which require Python 3.6 or higher. I used Python 3.9 for this project but you should be fine with 3.6-3.8 as well

You can use a conda environment and the file `conda_environment.yml` to have all the necessary packages installed:

```bash
$ conda create --name your_env_name --python=3.9 # or, e.g., 3.7
$ conda activate your_env_name
$ conda env update --file conda_environment.yml --name your_env_name
```

### How to use it:

**Warning: This code was not primarily meant for further development.**
If you really want to use this, understand it, or work on this further, [please reach out to me!](mailto:valentina.schueller@tum.de).

The code consists of Python scripts which create the plots used in the paper and presentation.
The following scripts are primarily meant for execution:

- `analytical_solution/sympy_analytical_solution.py`: create SymPy plots for the solution of the toy ODE system with different initial conditions and parameters. Unless you want to reproduce my plots, this is maybe the most helpful script for whatever you want to play around with :)
- `analytical_solution/plot_analytical_solution.py`: create the plots used in the paper and presentation showing the analytical solution
- `same_timescales/same_timescales_monolithic.py`: monolithic solution (same time scales case)
- `same_timescales/partitioned_*.py`: partitioned simulation (same time scales case)
- `diff_timescales/diff_timescales_monolithic.py`: monolithic solution (different time scales case)
- `diff_timescales/partitioned_*.py`: partitioned simulation (diff time scales case --> subcycling). These scripts do not work as expected, I did not use these plots in the paper but they can serve as a start for further work.
- everything that ends in `_pres` creates the presentation version of the plot, they are otherwise equivalent to the plots I used in the paper
