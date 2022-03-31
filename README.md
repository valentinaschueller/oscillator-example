# Error Reduction in Coupling Schemes

Python implementation for my seminar paper on "Methods for Error Reduction in Time-Coupled Partitioned Simulations".
I worked on this as part of the M.Sc. seminar "Partitioned Fluid-Structure Interaction and Multiphysics Simulations" in the winter term 2021/22.


## Notes on the code

### Requirements

- required packages: 
  - SymPy (I used version 1.8)
  - NumPy (I used version 1.21.2)
  - Matplotlib (I used version 3.4.3)
- I use f-strings, which require Python 3.6 or higher. I used Python 3.9 for this project but you should be fine with 3.6-3.8 as well
- For formatting I use [black](https://github.com/psf/black), for imports [isort](https://pycqa.github.io/isort/). Both can be installed via `pip`.

You can use a conda environment and the file `conda_environment.yml` to have all the necessary packages installed:

```bash
$ conda create --name your_env_name --python=3.9 # or, e.g., 3.7
$ conda activate your_env_name
$ conda env update --file conda_environment.yml --name your_env_name
```

**Warning: This code was not primarily meant for further development.**
If you really want to use this, understand it, or work on this further, [please reach out to me!](mailto:valentina.schueller@tum.de).
