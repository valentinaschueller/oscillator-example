# Oscillator Example

Python implementation for the oscillator example for partitioned simulations
This repository can be used to understand and reproduce the results of our paper: 
"A Simple Mass-Spring Test Case for Convergence Order in Time and Energy Conservation of Black-Box Coupling Schemes"


## Notes on the code

### Requirements

- required packages: 
  - NumPy (we used version 1.21.2)
  - Matplotlib (we used version 3.4.3)
  - Pandas (we used version 1.3.5)
- The project requires Python 3.6 or later due to the use of [f-strings, PEP 498](https://peps.python.org/pep-0498/) in the code
- For formatting we used [black](https://github.com/psf/black), for imports [isort](https://pycqa.github.io/isort/). Both can be installed via `pip`.

You can use a conda environment and the file `conda_environment.yml` to have all the necessary packages installed:

```bash
$ conda create --name your_env_name --python=3.8 # or your preferred Python version
$ conda activate your_env_name
$ conda env update --file conda_environment.yml --name your_env_name
```

**Warning: This code was not primarily meant for further development.**
If you really want to use this for anything besides reproducing the results, [please reach out to us.](mailto:valentina.schueller@tum.de).
