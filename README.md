# Oscillator Example

Python implementation for the oscillator example for partitioned simulations
This repository can be used to understand and reproduce the results of our paper:
"A Simple Test Case for Convergence Order in Time and Energy Conservation of Black-Box Coupling Schemes"


## Notes on the code

### Requirements

- required packages:
  - NumPy (we used version 1.21.2)
  - Matplotlib (we used version 3.4.3)
  - Pandas (we used version 1.3.5)
- The project requires Python 3.6 or later due to the use of [f-strings, PEP 498](https://peps.python.org/pep-0498/) in the code
- For formatting we use [autopep8](https://github.com/hhatto/autopep8), for imports [isort v5.10.1](https://pycqa.github.io/isort/). Refer to [our CI pipeline](https://github.com/valentinaschueller/oscillator-example/blob/main/.github/workflows/check-pep8.yml) for the options in use.

You can use a conda environment and the file `conda_environment.yml` to have all the necessary packages installed:

```bash
$ conda create --name your_env_name --python=3.8 # or your preferred Python version
$ conda activate your_env_name
$ conda env update --file conda_environment.yml --name your_env_name
```

**Warning: This code was not primarily meant for further development.**
If you really want to use this for anything besides reproducing the results, [please reach out to us.](mailto:valentina.schueller@tum.de)

### Running and generating results

This repository provides scripts to run the experiments from the paper in the folder `numerical_studies`. The following scripts are provided:

* `partitioned_explicit.py`
* `partitioned_implicit.py`
* `partitioned_strang.py`
* `partitioned_waveform.py`
* `energy_explicit.py`
* `energy_implicit.py`
* `energy_strang.py`
* `energy_waveform.py`

Each of these scripts can be run by executing `python3 <name of script>.py` from the terminal. This will create `.csv` files containing the results in the folder `numerical_studies/results`.

### Postprocessing of the results

To postprocess the results from `numerical_studies/results`, you can compile the latex files that are provided in subfolders of `numerical_studies/plotting`. The subfolders are named according to the figures from the paper.
