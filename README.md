# Exceptional coverage and description diversity in non-dominated class association rules
This repository contains the rules produced by the algorithms tested in the submission "Exceptional coverage and description diversity in non-dominated class association rules", together with the scripts that process them and generate the tables and figures.

## Figures and tables in the submission
1. Open `Original_results.html`

## Figures and tables generated from the available data (no colorectal dataset)
1. Open `Generated_results.html`

## You really want you generate the figures yourself

You just have to create a virtual environment, install the required libraries, and run the python scripts. Afterwards, you can open `Generated_results.html`, for the results generated with the available data (no colorectal dataset).

```bash
virtualenv venv

# The sofware needs to import package apt to check the requirements below IMPORTANT NOTICE
sed -i "s/include-system-site-packages = false/include-system-site-packages = true/g" venv/pyvenv.cfg

source venv/bin/activate
pip install -r requirements.txt
python scripts/apriori2latex.py
python scripts/results2latex.py
```

**IMPORTANT NOTICE:** This sofware has been successfully tested on three ubuntu 22.04 systems. In any case, be aware that the software is provided "as is", without warranty of any kind, express or implied. Please, read the license.

In addition, be aware that the program tries to apply the non-parametric version of the Scott-Knott-ESD statistical test (available at https://github.com/klainfo/ScottKnottESD). It is highly recommended that you follow the instructions at https://github.com/klainfo/ScottKnottESD - "Install from python (by calling R package via rpy2)", before running `apriori2latex.py` or `results2latex.py`, to make these two scripts able to compute the test. But nevertheless, these script can try to locate the required packages, or ask you if they should try to install them locally. For such installation, the software checks that the following tools are installed (this needs to use the python package apt):
- R (r-base and r-base-core)
- cmake
- libcurl4-openssl-dev
- gfortran
- libblas-dev
- liblapack-dev

Shall the Scott-Knott-ESD test not be available, the scripts can still produce the graphs and tables without the output of the test.
