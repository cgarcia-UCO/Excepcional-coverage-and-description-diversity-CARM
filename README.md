# Exceptional coverage and description diversity in class association rule mining
This repository contains the rules produced by the algorithms tested in the submission "Exceptional coverage and description diversity for mining non-dominated class association rules with genetic programming", together with the scripts that process them and generate the tables and figures.

## Figures and tables in the submission
Open Original_results.html

## Figures and tables generated from the available data (no colorectal dataset)
Open Generated_results.html

## You want you generate the figures yourself?

You just have to unzip data.zip, create a virtual environment, install the required libraries, and run the python scripts. Afterwards, you can open Generated_results.html, for the results generated with the available data (no colorectal dataset) or Original_results.html, for the original results.

```bash
unzip data.zip
virtualenv venv

# The sofware needs to import package apt to check the requirements below IMPORTANT NOTICE
sed -i "s/include-system-site-packages = false/include-system-site-packages = true/g" venv/pyvenv.cfg

source venv/bin/activate
pip install -r requirements.txt
python scripts/apriori2latex.py
python scripts/results2latex.py
```

**IMPORTANT NOTICE:** This sofware has been tested on three ubuntu 22.04 systems. In any case, be aware that the software is provided "as is", without warranty of any kind, express or implied. Please, read the license.

In addition, the program tries to apply the non-parametric version of the Scott-Knott-ESD statistical test (available at https://github.com/klainfo/ScottKnottESD). The software will try to locate it, or ask you if it should try to install it locally. For such installation, the software checks that the following tools are installed (this needs to use package apt):
- R
- cmake
- libcurl4-openssl-dev
- gfortran
- libblas-dev
- liblapack-dev

Shall the Scott-Knott-ESD test not be available, the software would produce the graphs and tables without the output of the test.
