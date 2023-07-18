# Exceptional coverage and description diversity class association rules
This repository contains the rules produced by the algorithms tested in the submission "Exceptional coverage and description diversity for mining non-dominated class association rules with genetic programming", together with the scripts that process them and generate the tables and figures.

IMPORTANT NOTICE: This sofware has been tested on three ubuntu 22.04 systems.

You just have to unzip data.zip, create a virtual environment, install the required libraries, and run the python scripts. Afterwards, you can open Generated_results.html, for the results generated with the available data (no colorectal dataset) or Original_results.html, for the original results.

```bash
unzip data.zip
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/apriori2latex.py
python scripts/results2latex.py
```

The program tries to apply the non-parametric version of the Scott-Knott-ESD statistical test. For that, it requires the following tools. Otherwise, it will produce the graphs without the application of that test. The required tools are:
- R
- cmake
