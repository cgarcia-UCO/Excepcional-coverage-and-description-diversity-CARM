# Exceptional coverage and description diversity class association rules
This repository contains the rules produced by the algorithms tested in the GECCO 2023 submission "Exceptional coverage and description diversity for mining non-dominated class association rules with genetic programming", together with the scripts that process them and generate the tables and figures.

You just have to unzip data.zip, create a virtual environment, install the required libraries, and run the python scripts. Afterwards, you can open Generated_results.html, for the results generated with the available data (no colorectal dataset) or Original_results.html, for the original results.

```bash
unzip data.zip
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/apriori2latex.py
python scripts/results2latex.py
```
