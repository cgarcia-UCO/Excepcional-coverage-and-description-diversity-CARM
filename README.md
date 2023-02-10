# Non-dominated-AssocRules-GECCO2023
This repository contains the rules produced by the algorithms tested in the GECCO 2023 submission "Exceptional coverage and description diversity for mining non-dominated class association rules with genetic programming", together with the scripts that process them and generate the tables and figures.

You just have to unzip data.zip, create a virtual environment, install the required libraries, and run the corresponding program:

   unzip data.zip
   virtualenv venv
   pip install -r requirements.txt
   source venv/bin/activate
   python scripts/results2latex.py

