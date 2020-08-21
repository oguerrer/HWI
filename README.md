# Decentralized Markets and the Emergence of Housing Wealth Inequality
## by Omar A. Guerrero

This repository contains the code and data used in the paper: *Decentralized Markets and the Emergence of Housing Wealth Inequality*.


## Code
The file `model.py` contains all the functions and classes to run the model.
The `example.ipynb` file is a Jupyter notebok with a script that shows how to run the model.


## Benchmark data
The file `params_theo.csv` contains the theoretical values for the benchmark analysis performed in section 3 of the paper.


## Empirical data

Unfortunatelly, I am unabe to post the empirical micro-data used in sections 4 and 5 of the paper, as doing so would violate the terms of agreement established by the UK Data Service.
However, these data can be downloaded from its original source as long as you register with the UK Data Service as an accredited researcher (it does not matter if you do not work in the UK).
Here you can find more information: https://beta.ukdataservice.ac.uk/datacatalogue/series/series?id=2000056

The UK life tables are provided by the Office of National Statistics. 
The values used in the paper can be found int he file `life_table.csv` and they correspond to the probability of **not surviving** the next year, conditional on age and sex (*m* for make and *f* for female).

The ONS house price index used for the validation in section 4.3 can be found in the file `housing_prices.csv`.
I computed the average index across all dates by region.
The regions have the following numerical codes:
- 1 North East (NE)
- 2 North West (NW)
- 4 Yorkshire and the Humber (YH)
- 5 East Midlands (EM)
- 6 West Midlands (WM)
- 7 East of England (EE)
- 8 London (LN)
- 9 South East (SE)
- 10 South West (SW)
- 11 Wales (WL)
- 12 Scotland (SC)



