# Decentralized Markets and the Emergence of Housing Wealth Inequality
## by Omar A. Guerrero

This repository contains the code and data used in the paper: *Decentralized Markets and the Emergence of Housing Wealth Inequality*.


## Code


## Benchmark data
The file `params_theo.csv` contains the theoretical values for the benchmark analysis performed in section 3 of the paper.

## Empirical data

Unfortunatelly, I am unabe to post the empirical micro-data used in sections 4 and 5 of the paper, as doing so would violate the agreement established with the UK Data Service.
However, these data can be downloaded from its original source as long as you register with the UK Data Service as an accredited researcher (it does not matter if you do not work in the UK).
Here you can find more information: https://beta.ukdataservice.ac.uk/datacatalogue/series/series?id=2000056

The UK life tables are provided by the Office of National Statistics. 
The values used in the paper can be found int he file `life_table.csv` and they correspond to the probability of **not surviving** the next year, conditional on age and sex (*m* for make and *f* for female).

The ONS house price index used for the validation in section 4.3 can be found in the file `housing_prices.csv`.
I computed the average index across all dates by region.
The regions are:
- 1
- 2
- 3

