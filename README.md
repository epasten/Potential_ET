# Potential_ET
Different scripts used to estimate potential evapotranspiration using clmate model (netcdf) data as input

Files:
1) pet_estimation.py:
    Uses netcdf outputs from several climate models and variables to estimate potential evapotranspiration based on 10 different methodologies
2) pet_postprocessing.py:
    Uses the output files saved by (1) to provide an annual accumulated potential evapotranspiration plot and a monthly accumulated pet plot. To add plots showing the Signal to       Noise Ratios and trend analysis
3) normality.py:
    Used to assess whether the distribution of the variables fit to a normal distribution
4) bias correction:
    Different bias correction methods: delta change, distribution based scaling and multivariate BC
5) Machine Learning methods
