# HMMToolkit

The **HMMToolkit** is a Julia-based framework for fitting, analyzing, and performing unsupervised anomaly detection 
using both discrete-time and continuous-time (multivariate) hidden Markov models (HMMs).
An application of the continuous-time HMM (CTHMM) for analyzing trip-level vehicle telematics data and 
detecting anomalous driving patterns is detailed in [Chan et al. (2025)](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/assessing-driving-risk-through-unsupervised-detection-of-anomalies-in-telematics-time-series-data/3E84AF8926F86916929FA9BEA807DC93).

This project leverages [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to ensure reproducibility and is authored by [Sophia Chan](https://ianwengchan.github.io/).


## Getting Started  

To reproduce this project locally, follow these steps:  

1. **Download the Code Base**  
   Note: Raw data is typically not included in the git history and may need to be downloaded separately.  

2. **Set Up the Julia Environment**  
   Open a Julia console and run the following commands:  
   ```julia  
   julia> using Pkg  
   julia> Pkg.add("DrWatson") # Install globally for using `quickactivate`  
   julia> Pkg.activate("path/to/this/project")  
   julia> Pkg.instantiate()  
   ```

These commands will install all necessary packages and ensure the environment is correctly configured.  The scripts should work out of the box, including resolving local paths.


## Example Usage

For a practical example of using the HMMToolkit to fit CTHMM and perform anomaly detection, 
refer to the Jupyter Notebook `UAH-example.ipynb`, located in the `notebooks` directory.

## Randomized pseudo-residuals

Both CTHMM and DTHMM ordinary and forecast pseudo-residuals use the conditional
CDF bounds `F(y^-)` and `F(y)`, drawing `U = F(y^-) + V * (F(y) - F(y^-))` with
`V` uniform on `(0, 1)`. For continuous observations the bounds coincide. At a
zero-inflated point mass, this spreads the residual across the probability jump.
Normal residuals are obtained by applying the standard normal quantile to `U`.

The batch functions infer zero inflation from `state_list`, so no `ZI_list` is
needed:

```julia
ordinary, forecast = CTHMM_batch_pseudo_residuals(
    0, df, response_list, Q_mat, π_list, state_list)
ordinary, forecast = DTHMM_batch_pseudo_residuals(
    0, df, response_list, P_mat, π_list, state_list)
```

Use `1` instead of `0` for uniform residuals. Existing batch calls containing
`ZI_list` remain accepted, but the expert types now determine the bounds.
The new low-level interface takes emission probabilities, upper-CDF matrices,
and lower-CDF matrices; obtain the latter with
`CTHMM_precompute_batch_data_emission_cdf_lower_separate`. The previous low-level
interface using `response_list` and `ZI_list` is retained for compatibility.

Run the regression checks with `julia --project=. test/runtests.jl` after
instantiating the project environment.
