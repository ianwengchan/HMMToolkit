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
