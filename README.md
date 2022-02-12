# Replication of "Self Targeting: Evidence from a Field Experiment in Indonesia"

This repository contains Julia code for replicating all main figures and tables of  "[Self-Targeting: Evidence from a Field Experiment in Indonesia](https://www.journals.uchicago.edu/doi/10.1086/685299)," by Vivi Alatas, Ririn Purnamasari, and Matthew Wai-Poi, Abhijit Banerjee and Benjamin A. Olken, and Rema Hanna. This code takes as input data these authors have graciously made public through [Harvard Dataverse](https://doi.org/10.7910/DVN/6ZUIUC).

## Usage

To run this code, first clone/download this repository into a directory of your choice. If the input files are not automatically downloaded into the `input/` directory, proceed to [Harvard Dataverse](https://doi.org/10.7910/DVN/6ZUIUC), download the replication codes, and copy over the following files into `input/`:
- `costs.dta`
- `matched_baseline.dta`
- `matched_midline.dta`
- `MATLAB_Input_small_sample.csv`
- `MATLAB_Input.csv`

Next, open the file `run_replication.jl`, and enter the location of your working directory for the variable `PATH`.

## Versioning

This code is currently compatible with [Julia v1.6.3](https://julialang.org/downloads/#long_term_support_release). Be sure to also add the packages included at the top of `run_replication.jl`. :ok_woman:

To install a package, enter the Julia REPL, type `]` to enter package manager, and then `add` any packages you need. It might be necessary to restart your Julia session once finished to access all updates. Example below:

```julia
pkg> add DataFrames
```

## Disclaimer

This repository is heavily ported from code available on Dataverse, originated by this papers' authors: Vivi Alatas, Ririn Purnamasari, and Matthew Wai-Poi, Abhijit Banerjee and Benjamin A. Olken, and Rema Hanna. Any reproduction, use, modification, and distribution of this code, in whole or in part, must retain this notice in the documentation associated with any distributed works. Any portions of the code attributed to third parties are subject to applicable third party licenses and rights. By your use of this code you accept this license and any applicable third party license.

## More Aggressive Disclaimer

Deviations from the Dataverse code are the product of own machinations; all aberrant results/bizarre output/suspicious inconsistencies with what is reported in the paper should be attributed to this graduate student's intellectual shortcomings and lousy sleep schedule.
