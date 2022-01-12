## Top-Level Execution of Scripts
# *******************************
# Reca Sarfati, December 2021
# *******************************

# Base packages
using CSV, DataFrames, Distributions, FileIO, ForwardDiff, LinearAlgebra
using Optim, Printf, Plots, Statistics, StatFiles, StatsBase
# Indulgent packages
using FixedEffectModels, GLM, RegressionTables, Vcov
using Binscatters, CategoricalArrays, CovarianceMatrices
import ForwardDiff.jacobian, Optim.minimizer
import Vcov.cluster
#import StatsBase.vcov, CovarianceMatrices.CRHC0

# Switch to desired working directory (user-specified file path)
run(`cd /Users/henrygenighx/Desktop/14.771\ -\ Dev/replication`)

# Build output folders if don't already exist
!isdir("output")        && run(`mkdir output/`)
!isdir("output/plots")  && run(`mkdir output/plots/`)
!isdir("output/tables") && run(`mkdir output/tables/`)

# ***************************
# Replicate Tables & Figures
# ***************************

# Define global constant(s)
F64 = Float64

# Prep for output formatting
labels = Dict("logconsumption" => "Log Consumption",
              "showup" => "Showed Up",
              "PMTSCORE" => "Observable Consumption",
              "eps" => "Unobservable Consumption",
              "logc" => "Log Consumption",
              "logc_ST" => "Log Cons. x self-targeting",
              "logc & selftargeting" => "Log Cons. x self-targeting",
              "selftargeting" => "Self-targeting",
              "getbenefits" => "Log Consumption (OLS)",
              "getbenefit" => "\\substack{Get Benefits\\\\(Logit)}",
              "mistarget" => "\\substack{Error\\\\(Logit)}",
              "excl_error" => "\\substack{Excl. Error\\\\(Logit)}",
              "incl_error" => "\\substack{Incl. Error\\\\(Logit)}",
              "__LABEL_CUSTOM_STATISTIC_comments__" => "Stratum fixed effects",
              "__LABEL_CUSTOM_STATISTIC_means__" => "Mean of Dep. Variable")

table_kwargs = Dict(:labels => labels,
                    :print_estimator_section => false,
                    :print_fe_section => false,
                    :regression_statistics => [:nobs])

# Load helper functions
include("helper_functions.jl")

# Run GMM estimation
#include("gmm.jl")

# Functions which produce tables and figures
include("tables.jl")
include("figures.jl")

###### Build Tables and Figures ##########
# figures = [eval(Meta.parse("figure_$i")) for i in [1:6...]]
# tables  = [eval(Meta.parse("table_$i"))  for i in [1, 3:10...]]
# for result in vcat(figures, tables)
#     result()
# end

###### Tables #######

table_1()
table_3()
table_4()
table_5()
table_6()
table_7()
# table_8()
# table_9()
# table_10()

###### Figures #######

figure_1()
figure_2()
figure_3()
# figure_4()
# figure_5()
# figure_6()
