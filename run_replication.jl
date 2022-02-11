## Top-Level Execution of Scripts
# *******************************
# Reca Sarfati, December 2021
# *******************************

# Base packages
using CSV, DataFrames, Distributions, FileIO, ForwardDiff, JLD2, LinearAlgebra
using Optim, Printf, Plots, Statistics, StatFiles, StatsBase, StatsFuns
# Indulgent packages
using FixedEffectModels, GLM, RegressionTables, QuadGK, Vcov
using Binscatters, CategoricalArrays, CovarianceMatrices
# Imports for shorthand
import ForwardDiff.jacobian, Optim.minimizer
import Vcov.cluster

# TODO: Specify path to working directory!
PATH = "/Users/henrygenighx/Desktop/14.771 - Dev/replication/"

# Build output folders if don't already exist
run(`cd $PATH`)
!isdir("output")        && run(`mkdir output/`)
!isdir("output/plots")  && run(`mkdir output/plots/`)
!isdir("output/tables") && run(`mkdir output/tables/`)

# ***************************
# Replicate Tables & Figures
# ***************************

# Define global constant(s)
F64 = Float64

# Prep labels + keyword shortcuts for table and figure formatting
labels = Dict("logconsumption" => "Log Consumption",
              "showup" => "Showed Up",
              "PMTSCORE" => "Observable Consumption",
              "eps" => "Unobservable Consumption",
              "logc" => "Log Consumption",
              "logc_ST" => "Log Cons. \$\\times\$ self-targeting",
              "close_logc" => "Close \$\\times\$ log cons.",
              "logc & selftargeting" => "Log Cons. \$\\times\$ self-targeting",
			  "logc & self" => "Log Cons. \$\\times\$ self-targeting",
              "selftargeting" => "Self-targeting", "self" => "Self-targeting",
              "getbenefits" => "Log Consumption (OLS)",
              "getbenefit" => "\\substack{Get Benefits\\\\(Logit)}",
			  "get" => "\\substack{Get Benefits\\\\(Logit)}",
              "mistarget" => "\\substack{Error\\\\(Logit)}",
              "excl_error" => "\\substack{Excl. Error\\\\(Logit)}",
              "incl_error" => "\\substack{Incl. Error\\\\(Logit)}",
              "benefit_hyp" => "\\substack{Get Benefits\\\\(Logit)}",
              "mistarget_hyp" => "\\substack{Error\\\\(Logit)}",
              "excl_err_hyp" => "\\substack{Excl. Error\\\\(Logit)}",
              "incl_err_hyp" => "\\substack{Incl. Error\\\\(Logit)}",
              "close" => "Close subtreatment",
              ["inc$i" => "Consumption quintile $i" for i=2:5]...,
              ["closeinc$i" => "Close \$\\times\$ consumption quintile $i" for i=2:5]...,
              "__LABEL_CUSTOM_STATISTIC_comments__" => "Stratum fixed effects",
              "__LABEL_CUSTOM_STATISTIC_means__" => "Mean of Dep. Variable")

table_kwargs = Dict(:labels => labels,
                    :print_estimator_section => false,
                    :print_fe_section => false,
                    :regression_statistics => [:nobs])

# Load helper functions
include("core_functions.jl")
include("gmm.jl")

# Functions which produce tables and figures
include("tables.jl")
include("figures.jl")

###### Tables #######
table_1()
table_3()
table_4()
table_5()
table_6()
table_7()

### Run GMM estimation, generate tables
table_8()
# table_9()
# table_10()

###### Figures #######
figure_1()
figure_2()
figure_3()
figure_4()
figure_5()
figure_6()
