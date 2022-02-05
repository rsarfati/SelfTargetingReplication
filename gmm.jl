"""
```
compute_quantiles(df::DataFrame;
                  N_q = Dict([:consumption, :pmtscore, :unobs_cons, :distt] .=>
                             [5, 3, 3, 4]]))
```
Computes quantiles of variables in inputted dictionary, and puts the associated
categorical labels into dataframe for series. Default inputs are observed consumption,
PMT, unobs. consumption (w), distance

Corresponds to `load_data.m` file in replication code.
"""
function compute_quantiles(df::Union{DataFrame, DataFrameRow}; N_q = Dict([:consumption, :pmtscore, :unobs_cons,
                                                      :distt] .=> [5, 3, 3, 4]))
    # Helper functions
    quant(N::Int64)     = [n/N for n=1:N-1]
    assign_q(x, quants) = [minimum(vcat(findall(>=(x[i]), quants),
                                        length(quants)+1)) for i=1:length(x)]

    # Compute unobs. consumption (residual regressing log(obs consumption) on PMT)
    !("unobs_cons" in names(df)) && insertcols!(df, :unobs_cons =>
                                    residuals(reg(df, @formula(log_c ~ pmtscore)), df))

    # Assign categorical IDs for quantiles
    for v in keys(N_q)
        v_name = Symbol(string(v) * "_q")
        insertcols!(df, v_name => assign_q(df[:,v], quantile(df[:,v], quant(N_q[v]))))
    end
    return df
end

"""
```
compute_moments(df0::D, showup::Vector{T}) where {T<:Number,D<:DataFrame}
```
Moments:
1-10.  mean showup rates in (measured) consumption quintiles in far and close
       subtreatment (separately). -> Ten moments
11-14. mean(showup-showuphat) of four extreme cells in grid of [tertiles of
       pmt * tertiles of w] (residual from the regression of log(consumption)
       on pmtscore) -> Four moments
15-16. two extreme cells in quartiles of distance -> Two moments
17-20. λ moments -> Four moments
"""
function compute_moments(df0::D, showup_hat::Union{T,Vector{T}}, true_λ, belief_λ,
                         induced_λ) where {T<:Union{Float64,Int64}, D<:Union{DataFrame, DataFrameRow}}

    # Initialize dictionary, manage NaNs in data
    moments = Vector{F64}(undef, 20)
    showup  = (df0.showup .== 1) .& .!ismissing.(df0.showup)

    # 1-10 Far/close subtreatment x i=1:5th quintile of consumption
    for i=1:5
        far_i   = (df0.consumption_q .== i) .&   iszero.(df0.close)
        close_i = (df0.consumption_q .== i) .& .!iszero.(df0.close)
        moments[i]   = (sum(showup[far_i])   - sum(showup_hat[far_i]))   / sum(far_i)
        moments[5+i] = (sum(showup[close_i]) - sum(showup_hat[close_i])) / sum(close_i)
    end

    # 11-14 {Top, bottom tercile} x {observable, unobservable consumption}
    for (i, Q) in enumerate([[3,1], [3,3], [1,1], [1,3]])
        idx = (df0.pmtscore_q .== Q[1]) .& (df0.unobs_cons_q .== Q[2])
        moments[10 + i] = (sum(showup[idx]) - sum(showup_hat[idx])) / sum(idx)
    end

    # 15-16 Top and bottom distance quartiles
    T_D = (df0.distt_q .== 4)
    B_D = (df0.distt_q .== 1)

    moments[15] =  (sum(showup[T_D]) - sum(showup_hat[T_D])) / sum(T_D)
    moments[16] =  (sum(showup[B_D]) - sum(showup_hat[B_D])) / sum(B_D)

    # 17-20 Mean λ function moments
    N_show   = sum(showup)
    moments[17] = sum((belief_λ .- df0.getbenefit) .* showup) / N_show
    moments[18] = sum((belief_λ .- df0.getbenefit) .*
                        (df0.log_c .- mean(df0.log_c)) .* showup) / N_show
    moments[19] = sum((induced_λ .- df0.getbenefit) .* showup) / N_show
    moments[20] = sum((induced_λ .- df0.getbenefit) .*
                        (df0.log_c .- mean(df0.log_c)) .* showup) / N_show

    # If showup hat exactly zero, the λ moments are NaN; replace w/ large number
    moments[isnan.(moments)] .= 10000.
    return moments
end

"""
Evaluate probability of showing up for each i. (Eq 22)
"""
function showuphat(df::Union{DataFrame,DataFrameRow}, t::Vector{T}, η_sd::T,
                   δ::T, μ_con_true::T, μ_β_true::T, λ_con_true::T, λ_β_true::T;
                   N_grid = 100) where {T<:F64}

    # Unpack parameters
    μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief = t

    # Convert mean and standard deviation into α and β
    s  = sqrt(3 * (σ_ϵ ^ 2) ./ (pi ^ 2))
    αa = μ_ϵ ./ s
    β  = 1 ./ s

    # Lower and upper bound, grid density for the numerical integration over eta
    lb = -η_sd*4
    ub = -lb
    N  = length(df.consumption)

    function util(η::F64)
        # Present Utility (Cost function adjusted because "totalcost" given is based on noisy consumption measure.)
        relu_2day = (df.consumption .* exp.(-η) - df.totcost_pc .* exp.(-η) +
                    (1 .- 1 .* exp.(-η)) .* df.moneycost) - (df.consumption .* exp.(-η))
        # Future Utility: note that cdf(Normal(),) in the middle is mu function
        relu_2mor = (df.consumption .* exp.(-η) .+ df.benefit) - (df.consumption .* exp.(-η))

        Mu = cdf.(Normal(), μ_con_true .+ df.FE2 .+ μ_β_true * (df.pmtscore .- η))
        Λ = cdf.(Normal(), λ_con_belief .+ λ_β_belief * (log.(df.consumption) .- η))

        prob_s = (1 .- 1 ./ (1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Mu .* relu_2mor) .+ αa)))
        prob_u = (1 .- 1 ./ (1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Λ  .* relu_2mor) .+ αa)))

        return (α * prob_s + (1 - α) * prob_u) .* pdf.(Normal(0, η_sd), η)
    end
    # Gaussian quadrature with Legendre orthofonal polynomials
    showup_hat = -(util(lb) + util(ub))
    for η_i in range(lb, stop=ub, length=N_grid)
        showup_hat += 2*util(η_i)
    end
    showup_hat *= 0.5*(ub-lb)/100
    #showup_hat, err = quadgk(util, lb, ub, rtol=1e-4)

    # Rather than running probit, apply WLS.
    # Calculate inverse of mu Phi(-1)(mu) where Phi is standard normal CDF
    muinv     = μ_con_true .+ df.FE2 .+ μ_β_true .* df.pmtscore
    const_t   = sqrt.(showup_hat)   # Conversion for the weight
    muinv_t   = const_t .* muinv    # Conversion for the frequency weight
    logcons_t = const_t .* df.log_c # Conversion for the weight
    X         = hcat(const_t, logcons_t)

    sigma2    = sqrt(sum(((Matrix{F64}(I, N, N) - X / (X' * X) * X') * muinv_t) .^ 2) / (N - 2))
    coef      = 1 / sigma2 * (X' * X) \ X' * muinv_t # Divide by sigma to impose sd=1 for error
    induced_λ = cdf.(Normal(), hcat(ones(N), df.log_c) * coef)

    return showup_hat, induced_λ
end

"""
Two Stage Feasible GMM for Targeting Paper

# Parameters:
# 1~2. mean, variance of epsilon (utility shock)
# 3.   sigma (coef of relative risk aversion = rho)
# 4.   α (fraction of people who are sophisticated)
# 5~6. constant and coefficient for the λ function (probability of
       getting benefit given showing up)

## Minimize Objective Function Locally From Many Starting Values
#	Use matlab's built-in lsqnonlin function to optimize from a random set
#	of initial values.

## How are moments computed?
#	For each obs, calculate showup_hat as the probability that eps>-gain
#	(i.e. that gain+epsilon>0) This is just 1 - the cdf
#	of epsilon evaluated at (-gain_i).
"""
function GMM_problem(df0::DataFrame, danual; δ_mom = 0., irate= 1.22, η_sd = 275, VERBOSE = true)

    μ_con_true = df0.reg_const2[1]
    μ_β_true   = df0.reg_pmt2[1]
    λ_con_true = df0.reg_nofe_const[1]
    λ_β_true   = df0.reg_nofe_logcon[1]

    df = df0[:, [:totcost_pc, :moneycost, :consumption, :log_c, :distt, :pmtscore,
                :FE2, :getbenefit, :benefit, :showup, :close]]
    df = compute_quantiles(df)

    N   = size(df, 1) # No. of households
    N_m = 20          # No. of moment conditions
    N_p = 5           # No. parameters to estmate

    # Compute NPV δ (yearly)
    irm = 1 / irate
    δ = danual * (1 + irm + irm^2 + irm^3 + irm^4 + irm^5)

    """
    Difference between empirical and estimated moments.
    """
    function g(t::Vector{T}, df1::D) where {T<:Float64, D<:Union{DataFrame, DataFrameRow}}
        showup_hat, induced_λ = showuphat(df1, t, η_sd, δ, μ_con_true, μ_β_true,
                                          λ_con_true, λ_β_true)
        true_λ    = cdf.(Normal(), λ_con_true   .+ λ_β_true   .* df1.log_c)
        belief_λ  = cdf.(Normal(), t[4] .+ t[5] .* df1.log_c)
        return compute_moments(df1, showup_hat, true_λ, belief_λ, induced_λ)
    end

    ### GMM: First Step (Initial values nabbed from Matlab)
    lb1 = [-200000,     0,  0.001,  0, -2]
    ub1 = [ 200000, 200000, 0.999, 20,  1]

    # Currently imposing some function of random rho + noise
    μ_ϵ_0 = -26215 .* (0.8 .+ 0.4 .* rand())
    σ_ϵ_0 =  26805 .* (0.8 .+ 0.4 .* rand())
    α_0   =  0.001 .+ 0.999 * rand()

    # "get benefit" function, λ for the naive people
    λ_con_0 = λ_con_true  * (0.8 .+ 0.4 * rand())
    λ_β_0   = λ_β_true    * (0.8 .+ 0.4 * rand())

    t0 = [-79681, 59715, 0.5, 8.04, -0.72]#[μ_ϵ_0, σ_ϵ_0, α_0, λ_con_0, λ_β_0]

    # GMM First step: begin with identity weighting matrix
    W0 = Matrix{Float64}(I, N_m, N_m)
    function gAg(x::Vector{F64}, A::Matrix{F64})
        g_eval = g(x, df)
        return abs.(g_eval' * A * g_eval)
    end
    println("First Stage: (takes approx. 1 min)")
    #t1 = minimizer(optimize(x -> gAg(x, W0), lb1, ub1, t0, NelderMead(),
    #                        Optim.Options(f_tol=1e-2, show_trace = VERBOSE)))

    t1 = [-61298.74693924103, 23479.892469625243, 0.5083359032616762, 2.7728175583383488, -0.3194655456909087]
    @show t1
    # GMM: Second stage
    g1 = g(t1, df)
    Om = inv(g1 * g1' / N)
    # Return final theta
    println("Second stage: (takes approx. 1 min)")
    return minimizer(optimize(x -> gAg(x, Om), lb1, ub1, t1, NelderMead(),
                              Optim.Options(g_tol=1e5, show_trace = VERBOSE)))
end

"""
Table 8. Estimated Parameter Values for the Model // estimation_1.m
Input:  MATLAB_Input.csv
Output: MATLAB_Estimates_main_d**.csv
** corresponds to the value of the annual discount factor:
   d=0.82 = 1/irate, d=0.50 or d=0.95
"""
function estimation_1(; irate = 1.22, η_sd = 0.275, δ_mom = 0.0, VERBOSE = true)
    # Load data
    df = CSV.read("input/MATLAB_Input.csv", DataFrame, header = true)
    insertcols!(df, :log_c => log.(df.consumption))
    rename!(df, [:closesubtreatment => :close])
    clean(df, [:log_c, :close, :getbenefit, :pmtscore, :distt], F64)
    @show names(df)

    # Run three estimations with varying discount factors
    for d_annual in [1/irate, 0.5, 0.95]
        min_t = GMM_problem(df, d_annual; δ_mom = δ_mom, irate = irate,
                               η_sd = η_sd, VERBOSE = VERBOSE)
        #min_t = [-61298.74678900998, 23479.89264016293, 0.5083359051349494, 2.772817555169909, -0.3194655456767444]
        temp     = round(d_annual*100)
        @show min_t
        CSV.write("output/MATLAB_Estimates_main_d$temp.csv", Tables.table(min_t))
    end

    # Standard errors // run bootstrapping.m
    # Input:  MATLAB_Input.csv
    # Output: bootstrap_**\bootstrap_[something]_d**.csv files
    N  = size(df, 1) # No. of households
    danual = 1/irate # alternatives: danual = 0.50 and = 0.95.
    N_m    = 20  # number of moments
    n_boot = 100 # Number of bootstrap iterations
    N_init = 50  # Number of N_init (initial conditions) per iteration

    # store bootstrap subsample, estimated parameters, weighting matrices,
    # and all N_init for each iteration
    boot_samples  = zeros(n_boot, N)
    θ_boots       = zeros(n_boot, N_m)
    weight_matrix = zeros(n_boot * N_m, N_m)
    θ_boots_all   = zeros(n_boot * N_init, N_m)

    for it=1:n_boot
        # Randomly draw households (with replacement) for bootstrap
        boot_sample        = sample(1:N, N, true)
        boot_samples[it,:] = boot_sample # Save sample

        W, p_all = GMM_problem(δ_mom, danual, irate, η_sd, N_init, boot_sample)

        Fval_ss       = p_all[:, 17]
        minindex      = argmin(Fval_ss)
        θ_boots[it,:] = p_all[minindex,:]
        θ_boots_all[   (it-1) * N_init + (1:N_init),:] = p_all
        weight_matrix[ (it-1) * N_m + (1:N_m),:] = W

        @show "Bootstrap iteration: $i. Last iteration took $i sec."

        # write output so far (overwrite)
        temp = round(danual*100)
        CSV.write("output/bootstrap_$(temp)_estimates.csv",     θ_boots)
        CSV.write("output/bootstrap_$(temp)_allN_init.csv",     θ_boots_all)
        CSV.write("output/bootstrap_$(temp)_samples.csv",       boot_samples)
        CSV.write("output/bootstrap_$(temp)_weight_matrix.csv", weight_matrix)
    end
end

# # counterfactual showup hat used in Tables 9 and 10, and
# # Online Appendix Table C.18.
# run counterfactuals_1.m
#     # Input:  MATLAB_Input.csv,
#     #         MATLAB_Input_small_sample.csv
#     # Output: MATLAB_showuphat_small_sample.csv
#     #         MATLAB_showuphat.csv
# function counterfactuals_1()
#     ##################
#     # fixed parameters
#     ##################
#
#     η_sd = 0.275;
#     irate = 1.22;
#     danual = 1/irate;
#     irm    = 1/irate;
#     δ = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5);
#
#     ##################
#     # Input parameters manually
#     # these are the baseline estimated parameters (danual = 1/irate)
#     ##################
#
#     μ_ϵ        = -79681;
#     σ_ϵ       = 59715;
#     α         = 0.5049;
#     λ_con_belief    = 8.0448;
#     λ_β_belief   = -0.71673;
#
#     # full sample
#     sample = [];
#
#     for small_sample_dummy = 0:1
#
#         # load full data
#         load_data(sample,small_sample_dummy);
#         global hhid quant_idx;
#
#         # Column 1 is showup, Column 2 is showuphat
#         [ showup_hat, ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#         showup_table   = [table(hhid), table(quant_idx), table(showup_hat)];
#
#         # Column 3 Half Standard Deviation of epsilon
#         [ showup_hat_halfsd , ~ ] = showuphat(μ_ϵ, σ_ϵ/2, α, λ_con_belief, λ_β_belief, η_sd, δ);
#         showup_table   = [showup_table, table(showup_hat_halfsd)];
#
#         # Column 4 No epsilon variance
#         [ showup_hat_noeps , ~ ] = showuphat(μ_ϵ, σ_ϵ/1e10, α, λ_con_belief, λ_β_belief, η_sd, δ);
#         showup_table   = [showup_table, table(showup_hat_noeps)];
#
#         ### replace travel cost by same travel cost
#         global totcost_smth_pc totcost_pc
#         totcost_pc = totcost_smth_pc; ##ok<NASGU>
#
#         # Column 5 no differential travel cost
#         [ showup_hat_smthc , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#         showup_table   = [showup_table, table(showup_hat_smthc)];
#
#         ### reload data, and assume constanta MU and λ
#         load_data(sample,small_sample_dummy);
#         mean_mu = 0.0967742; # mean benefit receipt conditional on applying
#
#         global μ_con_true μ_β_true FE FE2;
#         λ_con_belief_cml = norminv(mean_mu);
#         λ_β_belief_cml = 0;
#         μ_con_true = norminv(mean_mu);
#         μ_β_true = 0;
#         FE = FE*0;
#         FE2 = FE2*0;
#
#         # Column 6 (constant mu AND λ, update from the previous draft)
#         [ showup_hat_cml  , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief_cml, λ_β_belief_cml, η_sd, δ);
#         showup_table   = [showup_table, table(showup_hat_cml)];
#
#
#         if(small_sample_dummy==1)
#             global close
#             showup_table   = [showup_table, table(close)];
#             writetable(showup_table, [tempfolder 'MATLAB_showuphat_small_sample.csv']);
#
#         else
#
#             # Column 7: 3 extra km
#             load_data(sample,small_sample_dummy);
#             global totcost_pc totcost_3k_pc close
#             totcost_pc = (1-close).*totcost_3k_pc + close.*totcost_pc;
#             [ showup_hat_km3 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_km3)];
#
#             # Column 8: 6 extra km
#             load_data(sample,small_sample_dummy);
#             global totcost_pc totcost_6k_pc close
#             totcost_pc = (1-close).*totcost_6k_pc + close.*totcost_pc;
#             [ showup_hat_km6 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_km6)];
#
#             # Column 9: 3x waiting time
#             load_data(sample,small_sample_dummy);
#             global totcost_pc hhsize close ave_waiting wagerate
#             totcost_pc = totcost_pc + (1-close).*(2.*ave_waiting.*wagerate)./(hhsize.*60);
#             [ showup_hat_3aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_3aw)];
#
#             # Column 10: 6x waiting time
#             load_data(sample,small_sample_dummy);
#             global totcost_pc hhsize close ave_waiting wagerate
#             totcost_pc = totcost_pc + (1-close).*(5.*ave_waiting.*wagerate)./(hhsize.*60);
#             [ showup_hat_6aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_6aw)];
#
#
#             # Column 11-12 α=0 (all-unsophisticated) and α=1 (all sophisticated)
#             load_data(sample,small_sample_dummy);
#             [ showup_hat_α0 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 0, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_α0)];
#
#             [ showup_hat_α1 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 1, λ_con_belief, λ_β_belief, η_sd, δ);
#             showup_table   = [showup_table, table(showup_hat_α1)];
#
#             writetable(showup_table, [tempfolder 'MATLAB_showuphat.csv']);
#         end
#
#     end
# end
