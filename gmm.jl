"""
```
compute_quantiles(df::DataFrame; N_q = Dict([:c, :pmt,
           :unob_c, :distt] .=> [5, 3, 3, 4]]))
```
Computes quantiles of variables in inputted dictionary, and puts the associated
categorical labels into dataframe for series. Default inputs are obs. consumption,
PMT, unobs. consumption (w), distance

Corresponds roughly to `load_data.m` file in replication code.
"""
function compute_quantiles(df::DataFrame; N_q = Dict([:c, :pmt,
                                       :unob_c, :distt] .=> [5, 3, 3, 4]))
    # Helper functions
    quant(N::Int64)     = [n/N for n=1:N-1]
    assign_q(x, quants) = [minimum(vcat(findall(>=(x[i]), quants),
                                        length(quants)+1)) for i=1:length(x)]
    # Assign categoricals for quantiles
    for v in keys(N_q)
        v_n = Symbol(string(v) * "_q")
        insertcols!(df, v_n => assign_q(df[:,v], quantile(df[:,v], quant(N_q[v]))))
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
       on pmt) -> Four moments
15-16. two extreme cells in quartiles of distance -> Two moments
17-20. λ moments -> Four moments
"""
function compute_moments(df0::D, showup_hat::Union{T,Vector{T}}, true_λ, bel_λ,
                         ind_λ) where {T<:Union{Float64,Int64}, D<:DataFrame}

    # Setup
    N        = length(showup_hat)
    moments  = Matrix{F64}(undef, N, 20)
    Δ_logc   = df0.logc .- mean(df0.logc)
    Δ_showup = df0.showup - showup_hat

    # 1-10: Far/close subtreatment x i=1:5th quintile of consumption
    for i=1:5
        far_i   = (df0.c_q .== i) .&   iszero.(df0.close)
        close_i = (df0.c_q .== i) .& .!iszero.(df0.close)
        moments[:,i]   = Δ_showup .* far_i   / sum(far_i)
        moments[:,5+i] = Δ_showup .* close_i / sum(close_i)
    end

    # 11-14: {Top, bottom tercile} x {observable, unobservable consumption}
    for (i, Q) in enumerate([[3,1], [3,3], [1,1], [1,3]])
        idx = (df0.pmt_q .== Q[1]) .& (df0.unob_c_q .== Q[2])
        moments[:,10 + i] = Δ_showup .* idx / sum(idx)
    end

    # 15-16: Top and bottom distance quartiles
    T_D = (df0.distt_q .== 4)
    B_D = (df0.distt_q .== 1)
    moments[:,15] =  Δ_showup .* T_D / sum(T_D)
    moments[:,16] =  Δ_showup .* B_D / sum(B_D)

    # 17-20: Mean λ function moments
    N_show        = sum(df0.showup)
    moments[:,17] = (bel_λ .- df0.getbenefit)           .* df0.showup / N_show
    moments[:,18] = (bel_λ .- df0.getbenefit) .* Δ_logc .* df0.showup / N_show
    moments[:,19] = (ind_λ .- df0.getbenefit)           .* df0.showup / N_show
    moments[:,20] = (ind_λ .- df0.getbenefit) .* Δ_logc .* df0.showup / N_show

    # If showup hat exactly zero, the λ moments are NaN; replace w/ large number
    moments[isnan.(moments)] .= 10000.
    return moments .* N
end

"""
Evaluate probability of showing up for each i. (Eq 22)
"""
function showuphat(df::Union{DataFrame,DataFrameRow}, t::Vector{T}, η_sd::T,
                   δ::T, μ_con_true::T, μ_β_true::T, λ_con_true::T, λ_β_true::T;
                   N_grid = 100) where {T<:F64}

    # Unpack parameters
    μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel = t
    N = length(df.c)

    # Convert mean and standard deviation into α and β
    s  = sqrt(3 * (σ_ϵ ^ 2) ./ (pi ^ 2))
    αa = μ_ϵ ./ s
    β  = 1 ./ s

    # Lower and upper bound, grid density for the numerical integration over eta
    lb = -η_sd * 4
    ub = -lb

    function util(η::F64)
        # Present Utility
        relu_2day = (df.c .* exp.(-η) - df.totcost_pc .* exp.(-η) +
                    (1 .- 1 .* exp.(-η)) .* df.moneycost) - (df.c .* exp.(-η))
        # Future Utility
        relu_2mor = (df.c .* exp.(-η) .+ df.benefit) - (df.c .* exp.(-η))

        Mu = cdf.(Normal(), μ_con_true .+ df.FE2 .+ μ_β_true * (df.pmt .- η))
        Λ  = cdf.(Normal(), λ_con_bel  .+ λ_β_bel * (log.(df.c) .- η))

        prob_s = (1 .- 1 ./ (1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Mu .* relu_2mor) .+ αa)))
        prob_u = (1 .- 1 ./ (1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Λ  .* relu_2mor) .+ αa)))

        return (α * prob_s + (1 - α) * prob_u) .* pdf.(Normal(0, η_sd), η)
    end

    # Trapezoidal rule w/ uniform grid
    showup_hat = -(util(lb) + util(ub))
    for η_i in range(lb, stop=ub, length=N_grid)
        showup_hat += 2*util(η_i)
    end
    showup_hat *= 0.5 * (ub-lb) / 100

    # Rather than running probit, apply WLS.
    # Calculate inverse of mu Phi(-1)(mu), where Phi is standard normal CDF
    muinv     = μ_con_true .+ df.FE2 .+ μ_β_true .* df.pmt
    const_t   = sqrt.(showup_hat)   # Conversion for the weight
    muinv_t   = const_t .* muinv    # Conversion for the frequency weight
    logcons_t = const_t .* df.logc # Conversion for the weight
    X         = hcat(const_t, logcons_t)
    sigma2    = sqrt(sum(((Matrix{F64}(I, N, N) - X / (X' * X) * X') * muinv_t) .^ 2) / (N - 2))
    coef      = 1 / sigma2 * (X' * X) \ X' * muinv_t # Divide by sigma to impose sd=1 for error
    ind_λ     = cdf.(Normal(), hcat(ones(N), df.logc) * coef)

    return showup_hat, ind_λ
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
function GMM_problem(df0::DataFrame, danual::T; δ_mom::T = 0.0, irate::T = 1.22,
                     η_sd::T = 275.0, VERBOSE=true) where T<:F64

    μ_con_true = df0.reg_const2[1]
    μ_β_true   = df0.reg_pmt2[1]
    λ_con_true = df0.reg_nofe_const[1]
    λ_β_true   = df0.reg_nofe_logcon[1]

    df = df0[:, [:totcost_pc, :moneycost, :c, :logc, :distt, :pmt,
                :FE2, :getbenefit, :benefit, :showup, :close, :unob_c]]
    df = compute_quantiles(df)

    N   = size(df, 1) # No. of households
    N_m = 20          # No. of moment conditions
    N_p = 5           # No. parameters to estmate

    # Compute NPV δ (yearly)
    irm = 1 / irate
    δ   = danual * (1 + irm + irm^2 + irm^3 + irm^4 + irm^5)

    """
    Define objective functions!
    """
    function g(t::Vector{T}, df1::D) where {T<:Float64, D<:DataFrame}
        showup_hat, induced_λ = showuphat(df1, t, η_sd, δ, μ_con_true, μ_β_true,
                                          λ_con_true, λ_β_true)
        true_λ    = cdf.(Normal(), λ_con_true   .+ λ_β_true   .* df1.logc)
        bel_λ  = cdf.(Normal(), t[4] .+ t[5] .* df1.logc)
        return compute_moments(df1, showup_hat, true_λ, bel_λ, induced_λ)
    end
    function gAg(x::Vector{F64}, A::Matrix{F64})
        g_eval = mean(g(x, df), dims=1)
        return (g_eval * A * g_eval')[1]
    end

    # Julia's Optim package explores different starting values by default!
    # Thus will just start at mean of MATLAB code initial guesses
    t0  = [-79700,  59700,   0.5, 8.04, -0.72]
    lb1 = [-200000,     0,  0.001,   0, -2]
    ub1 = [ 200000, 200000, 0.999,  20,  1]

    ### GMM: First Step (Initial values nabbed from Matlab)
    println("Running First Stage... (approx. 1 min)")
    W0 = Matrix{Float64}(I, N_m, N_m)
    t1 = minimizer(optimize(x -> gAg(x, W0), lb1, ub1, t0, NelderMead(),
                            Optim.Options(f_tol=1e-2, show_trace = VERBOSE)))
    # GMM: Second stage
    println("Running Second Stage... (approx. 1 min)")
    g1 = g(t1, df)
    Om = inv(g1' * g1 / N)
    return minimizer(optimize(x -> gAg(x, Om), lb1, ub1, t1, NelderMead(),
                              Optim.Options(g_tol=1e5, show_trace = VERBOSE)))
end

"""
Table 8. Estimated Parameter Values for the Model // estimation_1.m
Input:  MATLAB_Input.csv
Output: MATLAB_est_d**.csv
** corresponds to the value of the annual discount factor:
   d=0.82 = 1/irate, d=0.50 or d=0.95
"""
function estimation_1(; irate = 1.22, η_sd = 0.275, δ_mom = 0.0, VERBOSE = false,
                      run_estimation = false, run_bootstrap = false,
                      output_tables = true, N_bs = 100)
    # Load + organize data
    df = CSV.read("input/MATLAB_Input.csv", DataFrame, header = true)
    df = insertcols!(df, :logc => log.(df.consumption))
    df = rename!(df, [:closesubtreatment => :close, :consumption => :c,
                      :pmtscore => :pmt])
    df = clean(df, [:logc, :close, :getbenefit, :pmt, :distt, :c], F64)

    # Compute unobs. consumption (residual regressing log(obs consumption) on PMT)
    df = insertcols!(df, :unob_c => residuals(reg(df, @formula(logc ~ pmt)), df))

    # Run three estimations with varying discount factors
    if run_estimation
        for δ_y in [1/irate, 0.5, 0.95]
            min_t = GMM_problem(df, δ_y; δ_mom = δ_mom, irate = irate,
                                η_sd = η_sd, VERBOSE = VERBOSE)
            println("Estimated parameters are: ", min_t)
            CSV.write("output/MATLAB_est_d$(round(δ_y * 100)).csv",
                      Tables.table(min_t))
        end
    end
    # Bootstrap SEs
    if run_bootstrap
        N    = size(df, 1) # No. of households
        N_p  = 5           # No. of parameters to estimate
        δ_y  = 1 / irate   # Alt: danual = 0.50 and = 0.95.
        θ_bs = zeros(N_bs, N_p)
        it   = 1
        while it <= N_bs
            println("\nBootstrap iteration: $it")
            println("------------------------")
            # Randomly draw households (with replacement)
            idx_bs     = sample(1:N, N; replace = true)
            df_bs      = df[idx_bs,:]
            try
                θ_bs[it,:] = GMM_problem(df_bs, δ_y; δ_mom = δ_mom, irate = irate,
                                         η_sd = η_sd, VERBOSE = false)
                CSV.write("output/MATLAB_bs_$(Int(round(δ_y * 100)))_secondhalf.csv", Tables.table(θ_bs))
                it += 1
            catch e
                @show e
                if typeof(e) <: DomainError
                    println("Domain error in estimation, running new bootstrap iteration!")
                else
                    throw(e)
                end
            end
        end
    end
    if output_tables
        δ_y   = 1 / irate
        t_est = CSV.read("output/MATLAB_est_d$(round(δ_y * 100)).csv")
        θ_bs  = CSV.read("output/MATLAB_bs_$(Int(round(δ_y * 100))).csv")
        bs_SE = [std(θ_bs[:,i]) for i=1:N_p]

        # Directly write LaTeX table
        io = open("output/tables/Table8.tex", "w")
        write(io, "\\begin{tabular}{ccccc}\\toprule" *
                  "\$\\nu_{\\epsilon}\$ & \$\\sigma_{\\epsilon}\$ &" *
                  " \$\\alpha\$ & \$\\gamma\$ & \$\\pi\$ \\\\ \\midrule")
        @printf(io, " %5i &  %5i & %0.2f & % 0.2fi & %0.2f \\\\", t_est...)
        @printf(io, "(%4i) & (%5i) & (%0.2f) & (%0.2f) & (%0.2f)\\\\", bs_SE...)
        write(io, "\\bottomrule\\end{tabular}")
        close(io)
    end
end

"""
# counterfactual showup hat used in Tables 9 and 10, and
# Online Appendix Table C.18.
# run counterfactuals_1.m
    # Input:  MATLAB_Input.csv,
    #         MATLAB_Input_small_sample.csv
    # Output: MATLAB_showuphat_small_sample.csv
    #         MATLAB_showuphat.csv
"""
# function counterfactuals_1()
#     ##################
#     # fixed parameters
#     ##################
#
#     η_sd = 0.275
#     irate = 1.22
#     danual = 1/irate
#     irm    = 1/irate
#     δ = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5)
#
#     ##################
#     # Input parameters manually
#     # these are the baseline estimated parameters (danual = 1/irate)
#     ##################
#
#     μ_ϵ       = -79681
#     σ_ϵ       = 59715
#     α         = 0.5049
#     λ_con_bel = 8.0448
#     λ_β_bel   = -0.71673
#
#     # full sample
#     sample = []
#
#     for small_sample_dummy = 0:1
#
#         # load full data
#         load_data(sample,small_sample_dummy)
#         global hhid quant_idx
#
#         # Column 1 is showup, Column 2 is showuphat
#         [ showup_hat, ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#         showup_table   = [table(hhid), table(quant_idx), table(showup_hat)]
#
#         # Column 3 Half Standard Deviation of epsilon
#         [ showup_hat_halfsd , ~ ] = showuphat(μ_ϵ, σ_ϵ/2, α, λ_con_bel, λ_β_bel, η_sd, δ)
#         showup_table   = [showup_table, table(showup_hat_halfsd)]
#
#         # Column 4 No epsilon variance
#         [ showup_hat_noeps , ~ ] = showuphat(μ_ϵ, σ_ϵ/1e10, α, λ_con_bel, λ_β_bel, η_sd, δ)
#         showup_table   = [showup_table, table(showup_hat_noeps)]
#
#         ### replace travel cost by same travel cost
#         global totcost_smth_pc totcost_pc
#         totcost_pc = totcost_smth_pc ##ok<NASGU>
#
#         # Column 5 no differential travel cost
#         [ showup_hat_smthc , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#         showup_table   = [showup_table, table(showup_hat_smthc)]
#
#         ### reload data, and assume constanta MU and λ
#         load_data(sample,small_sample_dummy)
#         mean_mu = 0.0967742 # mean benefit receipt conditional on applying
#
#         global μ_con_true μ_β_true FE FE2
#         λ_con_bel_cml = norminv(mean_mu)
#         λ_β_bel_cml = 0
#         μ_con_true = norminv(mean_mu)
#         μ_β_true = 0
#         FE = FE*0
#         FE2 = FE2*0
#
#         # Column 6 (constant mu AND λ, update from the previous draft)
#         [ showup_hat_cml  , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel_cml, λ_β_bel_cml, η_sd, δ)
#         showup_table   = [showup_table, table(showup_hat_cml)]
#
#
#         if(small_sample_dummy==1)
#             global close
#             showup_table   = [showup_table, table(close)]
#             writetable(showup_table, [tempfolder 'MATLAB_showuphat_small_sample.csv'])
#
#         else
#
#             # Column 7: 3 extra km
#             load_data(sample,small_sample_dummy)
#             global totcost_pc totcost_3k_pc close
#             totcost_pc = (1-close).*totcost_3k_pc + close.*totcost_pc
#             [ showup_hat_km3 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_km3)]
#
#             # Column 8: 6 extra km
#             load_data(sample,small_sample_dummy)
#             global totcost_pc totcost_6k_pc close
#             totcost_pc = (1-close).*totcost_6k_pc + close.*totcost_pc
#             [ showup_hat_km6 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_km6)]
#
#             # Column 9: 3x waiting time
#             load_data(sample,small_sample_dummy)
#             global totcost_pc hhsize close ave_waiting wagerate
#             totcost_pc = totcost_pc + (1-close).*(2.*ave_waiting.*wagerate)./(hhsize.*60)
#             [ showup_hat_3aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_3aw)]
#
#             # Column 10: 6x waiting time
#             load_data(sample,small_sample_dummy)
#             global totcost_pc hhsize close ave_waiting wagerate
#             totcost_pc = totcost_pc + (1-close).*(5.*ave_waiting.*wagerate)./(hhsize.*60)
#             [ showup_hat_6aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_6aw)]
#
#
#             # Column 11-12 α=0 (all-unsophisticated) and α=1 (all sophisticated)
#             load_data(sample,small_sample_dummy)
#             [ showup_hat_α0 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 0, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_α0)]
#
#             [ showup_hat_α1 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 1, λ_con_bel, λ_β_bel, η_sd, δ)
#             showup_table   = [showup_table, table(showup_hat_α1)]
#
#             writetable(showup_table, [tempfolder 'MATLAB_showuphat.csv'])
#         end
#
#     end
# end
