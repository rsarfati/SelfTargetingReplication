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
function compute_quantiles(df::DataFrame;
                           qN = Dict([:c,:pmt,:unob_c,:distt] .=> [5,3,3,4]))
    # Helper functions
    quant(N::Int64)     = [n/N for n=1:N-1]
    assign_q(x, quants) = [minimum(vcat(findall(>=(x[i]), quants),
                                   length(quants)+1)) for i=1:length(x)]
    # Assign categoricals for quantiles
    for v in keys(qN)
        v_n = Symbol(string(v) * "_q")
        insertcols!(df, v_n=>assign_q(df[:,v], quantile(df[:,v], quant(qN[v]))))
    end
    return df
end

"""
```
compute_moments(df0::DataFrame, showup_hat::Vector{T}, δ_mom::T,
                true_λ::T, bel_λ::T, ind_λ::T) where T<:Union{F64,Int64}
```
Moments:
1-10.  Mean showup rates in (measured) consumption quintiles in far and close
       subtreatment (separately). -> Ten moments
11-14. Mean(showup - showuphat) of four extreme cells in grid of [terciles of
       pmt * terciles of w] (residual from the regression of log(consumption)
       on pmt) -> Four moments
15-16. Two extreme cells in quartiles of distance -> Two moments
17-20. λ moments -> Four moments
"""
function compute_moments(df0::DataFrame, showup_hat::Vector{T}, δ_mom::T,
                         true_λ::T, bel_λ::T, ind_λ::T) where T<:Union{F64,Int64}
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
        # Following MATLAB code. Note that δ_mom = 0 for the main estimation
        moments[:,5+i] = (Δ_showup .* close_i / sum(close_i)) -
                         δ_mom * moments[:,i]
    end

    # 11-14: {Top, bottom tercile} x {observable, unobservable} consumption
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

    # If showuphat exactly zero, the λ moments are NaN; replace w/ large number
    moments[isnan.(moments)] .= 10000.
    return moments .* N
end

"""
```
showuphat(df::DataFrame, t::Vector{T}, η_sd::T, δ::T, μ_con_true::T,
          μ_β_true::T, λ_con_true::T, λ_β_true::T; N_grid=100) where T<:F64
```
Evaluate probability of showing up for each i. (Eq 22)
"""
function showuphat(df::DataFrame, t::Vector{T}, η_sd::T, δ::T, μ_con_true::T,
                   μ_β_true::T, λ_con_true::T, λ_β_true::T;
                   N_grid = 100) where T<:F64

    # Unpack parameters
    μ_ϵ, σ_ϵ, α, λ_con_bel, λ_β_bel = t
    N = length(df.c)

    # Convert mean and standard deviation into α and β
    s = sqrt(3 * (σ_ϵ ^ 2) / (pi ^ 2))
    A = μ_ϵ / s
    β = 1   / s

    # Lower and upper bound, grid density for the numerical integration over η
    lb = -η_sd * 4.
    ub = -lb

    function util(η::F64)
        # Present Utility
        relu_2day = (df.c .* exp.(-η) - df.totcost_pc .* exp.(-η) +
                    (1 .- 1 .* exp.(-η)) .* df.moneycost) - (df.c .* exp.(-η))
        # Future Utility
        relu_2mor = (df.c .* exp.(-η) .+ df.benefit) - (df.c .* exp.(-η))

        Mu = cdf.(Normal(), μ_con_true .+ df.FE2 .+ μ_β_true * (df.pmt .- η))
        Λ  = cdf.(Normal(), λ_con_bel  .+ λ_β_bel * (df.logc .- η))

        prob_s = (1 .- inv.(1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Mu .* relu_2mor) .+ A)))
        prob_u = (1 .- inv.(1 .+ exp.(β .* (relu_2day .+ 12 .* δ .* Λ  .* relu_2mor) .+ A)))
        return (α .* prob_s .+ (1 - α) .* prob_u) .* pdf(Normal(0, η_sd), η)
    end

    # Trapezoidal integration w/ uniform grid
    showup_hat = trapz(util, lb, ub, N_grid)

    # Rather than running probit, apply WLS...
    # Calculate inverse of μ, Φ^{-1}(μ), where Φ is standard normal CDF
    conv  = sqrt.(showup_hat)
    μ_inv = conv .* (μ_con_true .+ df.FE2 .+ μ_β_true .* df.pmt)
    X     = hcat(conv, conv .* df.logc)
    σ2    = sqrt(sum(((eye(N) - X / (X' * X) * X') * μ_inv) .^ 2) / (N - 2))
    # Divide by σ2 to impose sd = 1 for error
    coef  = (1. / σ2) * (X' * X) \ X' * μ_inv
    # Compute induced λ
    ind_λ = cdf.(Normal(), hcat(ones(N), df.logc) * coef)

    return showup_hat, ind_λ
end

"""
```
GMM_problem(df0::DataFrame, danual::T; δ_mom::T = 0.0, irate::T = 1.22,
            η_sd::T = 0.275, f_tol::T = 1e-2, VERBOSE = true) where T<:F64
```
Two Stage Feasible GMM

Parameters:
# 1-2. mean, variance of ϵ (utility shock)
# 3.   σ (coef of relative risk aversion = rho)
# 4.   α (fraction of people who are sophisticated)
# 5-6. constant and coefficient for λ function: P(get benefit | show up)

How are moments computed?
  For each obs, calculate showup_hat as the probability that ϵ > -gain (i.e.
  that gain + ϵ > 0) This is just 1 - {the cdf of F_ϵ evaluated at (-gain_i)}.
"""
function GMM_problem(df0::DataFrame, danual::T; δ_mom::T=0.0, irate::T=1.22,
                     η_sd::T=0.275, f_tol::T=1e-2, VERBOSE=true) where T<:F64

    # Fetch relevant constants before dropping columns
    μ_con_true = df0.reg_const2[1]
    μ_β_true   = df0.reg_pmt2[1]
    λ_con_true = df0.reg_nofe_const[1]
    λ_β_true   = df0.reg_nofe_logcon[1]

    # Drop unnecessary columns, compute relevant quantiles + augment!
    df = compute_quantiles(df0[:,[:totcost_pc, :moneycost, :c, :logc, :distt,
                                  :pmt, :FE2, :getbenefit, :benefit, :showup,
                                  :close, :unob_c]])

    N   = size(df, 1) # No. of households
    N_m = 20          # No. of moment conditions
    N_p = 5           # No. parameters to estmate
    irm = 1 / irate   # Compute NPV δ (annual)
    δ   = danual * (1 + irm + irm^2 + irm^3 + irm^4 + irm^5)

    # Define objective function!
    function g(t::Vector{T}, df1::D) where {T<:Float64, D<:DataFrame}
        showup_hat, induced_λ = showuphat(df1, t, η_sd, δ, μ_con_true, μ_β_true,
                                          λ_con_true, λ_β_true)
        true_λ = cdf.(Normal(), λ_con_true   .+ λ_β_true   .* df1.logc)
        bel_λ  = cdf.(Normal(), t[4] .+ t[5] .* df1.logc)
        return compute_moments(df1, showup_hat, δ_mom, true_λ, bel_λ, induced_λ)
    end

    # This function exists purely so we aren't calling showuphat/compute_moments
    # twice when we don't need to! Also, acts as a function closure on df.
    function gAg(x::Vector{F64}, A::Matrix{F64})
        g_eval = mean(g(x, df), dims=1)
        return (g_eval * A * g_eval')[1]
    end

    # Julia's Optim.jl package explores different starting values by default!
    # Thus, I will simply begin at mean of MATLAB code initial guesses.
    t0  = [-79700,  59700,   0.5, 8.04, -0.72]
    lb1 = [-200000,     0,  0.001,   0, -2]
    ub1 = [ 200000, 200000, 0.999,  20,  1]

    ### GMM: First stage (begin with identity matrix)
    println("Running First Stage... (approx. 1 min)")
    t1 = minimizer(optimize(x -> gAg(x, eye(N_m)), lb1, ub1, t0, NelderMead(),
                            Optim.Options(f_tol=f_tol, show_trace = VERBOSE)))

    ### GMM: Second stage (compute optimal weighting matrix)
    println("Running Second Stage... (approx. 1 min)")
    g1 = g(t1, df)
    Om = inv(g1' * g1 / N)
    return minimizer(optimize(x -> gAg(x, Om), lb1, ub1, t1, NelderMead(),
                              Optim.Options(f_tol=f_tol, show_trace = VERBOSE)))
end

"""
Table 8. Estimated Parameter Values for the Model // estimation_1.m
Input:  MATLAB_Input.csv
Output: MATLAB_est_d**.csv
** corresponds to the value of the annual discount factor: (* 100)
   d=0.82 = 1/irate, d=0.50 or d=0.95
"""
function estimation_1(; irate = 1.22, η_sd = 0.275, δ_mom = 0.0, N_bs = 100,
                      run_estimation = false, run_bootstrap = false,
                      output_table = true, f_tol = 1e-2, VERBOSE = false)

    # Load + organize data
    df = CSV.read("input/MATLAB_Input.csv", DataFrame, header = true)
    df = insertcols!(df, :logc => log.(df.consumption))
    df = rename!(df, [:closesubtreatment => :close, :consumption => :c,
                      :pmtscore => :pmt])
    df = clean(df, [:logc, :close, :getbenefit, :pmt, :distt, :c], F64)

    # Compute unobs. cons (residual regressing log(cons) on PMT)
    df = insertcols!(df, :unob_c => residuals(reg(df, @formula(logc~pmt)), df))
    N    = size(df, 1) # No. households
    N_p  = 5           # No. parameters to estimate
    pre  = "output/MATLAB"

    # Run 3 estimations with varying discount factors
    if run_estimation
        for δ_y in [1/irate, 0.5, 0.95]
            println("\nEstimating discount factor: ", δ_y)
            println("------------------------------------------------")
            min_t = GMM_problem(df, δ_y; δ_mom = δ_mom, irate = irate,
                                η_sd = η_sd, VERBOSE = VERBOSE, f_tol = f_tol)
            println("Estimated parameters are: ", min_t)
            CSV.write("$(pre)_est_d$(Int(round(δ_y * 100))).csv",
                      Tables.table(min_t))
        end
    end

    # Bootstrap SEs for δ_y = 0.82 (Table 8)
    if run_bootstrap
        δ_y  = 1 / irate   # Alt: 0.50, 0.95.
        θ_bs = zeros(N_bs, N_p)
        it   = 1
        while it <= N_bs
            println("\nBootstrap iteration: $it \n------------------------")
            # Randomly draw households (with replacement)
            idx_bs = sample(1:N, N; replace = true)
            df_bs  = df[idx_bs,:]
            try
                θ_bs[it,:] = GMM_problem(df_bs, δ_y; δ_mom=δ_mom, irate=irate,
                                         η_sd=η_sd, VERBOSE=false, f_tol=f_tol)
                CSV.write("$(pre)_bs_$(Int(round(δ_y*100))).csv", Tables.table(θ_bs))
                it += 1
            catch e
                @show e
                if typeof(e) <: DomainError
                    println("Domain err. in estimation, restarting iteration!")
                else
                    throw(e)
                end
            end
        end
    end

    # Generate LaTeX Table 8
    if output_table
        δ_y   = 1 / irate
        t_est = CSV.read("$(pre)_est_d$(Int(round(δ_y*100))).csv",
                         DataFrame, header=true)
        θ_bs  = CSV.read("$(pre)_bs_$(  Int(round(δ_y*100))).csv",
                         DataFrame, header=true)
        bs_SE = [std(θ_bs[:,i]) for i=1:N_p]

        # Directly write LaTeX table
        io = open("output/tables/Table8.tex", "w")
        write(io, "\\begin{tabular}{ccccc}\\toprule" *
                  "\$\\nu_{\\epsilon}\$ & \$\\sigma_{\\epsilon}\$ &" *
                  " \$\\alpha\$ & \$\\gamma\$ & \$\\pi\$ \\\\ \\midrule")
        @printf(io, " %d &  %d & %0.2f & % 0.2f & %0.2f \\\\", t_est[:,1]...)
        @printf(io, "(%d) & (%d) & (%0.2f) & (%0.2f) & (%0.2f)\\\\", bs_SE...)
        write(io, "\\bottomrule\\end{tabular}")
        close(io)
    end
end
