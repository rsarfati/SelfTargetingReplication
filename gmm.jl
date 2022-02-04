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
function compute_quantiles(df::DataFrame; N_q = Dict([:consumption, :PMTSCORE, :unobs_cons,
                                                      :distt] .=> [5, 3, 3, 4]))
    # Helper functions
    quant(N::Int64)     = [n/N for n=1:N-1]
    assign_q(x, quants) = [minimum(vcat(findall(>=(x[i]), quants),
                                        length(quants)+1)) for i=1:length(x)]

    # Compute unobs. consumption (residual regressing log(obs consumption) on PMT)
    !("unobs_cons" in names(df)) && insertcols!(df, :unobs_cons =>
                                    residuals(reg(df, @formula(log_c ~ PMTSCORE)), df))

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
function compute_moments(df0::D, showup::Vector{T} = Vector(df0.showup),
                         showup_hat::Vector{T}, true_λ, belief_λ,
                         induced_λ) where {T<:Union{Float64,Int64}, D<:DataFrame}

    # Initialize dictionary, manage NaNs in data
    moments = Dict{Int,Float64}()
    showup  = (showup .== 1) .& .!ismissing.(showup)

    # 1-10 Far/close subtreatment x i=1:5th quintile of consumption
    for i=1:5
        far_i   = (df0.consumption_q .== i) .&   iszero.(df0.close)
        close_i = (df0.consumption_q .== i) .& .!iszero.(df0.close)
        moments[i]   = (sum(showup[far_i])   - sum(showup_hat[far_i]))   / sum(far_i)
        moments[5+i] = (sum(showup[close_i]) - sum(showup_hat[close_i])) / sum(close_i)
    end

    # 11-14 {Top, bottom tercile} x {observable, unobservable consumption}
    for i, Q in zip([[3,1], [3,3], [1,1], [1,3]])
        idx = (df0.PMTSCORE_q .== Q[1]) .& (df0.unobs_cons_q .== Q[3])
        moments[10 + i] = (sum(showup[idx]) - sum(showup_hat[idx])) / sum(idx)
    end

    # 15-16 Top and bottom distance quartiles
    T_D = (df0.distt_q .== 4)
    B_D = (df0.distt_q .== 1)

    moments[15] =  (sum(showup[T_D]) - sum(showup_hat[T_D])) / sum(T_D)
    moments[16] =  (sum(showup[B_D]) - sum(showup_hat[B_D])) / sum(B_D)

    # 17-20 Mean λ function moments
    N_show   = sum(showup)
    moms[17] = sum((belief_λ - df0.getbenefit) .* showup) / N_show
    moms[18] = sum((belief_λ - df0.getbenefit) .*
                        (df0.log_c - mean(df0.log_c)) .* showup) / N_show
    moms[19] = sum((induced_λ - df0.getbenefit) .* showup) / N_show
    moms[20] = sum((induced_λ - df0.getbenefit) .*
                        (df0.log_c - mean(df0.log_c)) .* showup) / N_show

    # If showup hat exactly zero, the λ moments are NaN; replace w/ large number
    moments[isnan.(moments)] .= 10000.
    return moments
end

"""
Evaluate probability of showing up for each i. (Eq 22)
"""
function showuphat(df, t, η_sd, δ; N_grid = 100)

    μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief = t

    # Constants
    μ_con_true = df.reg_const2[1]
    μ_β_true   = df.reg_pmt2[1]
    λ_con_true = df.reg_nofe_const[1]
    λ_β_true   = df.reg_nofe_logcon[1]

    # convert mean and standard deviation into α and β
    s  = sqrt(3 * (σ_ϵ ^ 2) ./ (pi ^ 2))
    αa = μ_ϵ ./ s
    β  = 1 ./ s

    # Lower and upper bound, grid density for the numerical integration over eta
    lb  = -η_sd*4
    ub  = -lb
    N   = length(df.consumption)

    # Prepare mesh grid
    eta         = LinSpace(lb, ub, N_grid)
    ETA         = repeat(η, 1, N)
    Consumption = repeat(df.consumption', N_grid)
    Pmtscore    = repeat(df.pmtscore', N_grid)
    Totcost_pc  = repeat(df.totcost_pc', N_grid)
    Benefit     = repeat(df.benefit', N_grid)
    FFE2        = repeat(df.FE2', N_grid)
    FFE         = repeat(df.FE', N_grid)
    Moneycost   = repeat(df.moneycost', N_grid)

    # Calculate integrand
    # Present Utility (Cost function is adjusted because "totalcost" given is based on noisy consumption measure.)
    relu_2day = (Consumption .* exp.(-ETA) - Totcost_pc .* exp.(-ETA) + (1 .- 1 .* exp.(-ETA)) .* Moneycost) - (Consumption .* exp.(-ETA))

    # Future Utility: note that cdf(Normal(),) in the middle is mu function
    relu_2mor = (Consumption .* exp(-ETA) .+ Benefit) - (Consumption .* exp.(-ETA))
    Mu        = cdf(Normal(), μ_con_true .+ FFE2 .+ μ_β_true * (Pmtscore - ETA))
    Lambda    = cdf(Normal(), λ_con_belief+λ_β_belief * (log.(Consumption) - ETA))

    prob_s = (1 .- 1 ./ (1 .+ exp(β .* (relu_2day .+ 12 .* δ .* Mu     .* relu_2mor) .+ αa)))
    prob_u = (1 .- 1 ./ (1 .+ exp(β .* (relu_2day .+ 12 .* δ .* Lambda .* relu_2mor) .+ αa)))

    integ = (α * prob_s + (1 - α) * prob_u ) .* pdf.(Normal(0, η_sd), ETA)

    function util(η::F64)
        relu_2day = (Consumption .* exp.(-η) - Totcost_pc .* exp.(-η) + (1 .- 1 .* exp.(-η)) .* Moneycost) - (Consumption .* exp.(-η))

        # Future Utility: note that cdf(Normal(),) in the middle is mu function
        relu_2mor = (Consumption .* exp(-η) .+ Benefit) - (Consumption .* exp.(-η))
        Mu        = cdf(Normal(), μ_con_true .+ FFE2 .+ μ_β_true * (Pmtscore - η))
        Lambda    = cdf(Normal(), λ_con_belief+λ_β_belief * (log.(Consumption) - η))

        prob_s = (1 .- 1 ./ (1 .+ exp(β .* (relu_2day .+ 12 .* δ .* Mu     .* relu_2mor) .+ αa)))
        prob_u = (1 .- 1 ./ (1 .+ exp(β .* (relu_2day .+ 12 .* δ .* Lambda .* relu_2mor) .+ αa)))

        return (α * prob_s + (1 - α) * prob_u) .* pdf.(Normal(0, η_sd), η)
    end

    showup_hat, err = quadgk(util, lb, ub, rtol=1e-8)

    #showup_hat = real(trapz(eta,integ)')

    # Rather than running probit, apply WLS.
    # Calculate inverse of mu Phi(-1)(mu) where Phi is standard normal CDF
    muinv     = μ_con_true + FE2 + μ_β_true * df.pmtscore
    muinv_t   = sqrt.(showup_hat) .* muinv    # Conversion for the frequency weight
    const_t   = sqrt.(showup_hat)             # Conversion for the weight
    logcons_t = sqrt.(showup_hat) .* df.log_c # Conversion for the weight
    X = [const_t, logcons_t]

    sigma2 = sqrt.(sum.(((eye(N) - X / (X' * X) * X') * muinv_t) .^ 2) / (N - 2))
    coef      = 1 / sigma2 * (X' * X) \ X' * muinv_t # Divide by sigma to impose sd=1 for error
    induced_λ = cdf.(Normal(),[ones(N,1), df.log_c] * coef)

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
function GMM_problem(df, δ_mom, danual, irate, η_sd, N_init)
    # Load + clean data
    df = CSV.read("input/MATLAB_.csv", DataFrame, header = true)
    insertcols!(df, :log_c => log.(df.consumption))

    N  = size(df, 1) # No. of households
    N_m = 20 # No. of moment conditions

    df = compute_quantiles(df)

    λ_con_true = df.reg_nofe_const[1]
    λ_β_true   = df.reg_nofe_logcon[1]

    # compute NPV δ (yearly)
    irm = 1 / irate
    δ = danual * (1 + irm + irm^2 + irm^3 + irm^4 + irm^5)

    """
    Difference between empirical and estimated moments.
    """
    function g(t::Vector{T}, df1::D) where {T<:Float64, D<:Union{DataFrame, DataFrameRow}}
        showup_hat, induced_λ = showuphat(t, η_sd, δ)
        true_λ    = cdf(Normal(), λ_con_true   .+ λ_β_true   .* df1.log_c)
        belief_λ  = cdf(Normal(), λ_con_belief .+ λ_β_belief .* df1.log_c)
        return compute_moments(df0, showup, showup_hat, true_λ, belief_λ, induced_λ)
    end

    ### GMM: First Step

    # generate random sets of possible initial values for each parameter
    # "get benefit" function, λ for the naive people
    λ_con_rand = λ_con_true  * (0.8 + 0.4 * rand(N_init))
    λ_β_rand   = λ_β_true    * (0.8 + 0.4 * rand(N_init))

    # Currently imposing some function of random rho + noise
    σ_ϵ_rand =  26805 .* (0.8 .+ 0.4 .* rand(N_init))
    μ_ϵ_rand = -26215 .* (0.8 .+ 0.4 .* rand(N_init))
    α_rand   = 0.001 + 0.999*rand(N_init)
    Fval_fs       = zeros(N_init)
    Fitθ_fs       = zeros(N_init, 5)
    Showup_hat_fs = zeros(N, N_init)
    Induced_λ_fs  = zeros(N, N_init)
    exitflag_fs   = zeros(N_init)

    for p = 1:N_init
        # select initial parameters
        θ_in = [μ_ϵ_rand[p,1],
                σ_ϵ_rand[p,1],
                α_rand[p,1],
                λ_con_rand[p,1],
                λ_β_rand[p,1]]

        # begin with identity weigthing matrix
        W0 = eye(20)

        Fitθ_fs[p,:], Fval_fs[p], Showup_hat_fs[:,p],
        Induced_λ_fs[:,p], exitflag_fs[p] = run_GMM(θ_in, η_sd, δ, W0, δ_mom)
    end

    minindex = argmin(Fval_fs)
    θ_fs     = Fitθ_fs[minindex,:]
    t1       = minimizer(optimize(x -> g(x, df0)' * A * g(x, df0), t0))
    # GMM: Second step
    Om = mean([g(t1, df0[i,:]) * g(t1, df0[i,:])' for i=1:N_h])
    # Return final theta
    return minimizer(optimize(x -> g(x, df0)' * inv(Om) * g(x, df0), t1))





    ## First Stage: Find params with identity weighting matrix
    @show "First Stage Begin"



    ## Second Stage: Find Params with Optimal Weighting Matrix
    @show "Second Stage Begin"
    #θ_fs = [-67020	60897	0.22044	8.6504	-0.77535]
     μ_ϵ  = θ_fs[1]
     σ_ϵ = θ_fs[2]
     α   = θ_fs[3]
     λ_con_belief  = θ_fs[4]
     λ_β_belief = θ_fs[5]

     showup_hat, induced_λ = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ)

     belief_λ  = cdf(Normal(), λ_con_belief + λ_β_belief * df.log_c)
     true_λ    = cdf(Normal(), λ_con_true   + λ_β_true   * df.log_c)

    moms = moments( showup_hat, true_λ, belief_λ, induced_λ, δ_mom )

    N = size(consumption,1)
    Winv = (moms' * moms) / N
    W = inv(Winv)
    Whalf = chol(W)'
    norm((Whalf * Whalf') - W) #check decomposition worked.

    Fval_ss       = zeros(N_init,1)
    Fitθ_ss       = zeros(N_init,5)
    Showup_hat_ss = zeros(size(consumption,1),N_init)
    Induced_λ_ss  = zeros(size(consumption,1),N_init)
    exitflag_ss   = zeros(N_init,1)

    for p = 1:N_init
        tic
        # select initial parameters (the same ones as in first stage)
        θ_in = [μ_ϵ_rand[p,1],
                σ_ϵ_rand[p,1],
                α_rand[p,1],
                λ_con_rand[p,1],
                λ_β_rand[p,1]]

        # use optimal weigthing matrix Whalf
        Fitθ_ss[p,:], Fval_ss[p], Showup_hat_ss[:,p], Induced_λ_ss[:,p], exitflag_ss[p] =
            run_GMM(  θ_in, η_sd, δ, Whalf, δ_mom )
    end

    minindex = argmin(Fval_ss)
    ismax = (1:N_init) .== minindex

    all_θ = [μ_ϵ_rand, Fitθ_fs[:,1], Fitθ_ss[:,1],
              σ_ϵ_rand,Fitθ_fs[:,2], Fitθ_ss[:,2],
              α_rand,  Fitθ_fs[:,3], Fitθ_ss[:,3],
              λ_con_rand,  Fitθ_fs[:,4], Fitθ_ss[:,4],
              λ_β_rand, Fitθ_fs[:,5], Fitθ_ss[:,5],
              Fval_fs, Fval_ss, exitflag_fs, exitflag_ss, ismax']

    return W, all_θ
end



function GMM_obj(x, η_sd, δ, Whalf, δ_mom)
    #GMM_obj compute the product between the moment values (at given parameter
    #values) and Cholsky half of the weighting matrix
    global λ_con_true λ_β_true

    μ_ϵ  = x(1)
    σ_ϵ = x(2)
    α   = x(3)
    λ_con_belief  = x(4)
    λ_β_belief = x(5)

    showup_hat, induced_λ = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ)

    belief_λ  = cdf(Normal(), λ_con_belief+λ_β_belief*df.log_c)
    true_λ    = cdf(Normal(), λ_con_true  +λ_β_true  *df.log_c)

    moms = moments( showup_hat, true_λ, belief_λ, induced_λ, δ_mom )

    return mean(moms,1) * Whalf
end


function run_GMM(params_in, η_sd, δ, Whalf, δ_mom)

    # lower and upper bounds for all the parameters
    lb1 = [-200000,     0,  0.001,  0, -2]
    ub1 = [ 200000, 200000, 0.999, 20,  1]

    # run lsqnonlin
    fitparams,fval, _, exitflag =
        lsqnonlin(@(x)GMM_obj(x, η_sd, δ, Whalf, δ_mom), params_in, lb1, ub1, options2)

    # return showup_hat_return
    showuphat_return, induced_λ_return = showuphat(fitparams[1:5], η_sd, δ)
    return fitparams, fval, showuphat_return, induced_λ_return, exitflag

end



"""
Table 8. Estimated Parameter Values for the Model // estimation_1.m
Input:  MATLAB_Input.csv
Output: MATLAB_Estimates_main_d**.csv
** corresponds to the value of the annual discount factor:
   d=0.82 = 1/irate, d=0.50 or d=0.95
"""
function estimation_1(; irate = 1.22, η_sd = 0.275, δ_mom = 0.0, N_init = 100)
    df = CSV.read("input/MATLAB_Input.csv", DataFrame, header = true)
    insertcols!(df, :log_c => log.(df.consumption))
    rename!(df, [:closesubtreatment => :close])

    N  = size(df, 1) # No. of households

    # Full sample
    sample = Vector{F64}()

    for d_annual in [1/irate, 0.5, 0.95]
        _, p_all = GMM_problem(df, δ_mom, d_annual, irate, η_sd, N_init)
        temp     = round(d_annual*100)
        CSV.write("output/MATLAB_Estimates_main_d$temp.csv", p_all)
    end

    # Standard errors // run bootstrapping.m
    # Input:  MATLAB_Input.csv
    # Output: bootstrap_**\bootstrap_[something]_d**.csv files
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
function counterfactuals_1()
    ##################
    # fixed parameters
    ##################

    η_sd = 0.275;
    irate = 1.22;
    danual = 1/irate;
    irm    = 1/irate;
    δ = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5);

    ##################
    # Input parameters manually
    # these are the baseline estimated parameters (danual = 1/irate)
    ##################

    μ_ϵ        = -79681;
    σ_ϵ       = 59715;
    α         = 0.5049;
    λ_con_belief    = 8.0448;
    λ_β_belief   = -0.71673;

    # full sample
    sample = [];

    for small_sample_dummy = 0:1

        # load full data
        load_data(sample,small_sample_dummy);
        global hhid quant_idx;

        # Column 1 is showup, Column 2 is showuphat
        [ showup_hat, ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
        showup_table   = [table(hhid), table(quant_idx), table(showup_hat)];

        # Column 3 Half Standard Deviation of epsilon
        [ showup_hat_halfsd , ~ ] = showuphat(μ_ϵ, σ_ϵ/2, α, λ_con_belief, λ_β_belief, η_sd, δ);
        showup_table   = [showup_table, table(showup_hat_halfsd)];

        # Column 4 No epsilon variance
        [ showup_hat_noeps , ~ ] = showuphat(μ_ϵ, σ_ϵ/1e10, α, λ_con_belief, λ_β_belief, η_sd, δ);
        showup_table   = [showup_table, table(showup_hat_noeps)];

        ### replace travel cost by same travel cost
        global totcost_smth_pc totcost_pc
        totcost_pc = totcost_smth_pc; ##ok<NASGU>

        # Column 5 no differential travel cost
        [ showup_hat_smthc , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
        showup_table   = [showup_table, table(showup_hat_smthc)];

        ### reload data, and assume constanta MU and λ
        load_data(sample,small_sample_dummy);
        mean_mu = 0.0967742; # mean benefit receipt conditional on applying

        global μ_con_true μ_β_true FE FE2;
        λ_con_belief_cml = norminv(mean_mu);
        λ_β_belief_cml = 0;
        μ_con_true = norminv(mean_mu);
        μ_β_true = 0;
        FE = FE*0;
        FE2 = FE2*0;

        # Column 6 (constant mu AND λ, update from the previous draft)
        [ showup_hat_cml  , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief_cml, λ_β_belief_cml, η_sd, δ);
        showup_table   = [showup_table, table(showup_hat_cml)];


        if(small_sample_dummy==1)
            global close
            showup_table   = [showup_table, table(close)];
            writetable(showup_table, [tempfolder 'MATLAB_showuphat_small_sample.csv']);

        else

            # Column 7: 3 extra km
            load_data(sample,small_sample_dummy);
            global totcost_pc totcost_3k_pc close
            totcost_pc = (1-close).*totcost_3k_pc + close.*totcost_pc;
            [ showup_hat_km3 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_km3)];

            # Column 8: 6 extra km
            load_data(sample,small_sample_dummy);
            global totcost_pc totcost_6k_pc close
            totcost_pc = (1-close).*totcost_6k_pc + close.*totcost_pc;
            [ showup_hat_km6 , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_km6)];

            # Column 9: 3x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(2.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_3aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_3aw)];

            # Column 10: 6x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(5.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_6aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_6aw)];


            # Column 11-12 α=0 (all-unsophisticated) and α=1 (all sophisticated)
            load_data(sample,small_sample_dummy);
            [ showup_hat_α0 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 0, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_α0)];

            [ showup_hat_α1 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 1, λ_con_belief, λ_β_belief, η_sd, δ);
            showup_table   = [showup_table, table(showup_hat_α1)];

            writetable(showup_table, [tempfolder 'MATLAB_showuphat.csv']);
        end

    end
end
