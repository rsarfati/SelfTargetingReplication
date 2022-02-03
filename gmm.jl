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
    !("logc"       in names(df)) && insertcols!(df, :log_c => log.(df.consumption))
    !("unobs_cons" in names(df)) && insertcols!(df, :unobs_cons =>
                                    residuals(reg(df, @formula(log_c ~ PMTSCORE)), df))

    # Assign categorical IDs for quantiles
    for v in keys(N_q)
        v_name = Symbol(string(v) * "_q_idx")
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

    # Far/close subtreatment, i=1:5th quintile of consumption
    for i=1:5
        far_i_quant   = (df0.consumption_q_idx .== i) .&
                                                  iszero.(df0.closesubtreatment)
        close_i_quant = (df0.consumption_q_idx .== i) .&
                                                .!iszero.(df0.closesubtreatment)
        moments[i]   = sum(showup[far_i_quant])   / sum(far_i_quant)
        moments[5+i] = sum(showup[close_i_quant]) / sum(close_i_quant)
    end

    # {Top, bottom tercile} x {observable, unobservable consumption}
    TB = (df0.PMTSCORE_q_idx .== 3) .& (df0.unobs_cons_q_idx .== 1)
    TT = (df0.PMTSCORE_q_idx .== 3) .& (df0.unobs_cons_q_idx .== 3)
    BB = (df0.PMTSCORE_q_idx .== 1) .& (df0.unobs_cons_q_idx .== 1)
    BT = (df0.PMTSCORE_q_idx .== 1) .& (df0.unobs_cons_q_idx .== 3)

    moments[11] =  sum(showup[TB] - showup_hat[TB]) / sum(TB)
    moments[12] =  sum(showup[TT] - showup_hat[TT]) / sum(TT)
    moments[13] =  sum(showup[BB] - showup_hat[BB]) / sum(BB)
    moments[14] =  sum(showup[BT] - showup_hat[BT]) / sum(BT)

    # Top and bottom distance quartiles
    T_D = (df0.distt_quant .== 4)
    B_D = (df0.distt_quant .== 1)

    moments[15] =  sum(showup[T_D] - showup_hat[T_D]) / sum(T_D)
    moments[16] =  sum(showup[B_D] - showup_hat[B_D]) / sum(B_D)

    # Mean λ function moments
    N_show = sum(showup)
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
function showuphat(df, μ_ϵ, σ_ϵ, α, λ_con_belief, λ_β_belief, η_sd, δ;
                   N_grid = 100)

    # Constants
    μ_con_true = df.reg_const2[1]
    μ_β_true   = df.reg_pmt2[1]
    λ_con_true = df.reg_nofe_const[1]
    λ_β_true   = df.reg_nofe_logcon[1]

    # convert mean and standard deviation into alpha and β
    s      = sqrt( 3 * (σ_ϵ .^ 2) ./ (pi .^ 2))
    alphaa = μ_ϵ ./ s
    β      = 1 ./ s

    # Lower and upper bound, grid density for the numerical integration over eta
    lb    = -η_sd*4
    ub    = -lb
    N   = length(df.consumption)

    # Prepare mesh grid
    eta         = linspace(lb, ub, N_grid)'
    ETA         = η * ones(1,λ)
    Consumption = ones(N_grid,1)*(consumption')
    Pmtscore    = ones(N_grid,1)*(pmtscore')
    Totcost_pc  = ones(N_grid,1)*(totcost_pc')
    Benefit     = ones(N_grid,1)*(benefit')
    FFE2        = ones(N_grid,1)*(FE2')
    FFE         = ones(N_grid,1)*(FE')
    Moneycost   = ones(N_grid,1)*(moneycost')

    # Calculate integrand
    # Present Utility (Cost function is adjusted because "totalcost" given is based on noisy consumption measure.)
    relu_2day = (Consumption .* exp.(-ETA) - Totcost_pc .* exp(-ETA) + (1 - 1 .* exp(-ETA)) .* Moneycost) - (Consumption .* exp.(-ETA))

    # Future Utility: note that normcdf() in the middle is mu function
    relu_2mor = (Consumption .* exp(-ETA) + Benefit) - (Consumption .* exp.(-ETA))
    Mu     = normcdf(μ_con_true + FFE2 + μ_β_true * (Pmtscore-ETA))
    Lambda = normcdf(λ_con_belief+λ_β_belief * (log.(Consumption)-ETA))

    prob_s = (1 - 1 ./ (1 + exp(β .* (relu_2day + 12 .* delta .* Mu     .* relu_2mor) + alphaa)))
    prob_u = (1 - 1 ./ (1 + exp(β .* (relu_2day + 12 .* delta .* Lambda .* relu_2mor) + alphaa)))

    integ = (alpha * prob_s + (1 - alpha) * prob_u ) .* normpdf(ETA, 0, eta_sd)

    showup_hat = real(trapz(eta,integ)')

    # Rather than running probit, apply WLS.
    # Calculate inverse of mu Phi(-1)(mu) where Phi is standard normal CDF
    muinv = μ_con_true + FE2 + μ_β_true * (pmtscore)
    muinv_t = sqrt(showup_hat).*muinv # Conversion for the frequency weight
    const_t = sqrt(showup_hat) # Conversion for the weight
    logcons_t = sqrt(showup_hat).*log(consumption) # Conversion for the weight
    X = [const_t, logcons_t]
    sigma2 = sqrt(sum(((eye(N) - X / (X' * X) * X') * muinv_t) .^ 2) / (N - 2))
    coef = 1/sigma2*(X'*X)\X'*muinv_t # Divide by sigma to impose sd = 1 for the error term
    induced_λ = normcdf([ones(N,1), log.(consumption)] * coef)
    return showup_hat, induced_λ
end

"""
Two Stage Feasible GMM for Targeting Paper

# Parameters:
# 1~2. mean, variance of epsilon (utility shock)
# 3.   sigma (coef of relative risk aversion = rho, sorry for the confusion)
# 4.   alpha (fraction of people who are sophisticated)
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
function GMM_problem(delta_mom, danual, irate, eta_sd, passes, rng_seed, sample)

    df = compute_quantiles(df)

    λ_con_true = df.reg_nofe_const[1]
    λ_β_true   = df.reg_nofe_logcon[1]

    # Initialize random number generator
    rng(rng_seed)

    # generate random sets of #passes possible initial values for each parameter
    # "get benefit" function, λ for the naive people
    λ_con_rand = λ_con_true  * (0.8 + 0.4 * rand(passes,1))
    λ_β_rand = λ_β_true * (0.8 + 0.4 * rand(passes,1)) # These two are parameters to match the initial value

    # Currently imposing some function of random rho + noise
    σ_ϵ_rand = 26805 .* (0.8 .+ 0.4 .* rand(passes,1))
    μ_ϵ_rand = -26215 .* (0.8 .+ 0.4 .* rand(passes,1))

    alpha_rand = 0.001+0.999*rand(passes,1)

    # compute NPV delta (yearly)
    irm = 1 / irate
    delta = danual * (1 + irm + irm^2 + irm^3 + irm^4 + irm^5)

    ## First Stage: Find params with identity weighting matrix
    @show "First Stage Begin"

    Fval_fs       = zeros(passes,1)
    Fitθ_fs       = zeros(passes,5)
    Showup_hat_fs = zeros(size(consumption,1),passes)
    Induced_λ_fs  = zeros(size(consumption,1),passes)
    exitflag_fs   = zeros(passes,1)

    for p = 1:passes
        # select initial parameters
        θ_in = [μ_ϵ_rand[p,1],
                σ_ϵ_rand[p,1],
                alpha_rand[p,1],
                λ_con_rand[p,1],
                λ_β_rand[p,1]]

        # begin with identity weigthing matrix
        W0 = eye(20)

        Fitθ_fs[p,:], Fval_fs[p], Showup_hat_fs[:,p],
        Induced_λ_fs[:,p], exitflag_fs[p] = run_GMM(θ_in, eta_sd, delta, W0, delta_mom)
    end

    minindex = argmin(Fval_fs)
    θ_fs     = Fitθ_fs[minindex,:]

    ## Second Stage: Find Params with Optimal Weighting Matrix
    @show "Second Stage Begin"
    #θ_fs = [-67020	60897	0.22044	8.6504	-0.77535]
     μ_ϵ  = θ_fs[1]
     σ_ϵ = θ_fs[2]
     alpha   = θ_fs[3]
     λ_con_belief  = θ_fs[4]
     λ_β_belief = θ_fs[5]

     showup_hat, induced_λ = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta)

     belief_λ  = normcdf(λ_con_belief+λ_β_belief*log(consumption))
     true_λ    = normcdf(λ_con_true  +λ_β_true  *log(consumption))

    moms = moments( showup_hat, true_λ, belief_λ, induced_λ, delta_mom )

    N = size(consumption,1)
    Winv = (moms' * moms) / N
    W = inv(Winv)
    Whalf = chol(W)'
    norm((Whalf * Whalf') - W) #check decomposition worked.

    Fval_ss       = zeros(passes,1)
    Fitθ_ss       = zeros(passes,5)
    Showup_hat_ss = zeros(size(consumption,1),passes)
    Induced_λ_ss  = zeros(size(consumption,1),passes)
    exitflag_ss   = zeros(passes,1)

    for p = 1:passes
        tic
        # select initial parameters (the same ones as in first stage)
        θ_in = [μ_ϵ_rand[p,1],
                σ_ϵ_rand[p,1],
                alpha_rand[p,1],
                λ_con_rand[p,1],
                λ_β_rand[p,1]]

        # use optimal weigthing matrix Whalf
        Fitθ_ss[p,:], Fval_ss[p], Showup_hat_ss[:,p], Induced_λ_ss[:,p], exitflag_ss[p] =
            run_GMM(  θ_in, eta_sd, delta, Whalf, delta_mom )
    end

    minindex = argmin(Fval_ss)
    ismax = (1:passes) .== minindex

    all_θ = [μ_ϵ_rand, Fitθ_fs[:,1], Fitθ_ss[:,1],
              σ_ϵ_rand,Fitθ_fs[:,2], Fitθ_ss[:,2],
              alpha_rand,  Fitθ_fs[:,3], Fitθ_ss[:,3],
              λ_con_rand,  Fitθ_fs[:,4], Fitθ_ss[:,4],
              λ_β_rand, Fitθ_fs[:,5], Fitθ_ss[:,5],
              Fval_fs, Fval_ss, exitflag_fs, exitflag_ss, ismax']

    return W, all_θ
end



function GMM_obj(x, eta_sd, delta, Whalf, delta_mom)
    #GMM_obj compute the product between the moment values (at given parameter
    #values) and Cholsky half of the weighting matrix
    global λ_con_true λ_β_true

    μ_ϵ  = x(1)
    σ_ϵ = x(2)
    alpha   = x(3)
    λ_con_belief  = x(4)
    λ_β_belief = x(5)

    showup_hat, induced_λ = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta)

    belief_λ  = normcdf(λ_con_belief+λ_β_belief*log(consumption))
    true_λ    = normcdf(λ_con_true  +λ_β_true  *log(consumption))

    moms = moments( showup_hat, true_λ, belief_λ, induced_λ, delta_mom )

    return mean(moms,1) * Whalf
end


function run_GMM(params_in, eta_sd, delta, Whalf, delta_mom)

    # lower and upper bounds for all the parameters
    lb1 = [-200000,     0,  0.001,  0, -2]
    ub1 = [ 200000, 200000, 0.999, 20,  1]

    # run lsqnonlin
    fitparams,fval, _, exitflag =
        lsqnonlin(@(x)GMM_obj(x, eta_sd, delta, Whalf, delta_mom), params_in, lb1, ub1, options2)

    # return showup_hat_return
    showuphat_return, induced_λ_return = showuphat(fitparams[1:5], eta_sd, delta)
    return fitparams, fval, showuphat_return, induced_λ_return, exitflag

end



# # Table 8. Estimated Parameter Values for the Model
# run estimation_1.m
#     # Input:  MATLAB_Input.csv
#     # Output: MATLAB_Estimates_main_d**.csv
#     #    ** corresponds to the value of the annual discount factor:
#             d=0.82 = 1/irate, d=0.50 or d=0.95
function estimation_1()
    irate  = 1.22
    η_sd   = 0.275
    δ_mom  = 0.0
    passes = 100

    # Full sample
    sample = Vector{F64}()

    danuals   = [1/irate, 0.5, 0.95]
    rng_seeds = [945739485, 28364192, 2983742]

    for rng_seed, danual in zip(danuals, rng_seeds)
        _, p_all = GMM_problem(delta_mom, danual, irate, eta_sd, passes, rng_seed, sample)
        temp = round(danuals*100)
        # Write p_all to CSV.
    end

    # number of parameters
    N_p = 20

    # Bootstrap SEs
    danual = 1/irate # alternatives: danual = 0.50 and = 0.95.

    N = size(consumption, 1)

    rng(3797265)
    n_boot = 100 # Number of bootstrap iterations
    passes = 50  # Number of passes (initial conditions) per iteration

    rng_seed = 10000 * rand(n_boot, 1)

    # store bootstrap subsample, estimated parameters, weighting matrices,
    # and all passes for each iteration
    boot_samples = zeros(n_boot, N)
    θ_boots = zeros(n_boot, N_p)
    weight_matrix = zeros(n_boot*N_p, N_p)
    θ_boots_all = zeros(n_boot*passes, N_p)

    # Will store showup_hat induced by the parameters of each iteration here
    #showuphat_boots = zeros(size(consumption,1),n_boot)

    for it=1:n_boot
        # randomly draw index of individuals (with replacement) for bootstrap
        boot_sample = randsample(1:N, N, true)
        boot_samples[it,:] = boot_sample # save sample

        [W, p_all] = GMM_problem(delta_mom, danual, irate, eta_sd, passes, rng_seed(it), boot_sample)

        Fval_ss = p_all[:, 17]
        [~, minindex] = min(Fval_ss)
        θ_boots[it,:] = p_all[minindex,:]
        θ_boots_all[ (it-1)*passes + (1:passes),:] = p_all
        weight_matrix[ (it-1)*N_p + (1:N_p),:] = W

        @show "Bootstrap iteration: $i. Last iteration took $i sec."

        # write output so far (overwrite)
        temp = round(danual*100)
        # csvwrite(["output/bootstrap_'  num2str(temp) '\bootstrap_estimates_' num2str(temp) '.csv"],θ_boots)
        # csvwrite(["output/bootstrap_'  num2str(temp) '\bootstrap_allpasses_' num2str(temp) '.csv"],θ_boots_all)
        # csvwrite(["output/bootstrap_'  num2str(temp) '\bootstrap_samples_' num2str(temp) '.csv"],boot_samples)
        # csvwrite(["output/bootstrap_'  num2str(temp) '\bootstrap_weight_matrix_' num2str(temp) '.csv"],weight_matrix)
    end
end

# # Standard errors
# #run bootstrapping.m
#     # run three times, changing line 19: danual  = 1/irate danual = 0.50 and danual = 0.95
#     # Input:  MATLAB_Input.csv
#     # Output: bootstrap_**\bootstrap_[something]_d**.csv files
#
# # print standard deviations
# boot_estimates_82 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_82\bootstrap_estimates_82.csv'])
# boot_estimates_50 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_50\bootstrap_estimates_50.csv'])
# boot_estimates_95 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_95\bootstrap_estimates_95.csv'])
#
# boot_estimates_82_std = std(boot_estimates_82[:,[3,6,9,12,15]))
# boot_estimates_50_std = std(boot_estimates_50[:,[3,6,9,12,15]))
# boot_estimates_95_std = std(boot_estimates_95[:,[3,6,9,12,15]))
#
# display(['Boostrap standard deviation for delta = 0.50 is ' num2str(boot_estimates_50_std,5)])
# display(['Boostrap standard deviation for delta = 0.82 is ' num2str(boot_estimates_82_std,5)])
# display(['Boostrap standard deviation for delta = 0.95 is ' num2str(boot_estimates_95_std,5)])
#

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

    eta_sd = 0.275;
    irate = 1.22;
    danual = 1/irate;
    irm    = 1/irate;
    delta = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5);

    ##################
    # Input parameters manually
    # these are the baseline estimated parameters (danual = 1/irate)
    ##################

    μ_ϵ        = -79681;
    σ_ϵ       = 59715;
    alpha         = 0.5049;
    λ_con_belief    = 8.0448;
    λ_β_belief   = -0.71673;

    # full sample
    sample = [];

    for small_sample_dummy = 0:1

        # load full data
        load_data(sample,small_sample_dummy);
        global hhid quant_idx;

        # Column 1 is showup, Column 2 is showuphat
        [ showup_hat, ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
        showup_table   = [table(hhid), table(quant_idx), table(showup_hat)];

        # Column 3 Half Standard Deviation of epsilon
        [ showup_hat_halfsd , ~ ] = showuphat(μ_ϵ, σ_ϵ/2, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_halfsd)];

        # Column 4 No epsilon variance
        [ showup_hat_noeps , ~ ] = showuphat(μ_ϵ, σ_ϵ/1e10, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_noeps)];

        ### replace travel cost by same travel cost
        global totcost_smth_pc totcost_pc
        totcost_pc = totcost_smth_pc; ##ok<NASGU>

        # Column 5 no differential travel cost
        [ showup_hat_smthc , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
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
        [ showup_hat_cml  , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief_cml, λ_β_belief_cml, eta_sd, delta);
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
            [ showup_hat_km3 , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_km3)];

            # Column 8: 6 extra km
            load_data(sample,small_sample_dummy);
            global totcost_pc totcost_6k_pc close
            totcost_pc = (1-close).*totcost_6k_pc + close.*totcost_pc;
            [ showup_hat_km6 , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_km6)];

            # Column 9: 3x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(2.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_3aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_3aw)];

            # Column 10: 6x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(5.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_6aw , ~ ] = showuphat(μ_ϵ, σ_ϵ, alpha, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_6aw)];


            # Column 11-12 alpha=0 (all-unsophisticated) and alpha=1 (all sophisticated)
            load_data(sample,small_sample_dummy);
            [ showup_hat_alpha0 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 0, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_alpha0)];

            [ showup_hat_alpha1 , ~ ] = showuphat(μ_ϵ, σ_ϵ, 1, λ_con_belief, λ_β_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_alpha1)];

            writetable(showup_table, [tempfolder 'MATLAB_showuphat.csv']);
        end

    end
end
