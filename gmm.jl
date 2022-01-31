function compute_quantiles(df::DataFrame)
    N_q = Dict(:consumption => 5, :pmtscore => 3, :unobs_cons => 3, :distt => 4)

    # Helper functions
    quant(N::Int64) = [n/N for n=1:N-1]
    assign_q(x, quants) = [minimum(vcat(findall(>=(x[i]), quants),
                                        length(quants)+1)) for i=1:length(x)]

    # Compute unobs. consumption (residual regressing log(obs consumption) on PMT)
    insertcols!(df, :log_c => log.(df.consumption))
    insertcols!(df, :unobs_cons => residuals(reg(df, @formula(log_c ~ pmtscore)), df))

    # Quantilies of observed consumption, PMT, unobs. consumption (w), distance
    for v in [:consumption, :pmtscore, :unobs_cons, :distt]
        col_q_idx = Symbol(string(v) * "_q_idx"))
        insertcols!(df, col_q_idx => assign_q(df[:,v], quantile(df[:,v] quant(N_q[v])))
    end
    
    my_q = Dict()
    my_q[:consumption] = quantile(df.consumption, quant(N_q[:c]))
    my_q[:pmtscore]    = quantile(df.pmtscore,    quant(N_q[:pmt]))
    my_q[:unobs_cons]  = quantile(df.unobs_cons,  quant(N_q[:unobs_cons]))
    my_q[:distt]       = quantile(df.distt,       quant(N_q[:distt]))

    # Categoricals for quantiles
    insertcols!(df, :obs_cons_terc    => assign_q(df.consumption,
                                                    quantile(df.consumption, quant(3))))
    insertcols!(df, :obs_cons_q_idx   => assign_q(df.consumption, my_q[:consumption]))
    insertcols!(df, :pmt_q_idx        => assign_q(df.pmtscore,    my_q[:pmtscore]))
    insertcols!(df, :unobs_cons_q_idx => assign_q(df.unobs_cons,  my_q[:unobs_cons]))
    insertcols!(df, :distt_q_idx      => assign_q(df.distt,       my_q[:distt]))

    return df
end

function showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta)

    global consumption totcost_pc benefit FE FE2 moneycost pmtscore
    global mu_con_true mu_beta_true


    mu_con_true = reg_const2(1);
    mu_beta_true = reg_pmt2(1);

    lambda_con_true = reg_nofe_const(1);
    lambda_beta_true = reg_nofe_logcon(1);


    # convert mean and standard deviation into alpha and beta
    s = sqrt(3*(sig_eps.^2)./(pi.^2))
    alphaa = mu_eps./s
    beta = 1./s

    # Lower and upper bound, grid density for the numerical integration over eta
    lb = -eta_sd*4 ub = -lb
    dense = 100
    siz = size(consumption,1)

    #Prepare meshgrid
    eta = linspace(lb,ub,dense)'
    ETA = eta * ones(1,siz)
    Consumption = ones(dense,1)*(consumption')
    Pmtscore = ones(dense,1)*(pmtscore')
    Totcost_pc = ones(dense,1)*(totcost_pc')
    Benefit =  ones(dense,1)*(benefit')
    FFE2 = ones(dense,1)*(FE2')
    FFE  = ones(dense,1)*(FE')
    Moneycost = ones(dense,1)*(moneycost')

    #Calculate integrand
    #Present Utility (Cost function is adjusted because "totalcost" given is based on noisy consumption measure.)
    relu_2day = (Consumption.*exp(-ETA)-Totcost_pc.*exp(-ETA)+(1-1.*exp(-ETA)).*Moneycost)-(Consumption.*exp(-ETA))

    #Future Utility: note that normcdf() in the middle is mu function
    relu_2mor = (Consumption.*exp(-ETA)+Benefit)-(Consumption.*exp(-ETA))


    Mu     = normcdf(mu_con_true+FFE2+mu_beta_true*(Pmtscore-ETA))
    Lambda = normcdf(lambda_con_belief+lambda_beta_belief*(log(Consumption)-ETA))

    prob_s = (1-1./(1+exp(beta.*(relu_2day + 12.*delta.*Mu    .*relu_2mor) + alphaa)))
    prob_u = (1-1./(1+exp(beta.*(relu_2day + 12.*delta.*Lambda.*relu_2mor) + alphaa)))

    integ =(alpha*prob_s + (1-alpha)*prob_u ).*normpdf(ETA, 0, eta_sd)

    showup_hat = real(trapz(eta,integ)')

    # Rather than running probit, apply WLS.
    # Calculate inverse of mu Phi(-1)(mu) where Phi is standard normal CDF
    muinv = mu_con_true+FE2+mu_beta_true*(pmtscore)
    muinv_t = sqrt(showup_hat).*muinv # Conversion for the frequency weight
    const_t = sqrt(showup_hat) # Conversion for the weight
    logcons_t = sqrt(showup_hat).*log(consumption) # Conversion for the weight
    X = [const_t, logcons_t]
    sigma2 = sqrt(sum(((eye(size(consumption,1))-X/(X'*X)*X')*muinv_t).^2)/(size(consumption,1)-2))
    coef = 1/sigma2*(X'*X)\X'*muinv_t # Divide by sigma to impose sd = 1 for the error term
    induced_lambda = normcdf([ones(size(consumption)),log(consumption)]*coef)

#         initial_lambda = normcdf(lambda_con_belief+lambda_beta_belief*log(consumption))
#         true_lambda    = normcdf(lambda_con_true+lambda_beta_true*log(consumption))
    return showup_hat, induced_lambda
end

"""
###########################################################################
# Moments:
# 1. mean showup rates in (measured) consumption quintiles in far and close
     subtreatment (separately). -> Ten moments
# 2. mean(showup-showuphat) of four extreme cells in grid of [tertiles of
#    pmt * tertiles of w] (residual from the regression of log(consumption) on
#    pmtscore) -> Four moments
# 3. two extreme cells in quartiles of distance -> Two moments
# 4. Lambda moments -> Four moments
###########################################################################
"""
"""
```
compute_moments(df0::D, showup::Vector{T}) where {T<:Number,D<:DataFrame}
```
I'm sorry these functions are so ugly, and that there are two of them.
(Multiple dispatch: want functions type-stable for Optim speed.)
"""
function compute_moments(df0::D, showup::Vector{T} =
                            Vector(df0.showup)) where {T<:Union{Float64,Int64}, D<:DataFrame}
    moments = Dict()
    moments[1] = [mean(showup[(df0.obs_cons_quant .== i) .&
                              iszero.(df0.closesubtreatment)]) for i=1:5]
    moments[2] = [mean(showup[(df0.obs_cons_quant .== i) .&
                              .!iszero.(df0.closesubtreatment)]) for i=1:5]
    moments[3] =  mean(showup[(df0.obs_cons_terc .== 3) .& (df0.unobs_cons_quant .== 1)])
    moments[4] =  mean(showup[(df0.obs_cons_terc .== 3) .& (df0.unobs_cons_quant .== 3)])
    moments[5] =  mean(showup[(df0.obs_cons_terc .== 1) .& (df0.unobs_cons_quant .== 1)])
    moments[6] =  mean(showup[(df0.obs_cons_terc .== 1) .& (df0.unobs_cons_quant .== 3)])
    moments[7] =  mean(showup[df0.distt_quant .== 4])
    moments[8] =  mean(showup[df0.distt_quant .== 1])
    return moments
end
function compute_moments(df0::D, show_i::T =
                         df0.showup) where {T<:Union{Float64,Int64}, D<:DataFrameRow}
    moments = Dict()
    moments[1] = [((df0.obs_cons_quant .== i) .&
                  iszero.(df0.closesubtreatment)) ? show_i : 0.0 for i=1:5]
    moments[2] = [((df0.obs_cons_quant .== i) .&
                  .!iszero.(df0.closesubtreatment)) ? show_i : 0.0 for i=1:5]
    moments[3] =  (df0.obs_cons_terc .== 3) .& (df0.unobs_cons_quant .== 1) ? show_i : 0.0
    moments[4] =  (df0.obs_cons_terc .== 3) .& (df0.unobs_cons_quant .== 3) ? show_i : 0.0
    moments[5] =  (df0.obs_cons_terc .== 1) .& (df0.unobs_cons_quant .== 1) ? show_i : 0.0
    moments[6] =  (df0.obs_cons_terc .== 1) .& (df0.unobs_cons_quant .== 3) ? show_i : 0.0
    moments[7] =  df0.distt_quant .== 4 ? show_i : 0.0
    moments[8] =  df0.distt_quant .== 1 ? show_i : 0.0
    return moments
end
function moments(showup_hat, true_lambda, belief_lambda, induced_lambda, delta_mom)

    global consumption showup close nquant
    global quant_idx quant_idx_pmt quant_idx_w quant_idx_dist
    global q_pmt_max q_pmt_min q_w_max q_w_min q_dist_max q_dist_min
    global getbenefit

    # Store moments here
    N    = size(consumption, 1)
    moms = zeros(N, nquant*2+10)

    for i=1:5 # should be nquant, not 5, ideally

        # showup rates in far group by consuption quintile

        Ntemp =              sum(quant_idx==i & close==0)
        moms[:,i] =     showup.*(quant_idx==i & close==0)*N/Ntemp -
                    showup_hat.*(quant_idx==i & close==0)*N/Ntemp

        # if delta_mom=1 we use the momserence between far and close (in
        # each quintile). If delta_mom=0 we use the showup mean in the
        # close consumption quintile

        Ntemp  =              sum(quant_idx==i & close==1)
        Ntemp0 =              sum(quant_idx==i & close==0)

        moms[:,5+i] =    showup.*(quant_idx==i & close==1)*N/Ntemp -
                         showup.*(quant_idx==i & close==0)*N/Ntemp0*delta_mom  -
                     showup_hat.*(quant_idx==i & close==1)*N/Ntemp +
                     showup_hat.*(quant_idx==i & close==0)*N/Ntemp0*delta_mom
    end

    Ntemp =                     sum(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_min)
    moms[:,nquant*2+1] =     showup.*(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_min)*N/Ntemp -
                       showup_hat.*(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_min)*N/Ntemp

    Ntemp =                     sum(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_max)
    moms[:,nquant*2+2] =     showup.*(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_max)*N/Ntemp -
                       showup_hat.*(quant_idx_pmt==q_pmt_max & quant_idx_w==q_w_max)*N/Ntemp

    Ntemp =                     sum(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_min)
    moms[:,nquant*2+3] =     showup.*(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_min)*N/Ntemp -
                       showup_hat.*(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_min)*N/Ntemp

    Ntemp =                     sum(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_max)
    moms[:,nquant*2+4] =     showup.*(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_max)*N/Ntemp -
                       showup_hat.*(quant_idx_pmt==q_pmt_min & quant_idx_w==q_w_max)*N/Ntemp

    Ntemp =                     sum(quant_idx_dist==q_dist_max)
    moms[:,nquant*2+5] =     showup.*(quant_idx_dist==q_dist_max)*N/Ntemp -
                       showup_hat.*(quant_idx_dist==q_dist_max)*N/Ntemp

    Ntemp =                     sum(quant_idx_dist==q_dist_min)
    moms[:,nquant*2+6] =     showup.*(quant_idx_dist==q_dist_min)*N/Ntemp -
                       showup_hat.*(quant_idx_dist==q_dist_min)*N/Ntemp

    Ntemp = sum(showup==1)
    moms[:,nquant*2+7] = (belief_lambda-getbenefit).*(showup==1)*N/Ntemp
    moms[:,nquant*2+8] = (belief_lambda-getbenefit).*(log(consumption)-mean(log(consumption))).*(showup==1)*N/Ntemp

    moms[:,nquant*2+9] = (induced_lambda-getbenefit).*(showup==1)*N/Ntemp
    moms[:,nquant*2+10] =(induced_lambda-getbenefit).*(log(consumption)-mean(log(consumption))).*(showup==1)*N/Ntemp

    # if showup hat identically zero the lambda moments are NaN
    # replace with large number
    if !isempty(findall(isnan, moms))
        @show "Error: isnan in moments. Replacing with large number."
        moms[isnan.(moms)] .= 10000
    end
end



function GMM_obj(x, eta_sd, delta, Whalf, delta_mom)
    #GMM_obj compute the product between the moment values (at given parameter
    #values) and Cholsky half of the weighting matrix

    global consumption
    global lambda_con_true lambda_beta_true

    mu_eps  = x(1)
    sig_eps = x(2)
    alpha   = x(3)
    lambda_con_belief  = x(4)
    lambda_beta_belief = x(5)

    [ showup_hat, induced_lambda ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta)

    belief_lambda  = normcdf(lambda_con_belief+lambda_beta_belief*log(consumption))
    true_lambda    = normcdf(lambda_con_true  +lambda_beta_true  *log(consumption))

    moms = moments( showup_hat, true_lambda, belief_lambda, induced_lambda, delta_mom )

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
    showuphat_return, induced_lambda_return = showuphat(fitparams[1:5], eta_sd, delta)
    return fitparams, fval, showuphat_return, induced_lambda_return, exitflag

end

"""
Two Stage Feasible GMM for Targeting Paper

# Parameters:
# 1~2. mean, variance of epsilon (utility shock)
# 3.   sigma (coef of relative risk aversion = rho, sorry for the confusion)
# 4.   alpha (fraction of people who are sophisticated)
# 5~6. constant and coefficient for the lambda function (probability of
       getting benefit given showing up)

## Minimize Objective Function Locally From Many Starting Values
#	Use matlab's built-in lsqnonlin function to optimize from a random set
#	of initial values.

## How are moments computed?
#	For each obs, calculate showup_hat as the probability that eps>-gain
#	(i.e. that gain+epsilon>0) This is just 1 - the cdf
#	of epsilon evaluated at (-gain_i).
"""
function GMM_problem(delta_mom, danual, irate, eta_sd, passes, rng_seed,sample)

    # load data as globals
    small_sample_dummy = 0 # use regular data for estimation
    load_data(sample,small_sample_dummy)

    # consumption
    # global lambda_con_true lambda_beta_true

    # Initialize random number generator
    rng(rng_seed)

    # generate random sets of #passes possible initial values for each parameter
    # "get benefit" function, lambda for the naive people
    lambda_con_rand  = lambda_con_true*(0.8+0.4*rand(passes,1))
    lambda_beta_rand = lambda_beta_true*(0.8+0.4*rand(passes,1)) # These two are parameters to match the initial value

    # Currently imposing some function of random rho + noise
    sig_eps_rand = 26805.*(0.8+0.4*rand(passes,1))
    mu_eps_rand = -26215.*(0.8+0.4*rand(passes,1))

    alpha_rand = 0.001+0.999*rand(passes,1)

    # compute NPV delta (yearly)
    irm = 1/irate
    delta = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5)


    ## First Stage: Find params with identity weighting matrix
    @show "First Stage Begin"

    Fval_fs = zeros(passes,1)
    Fitθ_fs = zeros(passes,5)
    Showup_hat_fs     = zeros(size(consumption,1),passes)
    Induced_lambda_fs = zeros(size(consumption,1),passes)
    exitflag_fs       = zeros(passes,1)

    for pass = 1:passes
        # select initial parameters
        θ_in = [   mu_eps_rand[pass,1],
                        sig_eps_rand[pass,1],
                        alpha_rand[pass,1],
                        lambda_con_rand[pass,1],
                        lambda_beta_rand[pass,1]]

        # begin with identity weigthing matrix
        W0 = eye(20)

        [Fitθ_fs[pass,:], Fval_fs[pass], Showup_hat_fs[:,pass],
        Induced_lambda_fs[:,pass], exitflag_fs[pass]] = run_GMM(θ_in, eta_sd, delta, W0, delta_mom)
    end

    minindex = argmin(Fval_fs)
    θ_fs     = Fitθ_fs[minindex,:]

    ## Second Stage: Find Params with Optimal Weighting Matrix
    @show "Second Stage Begin"

    # Calculate Optimal Weighting Matrix
    # GMM_obj using W0 = identity returns the moment values

    #θ_fs = [-67020	60897	0.22044	8.6504	-0.77535]
     mu_eps  = θ_fs[1]
     sig_eps = θ_fs[2]
     alpha   = θ_fs[3]
     lambda_con_belief  = θ_fs[4]
     lambda_beta_belief = θ_fs[5]

     [ showup_hat, induced_lambda ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta)

     belief_lambda  = normcdf(lambda_con_belief+lambda_beta_belief*log(consumption))
     true_lambda    = normcdf(lambda_con_true  +lambda_beta_true  *log(consumption))

    moms = moments( showup_hat, true_lambda, belief_lambda, induced_lambda, delta_mom )

N = size(consumption,1)
    Winv = (moms'*moms)/N
    W = inv(Winv)
    Whalf = chol(W)'
    norm((Whalf*Whalf')-W) #check decomposition worked.

    #clear mu_eps sig_eps alpha lambda_con_belief lambda_beta_belief

    Fval_ss           = zeros(passes,1)
    Fitθ_ss           = zeros(passes,5)
    Showup_hat_ss     = zeros(size(consumption,1),passes)
    Induced_lambda_ss = zeros(size(consumption,1),passes)
    exitflag_ss       = zeros(passes,1)

    for pass = 1:passes
        tic
        # select initial parameters (the same ones as in first stage)
        θ_in = [   mu_eps_rand(pass,1),
                        sig_eps_rand(pass,1),
                        alpha_rand(pass,1),
                        lambda_con_rand(pass,1),
                        lambda_beta_rand(pass,1)]

        # use optimal weigthing matrix Whalf
        [Fitθ_ss[pass,:], Fval_ss[pass], Showup_hat_ss[:,pass], Induced_lambda_ss[:,pass], exitflag_ss[pass]] =
            run_GMM(  θ_in, eta_sd, delta, Whalf, delta_mom )

        disp(['Second Stage pass # ' num2str(pass) ' took ' num2str(toc)])
    end

    [~, minindex] = min(Fval_ss)
    ismax = (1:passes)==minindex
    # θ_ss = Fitθ_ss(minindex,:)
    #showup__hat = Showup_hat_ss[:,minindex)
    #induced_lambda_ss = Induced_lambda_ss[:,minindex)

    all_θ = [mu_eps_rand, Fitθ_fs[:,1], Fitθ_ss[:,1],
                  sig_eps_rand,Fitθ_fs[:,2], Fitθ_ss[:,2],
                  alpha_rand,  Fitθ_fs[:,3], Fitθ_ss[:,3],
                  lambda_con_rand,  Fitθ_fs[:,4], Fitθ_ss[:,4],
                  lambda_beta_rand, Fitθ_fs[:,5], Fitθ_ss[:,5],
                  Fval_fs, Fval_ss, exitflag_fs, exitflag_ss, ismax']

    disp('Second Stage Done')

    return W, all_θ
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

        Fval_ss = p_all[:, 17)
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

function counterfactuals_1()
    %%%%%%%%%%%%%%%%%%
    % fixed parameters
    %%%%%%%%%%%%%%%%%%

    eta_sd = 0.275;
    irate = 1.22;
    danual = 1/irate;
    irm    = 1/irate;
    delta = danual*(1 + irm + irm^2 + irm^3 + irm^4 + irm^5);

    %%%%%%%%%%%%%%%%%%
    % Input parameters manually
    % these are the baseline estimated parameters (danual = 1/irate)
    %%%%%%%%%%%%%%%%%%

    mu_eps        = -79681;
    sig_eps       = 59715;
    alpha         = 0.5049;
    lambda_con_belief    = 8.0448;
    lambda_beta_belief   = -0.71673;

    % full sample
    sample = [];

    for small_sample_dummy = 0:1

        % load full data
        load_data(sample,small_sample_dummy);
        global hhid quant_idx;

        % Column 1 is showup, Column 2 is showuphat
        [ showup_hat, ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
        showup_table   = [table(hhid), table(quant_idx), table(showup_hat)];

        % Column 3 Half Standard Deviation of epsilon
        [ showup_hat_halfsd , ~ ] = showuphat(mu_eps, sig_eps/2, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_halfsd)];

        % Column 4 No epsilon variance
        [ showup_hat_noeps , ~ ] = showuphat(mu_eps, sig_eps/1e10, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_noeps)];

        %%% replace travel cost by same travel cost
        global totcost_smth_pc totcost_pc
        totcost_pc = totcost_smth_pc; %#ok<NASGU>

        % Column 5 no differential travel cost
        [ showup_hat_smthc , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_smthc)];

        %%% reload data, and assume constanta MU and LAMBDA
        load_data(sample,small_sample_dummy);
        mean_mu = 0.0967742; % mean benefit receipt conditional on applying

        global mu_con_true mu_beta_true FE FE2;
        lambda_con_belief_cml = norminv(mean_mu);
        lambda_beta_belief_cml = 0;
        mu_con_true = norminv(mean_mu);
        mu_beta_true = 0;
        FE = FE*0;
        FE2 = FE2*0;

        % Column 6 (constant mu AND lambda, update from the previous draft)
        [ showup_hat_cml  , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief_cml, lambda_beta_belief_cml, eta_sd, delta);
        showup_table   = [showup_table, table(showup_hat_cml)];


        if(small_sample_dummy==1)
            global close
            showup_table   = [showup_table, table(close)];
            writetable(showup_table, [tempfolder 'MATLAB_showuphat_small_sample.csv']);

        else

            % Column 7: 3 extra km
            load_data(sample,small_sample_dummy);
            global totcost_pc totcost_3k_pc close
            totcost_pc = (1-close).*totcost_3k_pc + close.*totcost_pc;
            [ showup_hat_km3 , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_km3)];

            % Column 8: 6 extra km
            load_data(sample,small_sample_dummy);
            global totcost_pc totcost_6k_pc close
            totcost_pc = (1-close).*totcost_6k_pc + close.*totcost_pc;
            [ showup_hat_km6 , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_km6)];

            % Column 9: 3x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(2.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_3aw , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_3aw)];

            % Column 10: 6x waiting time
            load_data(sample,small_sample_dummy);
            global totcost_pc hhsize close ave_waiting wagerate
            totcost_pc = totcost_pc + (1-close).*(5.*ave_waiting.*wagerate)./(hhsize.*60);
            [ showup_hat_6aw , ~ ] = showuphat(mu_eps, sig_eps, alpha, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_6aw)];


            % Column 11-12 alpha=0 (all-unsophisticated) and alpha=1 (all sophisticated)
            load_data(sample,small_sample_dummy);
            [ showup_hat_alpha0 , ~ ] = showuphat(mu_eps, sig_eps, 0, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_alpha0)];

            [ showup_hat_alpha1 , ~ ] = showuphat(mu_eps, sig_eps, 1, lambda_con_belief, lambda_beta_belief, eta_sd, delta);
            showup_table   = [showup_table, table(showup_hat_alpha1)];

            writetable(showup_table, [tempfolder 'MATLAB_showuphat.csv']);
        end

    end
end
# # counterfactual showup hat used in Tables 9 and 10, and
# # Online Appendix Table C.18.
# run counterfactuals_1.m
#     # Input:  MATLAB_Input.csv,
#     #         MATLAB_Input_small_sample.csv
#     # Output: MATLAB_showuphat_small_sample.csv
#     #         MATLAB_showuphat.csv
