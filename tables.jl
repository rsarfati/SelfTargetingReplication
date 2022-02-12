###########################
# Table 1 statistics
###########################
function table_1(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load data, open output pipe
    df = DataFrame(load("input/matched_baseline.dta"))
    io = open("output/tables/Table1.tex", "w")

    # Directly write LaTeX table
    write(io, "\\begin{tabular}{lrrr}\\toprule\n & \\multicolumn{1}{c}")
    write(io, "{No. Villages (Households)}\\\\ \\midrule \n")
    @printf(io, "Automatic Screening & %3i (%4i)\\\\",
                length(unique(df.hhea[df.selftargeting .!= 1.0])),
                sum(df.selftargeting .!= 1.0))

    df = clean(df, [:hhea, :closesubtreatment, :selftargeting])

    write(  io, "Self-targeting: & \\\\")
    @printf(io, "\\hspace{2em} Close Treatment & %3i (%4i) \\\\",
                length(unique(df.hhea[df.closesubtreatment .== 1.])),
                length(       df.hhea[df.closesubtreatment .== 1.]))
    @printf(io, "\\hspace{2em} Far Treatment & %3i (%4i) \\\\",
                length(unique(df.hhea[df.closesubtreatment .== 0.])),
                length(       df.hhea[df.closesubtreatment .== 0.]))
    @printf(io, "\\hspace{4em} Total & %3i (%4i) \\\\",
                length(unique(df.hhea[df.selftargeting .== 1])),
                length(       df.hhid[df.selftargeting .== 1]))
    write(io, "\\bottomrule\\end{tabular}")
    close(io)
end

###########################
# Table 3 Statistics
###########################
function table_3(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load data, open output pipe
    df = DataFrame(load("input/matched_baseline.dta"))
    io = open("output/tables/Table3.tex", "w")

    # Minimize redundant typing
    col_w(x::F64) = "\\multicolumn{1}{p{" * string(x) * "0.1\\linewidth}}"

    write(io, "\\footnotesize\\begin{tabular}{lcccccc} \\toprule")
    write(io, " & $(col_w(0.1)){\\centering Total No. \\\\Households}")
    write(io, " & $(col_w(0.1)){\\centering No. Households \\\\ Interviewed}")
    write(io, " & $(col_w(0.1)){\\centering No. \\\\ Beneficiaries}")
    write(io, " & $(col_w(0.1)){\\centering Households \\\\ Interviewed (\\%)}")
    write(io, " & $(col_w(0.19)){\\centering Interviewed\\\\Households That")
    write(io,                   "\\\\Received Benefits (\\%)}")
    write(io, " & $(col_w(0.15)){\\centering Tot. Households\\\\")
    write(io,                "That Received\\\\Benefits (\\%)}\\\\\\hline\\\\")

    selft = (df.selftargeting .== 1.0)
    intv  = (df.showup        .== 1.0)
    ben   = (df.getbenefit    .== 1.0)

    t3_stats(x::BitVector) = [sum(x), sum(x .& intv), sum(x .& ben),
                              100*sum(x .& intv) / sum(x),
                              100*sum(x .& intv .& ben) / sum(x .& intv),
                              100*sum(x .& ben) / sum(x)]

    @printf(io, "%s & %4i & %3i & %2i & %0.2f & %0.2f & %0.2f \\\\[1mm]",
            "Auto Screening", t3_stats(.!selft)...)
    @printf(io, "%s & %4i & %3i & %2i & %0.2f & %0.2f & %0.2f \\\\[1mm]",
            "Self-targeting", t3_stats(selft)...)

    write(io, "\\bottomrule\n\\end{tabular}")
    close(io)
end

###########################
# Table 4 regressions
###########################
function table_4(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load data
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption => :logc, :closesubtreatment => :close,
                 :povertyline3 => :pov])

    # Drop non self-targeting households
    df = df[df.selftargeting .== 1, :]
    df = clean(df, [:showup, :PMTSCORE, :logc], F64)

    # Generate residuals
    r = reg(df, @formula(logc ~ PMTSCORE), cluster(:hhea), save = :residuals)
    insertcols!(df, :eps => Vector{Float64}(r.residuals))

    # Specify formula of interest
    f = @formula(showup ~ PMTSCORE + eps)

    # Run 3 regression specifications
    μ,    r    = Vector{F64}(undef, 3), Vector{FixedEffectModel}(undef, 3)
    μ[1], r[1] = glm_clust(f, df; clust = :hhea)
    μ[2], r[2] = glm_clust(f, df[df.pcexppred .<  df.pov,:]; clust = :hhea)
    μ[3], r[3] = glm_clust(f, df[df.pcexppred .>= df.pov,:]; clust = :hhea)

    # Print output
    mystats = NamedTuple{(:comments, :means)}((["No", "No", "No"], μ))
    regtable(r...; renderSettings = latexOutput("output/tables/Table4.tex"),
    		 regressors = ["PMTSCORE", "eps"],
             custom_statistics = mystats, table_kwargs...)
    return r, μ
end

###########################
# Table 5 regressions
###########################
function table_5(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load baseline data
    pre = "input/matched"
    df = DataFrame(load("$(pre)_baseline.dta"))
    rename!(df, [:logconsumption => :logc, :closesubtreatment => :close,
                 :selftargeting => :self])
    insertcols!(df, :base_or_end => 0.0, :get => (df.getbenefit .== 1),
                     :benefit => categorical(df.getbenefit))

    # Load midline data
    df_m = clean(DataFrame(load("$(pre)_midline.dta")),[:flag_newHH],F64)
    df_m = df_m[df_m.flag_newHH .== 1.0, :]
    rename!(df_m, [:logconsumption => :logc, :closesubtreatment => :close,
                   :selftargeting => :self])
    df_m = convert_types(df_m, [:logc, :showup] .=> F64)
    insertcols!(df_m, :base_or_end => 1.0, :get => (df_m.getbenefit .== 1))

    # Includes baseline and midline data
    df_bm = vcat(df, df_m, cols = :union)
    df_bm = clean(df_bm[df_bm.get .== 1, :], [:logc], F64)

    ####################################
    # Analysis (w/o Stratum FEs) - MATCH
    ####################################

    # Store output
    μ_5a, r_5a = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)
    μ_5b, r_5b = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)

    μ_5a[1], r_5a[1] = mean(df[df.get, :logc]),
                        reg(df[df.get, :], @formula(logc~self), cluster(:hhea))

    μ_5a[2], r_5a[2] = mean(df_bm.logc),
                        reg(df_bm, @formula(logc ~ self + base_or_end),
                            cluster(:hhea))

    μ_5a[3], r_5a[3] = glm_clust(@formula(get ~ self + logc + logc * self),
                           clean(df, [:logc, :get, :self], F64); clust = :hhea)

    μ_5a[4], r_5a[4] = glm_clust(@formula(mistarget ~ self),
                                 clean(df, [:mistarget], F64); clust = :hhea)

    μ_5a[5], r_5a[5] = glm_clust(@formula(excl_error ~ self),
                                 clean(df, [:excl_error], F64); clust = :hhea)

    μ_5a[6], r_5a[6] = glm_clust(@formula(incl_error ~ self),
                                 clean(df, [:incl_error], F64); clust = :hhea)

    ###########################
    # Analysis (w/ Stratum FEs)
    ###########################

    # Matches (get 144 obs instead of 159, but all else matches, so letting go.)
    μ_5b[1], r_5b[1] = mean(df[df.get, :logc]), reg(df[df.get, :],
                        @formula(logc ~ self + fe(kecagroup)), cluster(:hhea))

    # Matches
    μ_5b[2], r_5b[2] = mean(df_bm.logc), reg(df_bm,
                       @formula(logc ~ self + base_or_end + fe(kecagroup)),
                       cluster(:hhea))

    # Logistic Regressions with FEs
    insertcols!(df, :logc_ST => df.logc .* df.self)
    df = clean(df, [:self], F64)

    # Run conditional logit regressions, clustering at stratum level
    μ_5b[3], r_5b[3] = glm_clust(@formula(get ~ self + logc + logc_ST + kecagroup),
                df_drop(:get, clean(df, [:get,:logc,:logc_ST,:kecagroup], F64));
                group = :kecagroup, clust = :kecagroup)

    μ_5b[4], r_5b[4] = glm_clust(@formula(mistarget ~ self + kecagroup),
                    df_drop(:mistarget, clean(df, [:mistarget], F64));
                    group = :kecagroup, clust = :kecagroup)

    μ_5b[5], r_5b[5] = glm_clust(@formula(excl_error ~ self + kecagroup),
                    df_drop(:excl_error, clean(df, [:excl_error], F64));
                    group = :kecagroup, clust = :kecagroup)

    μ_5b[6], r_5b[6] = glm_clust(@formula(incl_error ~ self + kecagroup),
                    df_drop(:incl_error, clean(df, [:incl_error], F64));
                    group = :kecagroup, clust = :kecagroup)

    # Print output
    mystats = NamedTuple{(:comments, :means)}((repeat(["No"],  6), μ_5a))
    regtable(r_5a...; regressors = ["self", "logc", "logc & self"],
             renderSettings=latexOutput("output/tables/Table5_NoStratumFEs.tex"),
    		 custom_statistics = mystats, table_kwargs...)

     mystats = NamedTuple{(:comments, :means)}((repeat(["Yes"], 6), μ_5b))
     regtable(r_5b...; regressors = ["self", "logc", "logc_ST"],
              renderSettings=latexOutput("output/tables/Table5_StratumFEs.tex"),
              custom_statistics = mystats, table_kwargs...)

    return μ_5a, r_5a, μ_5b, r_5b
end

###########################
# Table 6 regressions
###########################
function table_6(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load baseline data
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption  => :logc, :closesubtreatment => :close,
                 :selftargeting => :self, :verypoor_povertyline1 => :verypoor])
    N  = size(df, 1)

    insertcols!(df, :base_or_end  => 0.0,
                    :logc_ST      => df.logc .* df.self,
                    :benefit      => categorical(df.getbenefit),
                    :benefit_hyp  => categorical(((df.pcexppred_noise .<
                                                   df.povertyline3_noise) .&
                                     .!((df.showup .== 0) .& (df.self .== 1)))),
                    :excl_err_hyp => Vector{Union{F64,Missing}}(missing, N),
                    :incl_err_hyp => Vector{Union{F64,Missing}}(missing, N))

    df[(df.verypoor .== 1) .& (df.benefit_hyp .== 1), :excl_err_hyp] .= 0.0
    df[(df.verypoor .== 1) .& (df.benefit_hyp .== 0), :excl_err_hyp] .= 1.0

    df[(df.verypoor .== 0) .& (df.benefit_hyp .== 0), :incl_err_hyp] .= 0.0
    df[(df.verypoor .== 0) .& (df.benefit_hyp .== 1), :incl_err_hyp] .= 1.0

    insertcols!(df, :mistarget_hyp => [ismissing(df.incl_err_hyp[i]) ?
                                       df.excl_err_hyp[i] :
                                       df.incl_err_hyp[i] for i=1:N])
    # Store output
    μ_6a, r_6a = Vector{F64}(undef, 5), Vector{FixedEffectModel}(undef, 5)
    μ_6b, r_6b = Vector{F64}(undef, 5), Vector{FixedEffectModel}(undef, 5)

    ####################################
    # Analysis (w/o Stratum FEs) - MATCH!
    ####################################

    μ_6a[1], r_6a[1] = mean(df[df.benefit_hyp .== 1, :logc]),
                        reg(df[df.benefit_hyp .== 1, :],
                            @formula(logc ~ self), cluster(:hhea))

    μ_6a[2], r_6a[2] = glm_clust(@formula(benefit_hyp ~ self + logc + logc_ST),
                        clean(df, [:logc, :benefit_hyp, :self, :logc_ST], F64);
                        clust = :kecagroup)

    μ_6a[3], r_6a[3] = glm_clust(@formula(mistarget_hyp ~ self),
                        clean(df, [:mistarget_hyp], F64); clust = :kecagroup)

    μ_6a[4], r_6a[4] = glm_clust(@formula(excl_err_hyp ~ self),
                        clean(df, [:excl_err_hyp], F64); clust = :kecagroup)

    μ_6a[5], r_6a[5] = glm_clust(@formula(incl_err_hyp ~ self),
                        clean(df, [:incl_err_hyp], F64); clust = :kecagroup)

    ###########################
    # Analysis (w/ Stratum FEs)
    ###########################
    # match, not quite SE
    μ_6b[1], r_6b[1] = mean(df[df.benefit_hyp .== 1, :logc]),
                        reg(df[df.benefit_hyp .== 1, :],
                            @formula(logc ~ self + fe(kecagroup)), cluster(:hhea))

    # approx match; N agrees, SE sort of differ
    μ_6b[2], r_6b[2] = glm_clust(@formula(benefit_hyp ~ self + logc + logc_ST +
                                          kecagroup),
                        df_drop(:benefit_hyp, clean(df, [:benefit_hyp, :logc,
                                :self, :logc_ST, :kecagroup], F64));
                                group = :kecagroup, clust = :kecagroup)

    μ_6b[3], r_6b[3] = glm_clust(@formula(mistarget_hyp ~ self + kecagroup),
                         df_drop(:mistarget_hyp, clean(df, [:mistarget_hyp,
                                 :self], F64));
                                 group = :kecagroup, clust = :kecagroup)

    # I get a WAY different value here. Does it have to do with the fact that I
    # drop missings before I determine
    # whether outcomes are all positives or all negatives?
    μ_6b[4], r_6b[4] = glm_clust(@formula(excl_err_hyp ~ self + kecagroup),
                                 df_drop(:excl_err_hyp,
                                         clean(df, [:excl_err_hyp], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_6b[5], r_6b[5] = glm_clust(@formula(incl_err_hyp ~ self + kecagroup),
                                 df_drop(:incl_err_hyp,
                                         clean(df, [:incl_err_hyp, :self], F64));
                                 group = :kecagroup, clust = :kecagroup)

    # Print output
    mystats = NamedTuple{(:comments, :means)}((repeat(["No"],  5), μ_6a))
    regtable(r_6a...; regressors = ["self", "logc", "logc_ST"],
             renderSettings=latexOutput("output/tables/Table6_NoStratumFEs.tex"),
    		 custom_statistics = mystats, table_kwargs...)

     mystats = NamedTuple{(:comments, :means)}((repeat(["Yes"], 5), μ_6b))
     regtable(r_6b...; regressors = ["self", "logc", "logc_ST"],
              renderSettings=latexOutput("output/tables/Table6_StratumFEs.tex"),
              custom_statistics = mystats, table_kwargs...)

    return μ_6a, r_6a, μ_6b, r_6b
end

###########################
# Table 7 regressions
###########################
function table_7(; table_kwargs::Dict{Symbol,Any} = table_kwargs)
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption  => :logc, :closesubtreatment => :close,
                 :selftargeting => :self])

    # Helper functions for speedy definition of quintiles
    quints = [quantile(clean(df,[:logc],F64).logc, p) for p in 0.0:0.2:1.0]
    quint(x, n) = ismissing(x) ? missing : (x > quints[n]) & (x <= quints[n+1])

    insertcols!(df, :logc_ST => df.logc .* df.self,
                    :inc1    => quint.(df.logc, 1),
                    :inc2    => quint.(df.logc, 2), :inc3 => quint.(df.logc, 3),
                    :inc4    => quint.(df.logc, 4), :inc5 => quint.(df.logc, 5))
    insertcols!(df, :close_logc => df.logc  .* df.close,
                    :closeinc1  => df.close .* df.inc1,
                    :closeinc2  => df.close .* df.inc2,
                    :closeinc3  => df.close .* df.inc3,
                    :closeinc4  => df.close .* df.inc4,
                    :closeinc5  => df.close .* df.inc5)

    # Drop non-selftargeting and those whom we don't have outcome data on
    df = df[df.self .== 1, :]
    df = clean(df, [:showup, :close], F64)

    # Store output
    μ_7,    r_7    = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)
    # R1: Logit
    μ_7[1], r_7[1] = glm_clust(@formula(showup ~ close), df; clust = :hhea)
    # R2: Logit
    μ_7[2], r_7[2] = glm_clust(@formula(showup ~ close + logc + close_logc),
                               clean(df, [:close_logc], F64); clust = :hhea)
    # R3: Logit
    μ_7[3], r_7[3] = glm_clust(@formula(showup ~ close + inc2 + inc3 + inc4 +
                        inc5 + closeinc2 + closeinc3 + closeinc4 + closeinc5),
                        clean(df, [:inc2, :inc3, :inc4, :inc5, :closeinc2,
                        :closeinc3, :closeinc4, :closeinc5], F64); clust=:hhea)

    # R4: Conditional Logit
    μ_7[4], r_7[4] = glm_clust(@formula(showup ~ close + kecagroup),
                               df_drop(:showup, df); group = :kecagroup,
                               clust = :kecagroup)
    # R5: Conditional Logit
    μ_7[5], r_7[5] = glm_clust(@formula(showup ~ close + logc + close_logc +
                                        kecagroup),
                        df_drop(:showup, clean(df, [:logc, :close_logc], F64));
                        group = :kecagroup, clust = :kecagroup)
    # R6: Conditional Logit
    μ_7[6], r_7[6] = glm_clust(@formula(showup ~ close + inc2 + inc3 + inc4 +
                                        inc5 + closeinc2 + closeinc3 +
                                        closeinc4 + closeinc5 + kecagroup),
                        df_drop(:showup, clean(df, [:inc2, :inc3, :inc4, :inc5,
                        :closeinc2, :closeinc3, :closeinc4, :closeinc5], F64));
                        group = :kecagroup, clust = :kecagroup)
    # Print output
    mystats = NamedTuple{(:comments, :means)}((["No","No","No","Yes","Yes","Yes"], μ_7))
    regtable(r_7...; renderSettings = latexOutput("output/tables/Table7.tex"),
             regressors = ["close", "logc", "close_logc", "inc2", "inc3", "inc4", "inc5",
                           "closeinc2", "closeinc3", "closeinc4", "closeinc5"],
    		 custom_statistics = mystats, table_kwargs...)
    return μ_7, r_7
end

###########################
# Table 8: Estimated Parameters from 2 stage GMM (415)
###########################
function table_8(; run_estimation = true, run_bootstrap = true,
                   overwrite_output = false, VERBOSE = false, f_tol = 1e-2)
    irate = 1.22
    δ_y   = 1 / irate

    have_est = isfile("output/MATLAB_est_d$(Int(round(δ_y * 100))).csv")
    have_bs  = isfile("output/MATLAB_bs_$(Int(round(δ_y * 100))).csv")
    if overwrite_output
        println("NOTE on Table 8: You have set overwrite_output = true. \n\n" *
                "This is going to overwrite your estimation/bootstrap output " *
                "files in your output/ directory. If you wish to intercept this, "*
                "there is still time! Slam CTRL-C! \n")
    else
        run_estimation = !have_est
        run_bootstrap  = !have_bs
        println("(Table 8) NOTE: You have set the keyword overwrite_output = " *
                "false.\n\nEven if `run_{estimation, bootstrap} = true`, these " *
                "flags will be ignored if the corresponding output file already " *
                "exists in your `output/ directory`. (Overwriting is set to false " *
                "by default so you don't do something you regret.) If you wish "  *
                "to proceed, re-run this command with overwrite_output = true, "  *
                "or rename/move the existing file(s) for safe-keeping. \n")
    end
    println("✨  For reference, you currently " *
                    (have_est ? "✅  DO " : " ⛔  DO NOT ") *
            "have an output estimation file, and " *
                    (have_bs  ? "✅  DO " : " ⛔  DO NOT ") *
            "have an output bootstrap file. ✨\n")
    if !(run_estimation | have_est) | !(run_bootstrap | have_bs)
        println("Since one or more of the above files does not exist, and you " *
                "aren't indicating intent to generate it, you're going to be " *
                "hitting an error building your LaTeX table in T - 1, 2, ...")
    end
    estimation_1(; run_estimation = run_estimation, run_bootstrap = run_bootstrap,
                   output_table = true, VERBOSE = VERBOSE, f_tol = f_tol)
end

"""
Corresponds to counterfactuals_1.m
"""
function counterfactuals_1(df::DataFrame; N_grid = 100, store_output = false)
    # Compute unobs. consumption (residual regressing log(obs cons) on PMT)
    df = insertcols!(df, :unob_c => residuals(reg(df, @formula(logc ~ pmt)), df))
    # Corresponds to "load data" step
    df = compute_quantiles(df)

    N, N_p = size(df, 1), 5 # No. of households, parameters to estimate
    irate = 1.22
    η_sd  = 0.275
    δ_mom = 0.0
    δ_y   = 1 / irate
    μ_con_true = df.reg_const2[1]
    μ_β_true   = df.reg_pmt2[1]
    λ_con_true = df.reg_nofe_const[1]
    λ_β_true   = df.reg_nofe_logcon[1]

    δ = δ_y * (1 + δ_y + δ_y^2 + δ_y^3 + δ_y^4 + δ_y^5)
    t = [-79681, 59715, 0.5049, 8.0448, -0.71673]

    #df_show_hat = DataFrame([:hhid => df.hhid, :c_q => df.c_q])

    # Column 2: Baseline estimate of show_hat
    insertcols!(df, :col2 =>
        showuphat(df, t, η_sd, δ, μ_con_true, μ_β_true, λ_con_true, λ_β_true;
                  N_grid = N_grid)[1])
    # Column 3: Half Standard deviation of epsilon
    t_3 = [t[1], t[2]/2, t[3:5]...]
    insertcols!(df, :col3 =>
        showuphat(df, t_3, η_sd, δ, μ_con_true, μ_β_true, λ_con_true, λ_β_true;
                  N_grid = N_grid)[1])
    # Column 4: No epsilon variance
    t_4 = [t[1], t[2]/1e10, t[3:5]...]
    insertcols!(df, :col4 =>
        showuphat(df, t_4, η_sd, δ, μ_con_true, μ_β_true, λ_con_true, λ_β_true;
                  N_grid = N_grid)[1])
    # Column 5: No differential travel cost
    df_5 = deepcopy(df)
    df_5.totcost_pc = df.totcost_smth_pc
    insertcols!(df, :col5 =>
        showuphat(df_5, t, η_sd, δ, μ_con_true, μ_β_true, λ_con_true, λ_β_true;
                  N_grid = N_grid)[1])

    # Column 6: (constant mu AND lambda)
    mean_mu = 0.0967742 # Mean benefit receipt conditional on applying
    λ_con_bel_cml = μ_con_true_cml = norminvcdf(mean_mu)
    λ_β_bel_cml   = μ_β_true_cml   = 0.
    df_6      = deepcopy(df)
    df_6.FE  .= 0.
    df_6.FE2 .= 0.
    t_6 = [t[1:3]..., λ_con_bel_cml, λ_β_bel_cml]

    insertcols!(df, :col6 => showuphat(df_6, t_6, η_sd, δ, μ_con_true_cml,
                                       μ_β_true_cml, λ_con_true, λ_β_true;
                                       N_grid = N_grid)[1])
    # Column 7: 3 extra km
    df_7 = deepcopy(df)
    df_7.totcost_pc = (1 .- df.close) .* df.totcost_3k_pc + df.close .* df.totcost_pc
    insertcols!(df, :col7 => showuphat(df_7, t, η_sd, δ, μ_con_true, μ_β_true,
                                       λ_con_true, λ_β_true; N_grid = N_grid)[1])
    # Column 8: 6 extra km
    df_8 = deepcopy(df)
    df_8.totcost_pc = (1 .- df.close) .* df.totcost_6k_pc + df.close .* df.totcost_pc
    insertcols!(df, :col8 => showuphat(df_8, t, η_sd, δ, μ_con_true, μ_β_true,
                                       λ_con_true, λ_β_true; N_grid = N_grid)[1])
    # Column 9: 3x waiting time
    df_9 = deepcopy(df)
    df_9.totcost_pc = df.totcost_pc + (1 .- df.close) .* (2 .* df.ave_waiting .*
                                                df.wagerate) ./ (df.hhsize .* 60)
    insertcols!(df, :col9 => showuphat(df_9, t, η_sd, δ, μ_con_true, μ_β_true,
                                       λ_con_true, λ_β_true; N_grid = N_grid)[1])
    # Column 10: 6x waiting time
    df_10 = deepcopy(df)
    df_10.totcost_pc = df.totcost_pc + (1 .- df.close) .* (5 .* df.ave_waiting .*
                                                df.wagerate) ./ (df.hhsize .* 60)
    insertcols!(df, :col10 => showuphat(df_10, t, η_sd, δ, μ_con_true, μ_β_true,
                                        λ_con_true, λ_β_true; N_grid = N_grid)[1])

    # Column 11-12 alpha=0 (all-unsophisticated) and alpha=1 (all sophisticated)
    t_11 = [t[1], t[2], 0.0, t[4], t[5]]
    insertcols!(df, :col11 => showuphat(df, t_11, η_sd, δ, μ_con_true, μ_β_true,
                                        λ_con_true, λ_β_true; N_grid = N_grid)[1])
    t_11[3] = 1.0
    insertcols!(df, :col12 => showuphat(df, t_11, η_sd, δ, μ_con_true, μ_β_true,
                                        λ_con_true, λ_β_true; N_grid = N_grid)[1])
    if store_output
        CSV.write("output/MATLAB_table9_showup.csv", df)
    end
    return df
end

###########################
# Table 9: Modeled effects of time and distance costs on show-up rates (416)
###########################
function table_9(; N_grid = 100, run_counterfactuals = true, bootstrap_se = true,
                 N_bs = 1000)
                 #panel_A = true, panel_B = true, panel_C = true)

    # Load + clean full dataset, to be merged on later
    df = DataFrame(load("input/matched_baseline.dta"))
    df = insertcols!(df, :logc => log.(df.consumption))
    df = rename!(df, [:closesubtreatment => :close1,
                      :verypoor_povertyline1 => :below])
    df = clean(df, [:hhid], Int64)
    df = clean(df, [:logc, :close1, :below], F64)

    # Define interactions for Panel B
    insertcols!(df, :close_logc  =>  df.logc            .*  df.close1,
                    :close_below =>  df.close1          .*  df.below,
                    :close_above =>  df.close1          .* (df.below .== 0.0),
                    :far_below   => (df.close1 .== 0.0) .*  df.below,
                    :far_above   => (df.close1 .== 0.0) .* (df.below .== 0.0))

    # Define outcomes and population cleaves
    outcomes = [:showup, :col2, :col3, :col4, :col5, :col6]
    groups   = [:far_above, :close_above, :far_below, :close_below]

    # Load + clean input dataset for simulation
    df_sim = CSV.read("input/MATLAB_Input.csv", DataFrame, header = true)
    df_sim = insertcols!(df_sim, :logc => log.(df_sim.consumption))
    df_sim = rename!(df_sim, [:closesubtreatment => :close, :consumption => :c,
                              :pmtscore => :pmt])
    df_sim = clean(df_sim, [:logc, :close, :getbenefit, :pmt, :distt, :c], F64)

    # Run scenarios if one has not done so already, or wishes to do so again!
    scen_file = "output/MATLAB_table9_showup.csv"
    df_show = if run_counterfactuals | !isfile(scen_file)
        counterfactuals_1(deepcopy(df_sim); N_grid = N_grid, store_output=true)
    else
        CSV.read(scen_file, DataFrame, header = true)
    end
    to_keep = Symbol.(filter(x->x ∉ names(df), names(df_show)))

    ### Panel A:
    function panel_A(df0::DataFrame, arg::Symbol)
        return coef(glm_clust(eval(Meta.parse(
                   "@formula($(string(arg)) ~ close + logc + close_logc)")),
                   clean(df0, [arg], F64); clust = :hhea)[2])
    end
    ### Panel B:
    function panel_B(df0::DataFrame)
        r_9b = [zeros(length(outcomes)) for i=1:length(groups)]
        for (i, g) in enumerate(groups)
            for (j, out) in enumerate(outcomes)
                df_tmp     = clean(df0, [g, out], F64)
                r_9b[i][j] = 100 * sum(df_tmp[:,out] .* df_tmp[:,g]) / sum(df_tmp[:,g])
            end
        end
        return r_9b
    end
    ### Panel C:
    function panel_C(r_B::Vector)
        poor_rich_far   = r_B[3] ./ r_B[1]
        poor_rich_close = r_B[4] ./ r_B[2]
        return [poor_rich_far, poor_rich_close, poor_rich_far .- poor_rich_close]
    end

    df_m = innerjoin(df, df_show[:, [to_keep..., :hhid]], on = :hhid)
    r_9a = [panel_A(df_m, o) for o in outcomes]
    r_9b = panel_B(df_m)
    r_9c = panel_C(r_9b)

    @save "output/table9_panel_A.jld2" r_9a
    @save "output/table9_panel_B.jld2" r_9b
    @save "output/table9_panel_C.jld2" r_9c

    ### Bootstrap Panel A and C's standard errors
    if bootstrap_se
        # Draw random clusters, then include all obs. w/in cluster into index set
        c_set        = unique(df_sim[:,:hhea])
        bs_clust_idx = [sample(c_set, length(c_set); replace=true) for i=1:N_bs]
        bs_idx       = [vcat([findall(isequal(c), df_sim[:,:hhea])
                                for c in bs_clust]...) for bs_clust in bs_clust_idx]
        bs_A_out  = Dict{Symbol,Matrix{F64}}([o =>
                    Matrix{F64}(undef, N_bs, 3) for o in outcomes])
        bs_C_out  = deepcopy(bs_A_out)
        obs_count = 0

        # Run bootstrap!
        for i=1:N_bs
            println("(Table 9) Bootstrap iter. $i")
            # Grab subsample, note # of observations in subsample
            idx        = bs_idx[i]
            obs_count += length(idx)
            # Run counterfactual estimates on subsample
            df_i   = counterfactuals_1(deepcopy(df_sim)[idx,:])
            df_tmp = innerjoin(df, df_i[:, [to_keep..., :hhid]], on = :hhid)
            rc_i   = panel_C(panel_B(df_tmp))
            for (j,o) in enumerate(outcomes)
                bs_A_out[o][i,:] = panel_A(df_tmp, o)[2:end] # Drop constant
                bs_C_out[o][i,:] = [rc[j] for rc in rc_i]
            end
        end
        # Compute standard errors + save output
        bs_A_se = Dict{Symbol,Vector{F64}}([o => vec(std(bs_A_out[o], dims=1)) for o in outcomes])
        bs_C_se = Dict{Symbol,Vector{F64}}([o => vec(std(bs_C_out[o], dims=1)) for o in outcomes])
        @save "output/table9_panel_A_SE.jld2" bs_A_se
        @save "output/table9_panel_C_SE.jld2" bs_C_se
        @save "output/table9_obs_count.jld2" obs_count
    end
    bs_A_se   = load("output/table9_panel_A_SE.jld2")["bs_A_se"]
    bs_C_se   = load("output/table9_panel_C_SE.jld2")["bs_C_se"]
    obs_count = load("output/table9_obs_count.jld2")["obs_count"]
    r_9a = [panel_A(df_m,o) for o in outcomes]
    r_9b = panel_B(df_m)
    r_9c = panel_C(r_9b)

    # Directly write LaTeX table
    io = open("output/tables/Table9.tex", "w")
    write(io, "\\begin{tabular}{lcccccc}\\toprule" *
              "& (Experimental) & \\multicolumn{5}{c}{(See Models Above)} \\\\ \\cmidrule(lr){3-7} \n" *
              "& (1) & (2) & (3) & (4) & (5) & (6) \\\\ " *
              "\\midrule\n & \\multicolumn{6}{c}{\\centering A. Logistic " *
              "Regressions}\\\\\\cmidrule(lr){2-7} \n")
    # Print Panel A
    for (i, sym) in enumerate([labels["close"], labels["logc"], labels["close_logc"]])
        @printf(io, "%s & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f \\\\",
                    vcat(sym, [r_9a[j][i+1] for j=1:length(outcomes)])...)
        @printf(io, " & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) \\\\",
                    [bs_A_se[o][i] for o in outcomes]...)
    end
    write(io, "Observations & $(size(df_m,1)) " * repeat("& $obs_count", 5) * "\\\\")
    write(io, "\\textit{p-value} & &")

    # Compute p-values here
    n1 = size(df_m, 1)
    n2 = obs_count
    @printf(io, " %0.3f & %0.3f & %0.3f & %0.3f & %0.3f \\\\",
        cdf.(Normal(), [abs(r_9a[1][4]-r_9a[j][4]) / sqrt((bs_A_se[:showup][3]^2 +
                        bs_A_se[outcomes[j]][3]^2)/2) *
                        sqrt(1/n1 + 1/n2) for j=2:length(outcomes)])...)
    # Panel B
    write(io, "\\midrule\n & \\multicolumn{6}{c}{\\centering B. Show " *
    "Up Rates}\\\\\\cmidrule(lr){2-7} \n")
    for (i, g) in enumerate(groups)
        @printf(io, "%s & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f \\\\",
                    vcat(labels[string(g)], r_9b[i])...)
    end

    # Print Panel C
    write(io, "\\midrule\n & \\multicolumn{6}{c}{\\centering C. Show " *
    "Up Rate Ratios}\\\\\\cmidrule(lr){2-7} \n")
    for (i, sym) in enumerate(["Poor to rich ratio, far", "Poor to rich ratio, close",
                               "Difference of ratios"])
        @printf(io, "%s & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f \\\\",
                    vcat(sym, r_9c[i])...)
        @printf(io, " & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) & (%0.3f) \\\\",
                    [bs_C_se[o][i] for o in outcomes]...)
    end
    # write(io, "\\textit{p-value} & &")
    # Compute p-values here
    # @printf(io, " %0.3f & %0.3f & %0.3f & %0.3f & %0.3f \\\\",
    #     cdf.(Normal(), [abs(r_9c[3][1]-r_9c[3][j]) / sqrt((bs_C_se[:showup][3]^2 +
    #                     bs_C_se[outcomes[j]][3]^2)/2) *
    #                     sqrt(1/n1 + 1/n2) for j=2:length(outcomes)])...)

    write(io, "\\bottomrule\\end{tabular}")
    close(io)
    return bs_A_se
end

###########################
# Table 10: Impact of alternative testing approaches on poverty gap (424)
###########################
