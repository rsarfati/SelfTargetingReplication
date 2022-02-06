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
    write(io,                   "That Received\\\\Benefits (\\%)}\\\\\\hline\\\\")

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
    rename!(df, [:logconsumption => :logc, :closesubtreatment => :close])

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
    μ[2], r[2] = glm_clust(f, df[df.pcexppred .<  df.povertyline3,:]; clust = :hhea)
    μ[3], r[3] = glm_clust(f, df[df.pcexppred .>= df.povertyline3,:]; clust = :hhea)

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
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption => :logc, :closesubtreatment => :close])
    insertcols!(df, :base_or_end => 0.0,
                    :benefit     => categorical(df.getbenefit))

    # Load midline data
    df_m = clean(DataFrame(load("input/matched_midline.dta")), [:flag_newHH], F64)
    df_m = df_m[df_m.flag_newHH .== 1.0, :]
    rename!(df_m, [:logconsumption => :logc, :closesubtreatment => :close])
    df_m = convert_types(df_m, [:logc, :showup] .=> F64)
    insertcols!(df_m, :base_or_end => 1.0)

    # Includes baseline and midline data
    df_bm = vcat(df, df_m, cols = :union)
    df_bm = clean(df_bm[df_bm.getbenefit .== 1, :], [:logc], F64)

    ####################################
    # Analysis (w/o Stratum FEs) - MATCH
    ####################################

    # Store output
    μ_5a, r_5a = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)
    μ_5b, r_5b = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)

    μ_5a[1], r_5a[1] = mean(df[df.getbenefit .== 1, :logc]),
                        reg(df[df.getbenefit .== 1, :], @formula(logc ~ selftargeting), cluster(:hhea))

    μ_5a[2], r_5a[2] = mean(df_bm.logc),
                        reg(df_bm, @formula(logc ~ selftargeting + base_or_end), cluster(:hhea))

    μ_5a[3], r_5a[3] = glm_clust(@formula(getbenefit ~ selftargeting + logc + logc * selftargeting),
                                 clean(df, [:logc, :getbenefit, :selftargeting], F64); clust = :hhea)

    μ_5a[4], r_5a[4] = glm_clust(@formula(mistarget ~ selftargeting),
                                 clean(df, [:mistarget], F64); clust = :hhea)

    μ_5a[5], r_5a[5] = glm_clust(@formula(excl_error ~ selftargeting),
                                 clean(df, [:excl_error], F64); clust = :hhea)

    μ_5a[6], r_5a[6] = glm_clust(@formula(incl_error ~ selftargeting),
                                 clean(df, [:incl_error], F64); clust = :hhea)

    ###########################
    # Analysis (w/ Stratum FEs)
    ###########################

    # Matches (get 144 obs instead of 159, but all else matches, so letting go.)
    μ_5b[1], r_5b[1] = mean(df[df.getbenefit .== 1, :logc]),
                        reg(df[df.getbenefit .== 1, :],
                        @formula(logc ~ selftargeting + fe(kecagroup)), cluster(:hhea))

    # Matches
    μ_5b[2], r_5b[2] = mean(df_bm.logc), reg(df_bm,
                       @formula(logc ~ selftargeting + base_or_end + fe(kecagroup)), cluster(:hhea))

    # Logistic Regressions with FEs
    insertcols!(df, :logc_ST => df.logc .* df.selftargeting)

    # Manually drop groups which fail positivity!
    G_drop(d::DataFrame,
           v::Symbol) = findall(g->(prod(d[d.kecagroup .== g, v] .== 0.0) ||
                                    prod(d[d.kecagroup .== g, v] .== 1.0)),
                                unique(d.kecagroup))
    df_drop(v::Symbol, d::DataFrame) = d[d.kecagroup .∉ (G_drop(d, v),), :]

    # Run conditional logit regressions, clustering at stratum level
    μ_5b[3], r_5b[3] = glm_clust(@formula(getbenefit ~ selftargeting + logc + logc_ST + kecagroup),
                                 df_drop(:getbenefit, clean(df, [:getbenefit, :logc, :selftargeting,
                                                                 :logc_ST, :kecagroup], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_5b[4], r_5b[4] = glm_clust(@formula(mistarget ~ selftargeting + kecagroup),
                                 df_drop(:mistarget, clean(df, [:mistarget, :selftargeting], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_5b[5], r_5b[5] = glm_clust(@formula(excl_error ~ selftargeting + kecagroup),
                                 df_drop(:excl_error, clean(df, [:excl_error, :selftargeting], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_5b[6], r_5b[6] = glm_clust(@formula(incl_error ~ selftargeting + kecagroup),
                                 df_drop(:incl_error, clean(df, [:incl_error, :selftargeting], F64));
                                 group = :kecagroup, clust = :kecagroup)

    # Print output
    mystats = NamedTuple{(:comments, :means)}((repeat(["No"],  6), μ_5a))
    regtable(r_5a...; renderSettings = latexOutput("output/tables/Table5_NoStratumFEs.tex"),
             regressors = ["selftargeting", "logc", "logc & selftargeting"],
    		 custom_statistics = mystats, table_kwargs...)

     mystats = NamedTuple{(:comments, :means)}((repeat(["Yes"], 6), μ_5b))
     regtable(r_5b...; renderSettings = latexOutput("output/tables/Table5_StratumFEs.tex"),
     		  regressors = ["selftargeting", "logc", "logc_ST"],
              custom_statistics = mystats, table_kwargs...)

    return μ_5a, r_5a, μ_5b, r_5b
end

###########################
# Table 6 regressions
###########################
function table_6(; table_kwargs::Dict{Symbol,Any} = table_kwargs)

    # Load baseline data
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption  => :logc, :closesubtreatment => :close])
    N  = size(df, 1)

    insertcols!(df, :base_or_end  => 0.0,
                    :logc_ST      => df.logc .* df.selftargeting,
                    :benefit      => categorical(df.getbenefit),
                    :benefit_hyp  => categorical(((df.pcexppred_noise .< df.povertyline3_noise) .&
                                     .!((df.showup .== 0) .& (df.selftargeting .== 1)))),
                    :excl_err_hyp => Vector{Union{F64,Missing}}(missing, N),
                    :incl_err_hyp => Vector{Union{F64,Missing}}(missing, N))

    df[(df.verypoor_povertyline1 .== 1) .& (df.benefit_hyp .== 1), :excl_err_hyp] .= 0.0
    df[(df.verypoor_povertyline1 .== 1) .& (df.benefit_hyp .== 0), :excl_err_hyp] .= 1.0

    df[(df.verypoor_povertyline1 .== 0) .& (df.benefit_hyp .== 0), :incl_err_hyp] .= 0.0
    df[(df.verypoor_povertyline1 .== 0) .& (df.benefit_hyp .== 1), :incl_err_hyp] .= 1.0

    insertcols!(df, :mistarget_hyp => [ismissing(df.incl_err_hyp[i]) ? df.excl_err_hyp[i] :
                                       df.incl_err_hyp[i] for i=1:N])

    # Store output
    μ_6a, r_6a = Vector{F64}(undef, 5), Vector{FixedEffectModel}(undef, 5)
    μ_6b, r_6b = Vector{F64}(undef, 5), Vector{FixedEffectModel}(undef, 5)

    ####################################
    # Analysis (w/o Stratum FEs) - MATCH!
    ####################################

    μ_6a[1], r_6a[1] = mean(df[df.benefit_hyp .== 1, :logc]),
                        reg(df[df.benefit_hyp .== 1, :],
                            @formula(logc ~ selftargeting), cluster(:hhea))

    μ_6a[2], r_6a[2] = glm_clust(@formula(benefit_hyp ~ selftargeting + logc + logc_ST),
                                 clean(df, [:logc, :benefit_hyp, :selftargeting, :logc_ST], F64);
                                 clust = :kecagroup)

    μ_6a[3], r_6a[3] = glm_clust(@formula(mistarget_hyp ~ selftargeting),
                                 clean(df, [:mistarget_hyp], F64); clust = :kecagroup)

    μ_6a[4], r_6a[4] = glm_clust(@formula(excl_err_hyp ~ selftargeting),
                                 clean(df, [:excl_err_hyp], F64); clust = :kecagroup)

    μ_6a[5], r_6a[5] = glm_clust(@formula(incl_err_hyp ~ selftargeting),
                                 clean(df, [:incl_err_hyp], F64); clust = :kecagroup)

    ###########################
    # Analysis (w/ Stratum FEs)
    ###########################
    # match, not quite SE
    μ_6b[1], r_6b[1] = mean(df[df.benefit_hyp .== 1, :logc]),
                        reg(df[df.benefit_hyp .== 1, :],
                            @formula(logc ~ selftargeting + fe(kecagroup)), cluster(:hhea))

    # Manually drop groups which fail positivity!
    G_drop(v::Symbol, d::DataFrame) = findall(g->(prod(d[d.kecagroup .== g, v] .== 0.0) ||
                                                  prod(d[d.kecagroup .== g, v] .== 1.0)),
                                              unique(d.kecagroup))
    df_drop(v::Symbol, d::DataFrame) = d[d.kecagroup .∉ (G_drop(v, d),), :]

    # approx match; N agrees, SE sort of differ
    μ_6b[2], r_6b[2] = glm_clust(@formula(benefit_hyp ~ selftargeting + logc + logc_ST + kecagroup),
                                 df_drop(:benefit_hyp, clean(df, [:benefit_hyp, :logc, :selftargeting,
                                                                 :logc_ST, :kecagroup], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_6b[3], r_6b[3] = glm_clust(@formula(mistarget_hyp ~ selftargeting + kecagroup),
                                 df_drop(:mistarget_hyp, clean(df, [:mistarget_hyp, :selftargeting], F64));
                                 group = :kecagroup, clust = :kecagroup)

    # I get a WAY different value here. Does it have to do with the fact that I drop missings before I determine
    # whether outcomes are all positives or all negatives?
    μ_6b[4], r_6b[4] = glm_clust(@formula(excl_err_hyp ~ selftargeting + kecagroup),
                                 df_drop(:excl_err_hyp, clean(df, [:excl_err_hyp], F64));
                                 group = :kecagroup, clust = :kecagroup)

    μ_6b[5], r_6b[5] = glm_clust(@formula(incl_err_hyp ~ selftargeting + kecagroup),
                                 df_drop(:incl_err_hyp, clean(df, [:incl_err_hyp, :selftargeting], F64));
                                 group = :kecagroup, clust = :kecagroup)

    # Print output
    mystats = NamedTuple{(:comments, :means)}((repeat(["No"],  5), μ_6a))
    regtable(r_6a...; renderSettings = latexOutput("output/tables/Table6_NoStratumFEs.tex"),
             regressors = ["selftargeting", "logc", "logc_ST"],
    		 custom_statistics = mystats, table_kwargs...)

     mystats = NamedTuple{(:comments, :means)}((repeat(["Yes"], 5), μ_6b))
     regtable(r_6b...; renderSettings = latexOutput("output/tables/Table6_StratumFEs.tex"),
     		  regressors = ["selftargeting", "logc", "logc_ST"],
              custom_statistics = mystats, table_kwargs...)

    return μ_6a, r_6a, μ_6b, r_6b
end

###########################
# Table 7 regressions
###########################
function table_7(; table_kwargs::Dict{Symbol,Any} = table_kwargs)
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption  => :logc, :closesubtreatment => :close])
    N  = size(df, 1)

    quints = [quantile(clean(df, [:logc], F64).logc, p) for p in [0.0, .2, .4, .6, .8, 1.0]]
    quint(x, n::Int64) = ismissing(x) ? missing : (x > quints[n]) & (x <= quints[n+1])

    insertcols!(df, :close_logc  => df.logc .* df.close,
                    :logc_ST     => df.logc .* df.selftargeting,
                    :inc1 => quint.(df.logc, 1),
                    :inc2 => quint.(df.logc, 2), :inc3 => quint.(df.logc, 3),
                    :inc4 => quint.(df.logc, 4), :inc5 => quint.(df.logc, 5))
    insertcols!(df, :closeinc1   => df.close .* df.inc1,
                    :closeinc2   => df.close .* df.inc2,
                    :closeinc3   => df.close .* df.inc3,
                    :closeinc4   => df.close .* df.inc4,
                    :closeinc5   => df.close .* df.inc5)

    df = df[df.selftargeting .== 1, :]

    # Store output
    μ_7,    r_7    = Vector{F64}(undef, 6), Vector{FixedEffectModel}(undef, 6)
    # R1: Logit
    μ_7[1], r_7[1] = glm_clust(@formula(showup ~ close),
                               clean(df, [:showup, :close], F64); clust = :hhea)
    # R2: Logit
    μ_7[2], r_7[2] = glm_clust(@formula(showup ~ close + logc + close_logc),
                               clean(df, [:showup, :close, :close_logc], F64); clust = :hhea)
    # R3: Logit
    μ_7[3], r_7[3] = glm_clust(@formula(showup ~ close + inc2 + inc3 + inc4 + inc5 +
                                        closeinc2 + closeinc3 + closeinc4 + closeinc5),
                               clean(df, [:showup, :close, :inc2, :inc3, :inc4, :inc5, :closeinc2,
                                          :closeinc3, :closeinc4, :closeinc5], F64); clust = :hhea)

    # Manually drop groups which fail positivity!
    G_drop(v::Symbol, d::DataFrame) = findall(g->(prod(d[d.kecagroup .== g, v] .== 0.0) ||
                                                  prod(d[d.kecagroup .== g, v] .== 1.0)),
                                            unique(d.kecagroup))
    df_drop(v::Symbol, d::DataFrame) = d[d.kecagroup .∉ (G_drop(v, d),), :]

    # R4: Conditional Logit
    μ_7[4], r_7[4] = glm_clust(@formula(showup ~ close + kecagroup),
                               df_drop(:showup, clean(df, [:showup, :close], F64));
                               group = :kecagroup, clust = :kecagroup)
    # R5: Conditional Logit
    μ_7[5], r_7[5] = glm_clust(@formula(showup ~ close + logc + close_logc + kecagroup),
                               df_drop(:showup, clean(df, [:showup, :close, :logc, :close_logc], F64));
                               group = :kecagroup, clust = :kecagroup)
    # R6: Conditional Logit
    μ_7[6], r_7[6] = glm_clust(@formula(showup ~ close + inc2 + inc3 + inc4 + inc5 +
                                        closeinc2 + closeinc3 + closeinc4 + closeinc5 + kecagroup),
                               df_drop(:showup, clean(df, [:showup, :close, :inc2, :inc3, :inc4, :inc5,
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
function table_8()
    irate = 1.22
    δ_y   = 1 / irate
    run_estimation = !isfile("output/MATLAB_est_d$(round(δ_y * 100)).csv")
    run_bootstrap  = !isfile("output/MATLAB_bs_$(Int(round(δ_y * 100))).csv")
    estimation_1(; run_estimation = run_estimation, run_bootstrap = run_bootstrap,
                   output_table = true)
end

###########################
# Table 9: Modeled effects of time and distance costs on show-up rates (416)
###########################
function table_9()
    irate = 1.22
    δ_y   = 1 / irate
    run_estimation = !isfile("output/MATLAB_est_d$(round(δ_y * 100)).csv")
    run_bootstrap  = !isfile("output/MATLAB_bs_$(Int(round(δ_y * 100))).csv")
    estimation_1(; run_estimation = run_estimation, run_bootstrap = run_bootstrap,
                   output_table = true)
end
# ******** Table 9 (smallsample==0)/Appendix Table C19 (smallsample==1)
#
# ******** Data set-up, interactions

# /*if(`smallsample'==0){
# 	rename showup_hat_alpha0 showup_hat_a0
# 	rename showup_hat_alpha1 showup_hat_a1
# }*/
# keep closesubtreatment consumption logc showup showup_hat showup_hat_halfsd ///
#      showup_hat_noeps showup_hat_smthc showup_hat_cml hhid hhea *poverty*
#
# cap drop *sig?
# rename closesubtreatment close
#
# * gen RHS vars
# gen logc=log(consumption)
# gen close_logc = close*logc
# la var close "Close"
# la var logc "Log PCE"
# la var close_logc "Close * Log PCE"
# g closepoor = close*verypoor_povertyline1
# g poor = verypoor_povertyline1
# g showup_b = showup
#
#
# ******** Row names for table
# so hhid
# g col_names = "Close" in 1
# replace col_names = "Log per capita expenditure" in 3
# replace col_names = "Close * Log per capita expenditure" in 5
# replace col_names = "N" in 7
# replace col_names = "P-value" in 8
# replace col_names = "Above poverty line, far" in 10
# replace col_names = "Above poverty line, close" in 11
# replace col_names = "Below poverty line, far" in 12
# replace col_names = "Below poverty line, close" in 13
# replace col_names = "Poor to rich ratio, far" in 15
# replace col_names = "Poor to rich ratio, close" in 17
# replace col_names = "Difference of ratios" in 19
# replace col_names = "P-value" in 20
#
#
#
#
#
# ******** Panel A (Logistic Regressions) and Panel C (Show-Up Rate Ratios) for Column 1a, empirical results
# sort hhid
# * Show-up
# g col_showup_a = .
# g sig_showup_a = .
# logit showup close logc close_logc, cluster(hhea)
# 	matrix b = e(b)
# 	replace col_showup_a = b[1,1] in 1
# 	replace col_showup_a = b[1,2] in 3
# 	replace col_showup_a = b[1,3] in 5
# 	replace col_showup_a = _se[close] in 2
# 	replace col_showup_a = _se[logc] in 4
# 	replace col_showup_a = _se[close_logc] in 6
# 	replace col_showup_a = `e(N)' in 7
# 	local Z1 = _b[close]/_se[close]
# 	local Z3 = _b[logc]/_se[logc]
# 	local Z5 = _b[close_logc]/_se[close_logc]
# 	forv x = 1(2)5 {
# 		replace sig_showup_a = 2*(1-normal(abs(`Z`x''))) in `x'
# 		}
# reg showup poor close closepoor, cluster(hhea)
# 	** Ratio of poor to rich showup rate in far treatment
# 	nlcom ((_b[_cons] + _b[poor]) / _b[_cons])
# 	matrix bb = r(b)
# 	replace col_showup_a = bb[1,1] in 15
# 	matrix vv = r(V)
# 	replace col_showup_a = (vv[1,1])^(1/2) in 16
# 	** Ratio of poor to rich showup rate in close treatment
# 	nlcom ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 	matrix bb = r(b)
# 	replace col_showup_a = bb[1,1] in 17
# 	matrix vv = r(V)
# 	replace col_showup_a = (vv[1,1])^(1/2) in 18
# 	**Difference in ratios
# 	nlcom  ((_b[_cons] + _b[poor]) / _b[_cons]) - ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 	matrix bb = r(b)
# 	replace col_showup_a = bb[1,1] in 19
# 	matrix vv = r(V)
# 	replace col_showup_a = (vv[1,1])^(1/2) in 20
# g col_showup_b = col_showup_a		// standard errors will be replaced
# g sig_showup_b = .
#
#
# ******** Expand data for simulation
#
# sort hhid
# g id = _n
# compress
# expand `expansion'
# bys id: g subid = _n
#
# ******** Estimates, Panel A & C, for 6 simulations
#
#
# foreach v of varlist showup_hat* {
#
# ***Prepare
# 	g col_`v' = .
# 	g sig_`v' = .
# 	cap g `v'_prop = round(`expansion'*`v')
# 	cap g `v'_sim = (subid<=`v'_prop) if `v'_prop<.
#
# ***Logits, Panel A
# 	if(`smallsample'==0){
# 		version 9: logit `v'_sim close logc close_logc, cluster(hhea)
# 	}
# 	else{ // smallsample==1
# 		version 9: logit `v'_sim close logc close_logc, cluster(hhea)
# 	}
# 		est sto `v'_sim
# 		matrix b = e(b)
# 		replace col_`v' = b[1,1] if id==1
# 		replace col_`v' = b[1,2] if id==3
# 		replace col_`v' = b[1,3] if id==5
# 		replace col_`v' = `e(N)' if id==7
#
# ****Ratios from panel C
# 	reg `v'_sim poor close closepoor, cluster(hhea)
# 	**Ratio poor to rich in far
# 	nlcom ((_b[_cons] + _b[poor]) / _b[_cons])
# 	matrix bb = r(b)
# 	replace col_`v' = bb[1,1] if id==15
# 	** Ratio of poor to rich showup rate in close treatment
# 	nlcom ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 	matrix bb = r(b)
# 	replace col_`v' = bb[1,1] if id==17
# 	**Difference in ratios
# 	nlcom  ((_b[_cons] + _b[poor]) / _b[_cons]) - ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 	matrix bb = r(b)
# 	replace col_`v' = bb[1,1] if id==19
#
# drop `v'_sim `v'_prop
# }
#
#
#
#
# ******** Back to one observation
#
# pause off
# bys hhid: keep if _n==1
# isid hhid
#
# ******** Panel B for all columns
#
# g showup_a = showup
# sort verypoor_povertyline1 close
# egen pline_close = group(verypoor_povertyline1 close)
# label define pline_close 1 "Above Poverty Line, Far" 2 "Above Poverty Line, Close" 3 "Below Poverty Line, Far" 4 "Below Poverty Line, Close", replace
# la val pline_close pline_close
# sort hhid
# foreach v of varlist showup_a showup_b showup_hat* {
# forv x = 1/4 {
# summ `v' if pline_close==`x'
# local y = `x'+9
# replace col_`v' = `r(mean)'*100 in `y'
# }
# }
#
#
#
#
# ******** Manual BS standard errors for simulations and empirical, 1b
#
# pause off
#
# qui foreach v of varlist showup_b showup_hat* {
# * Set up
# g a`v'_bclose = .
# g a`v'_blogc = .
# g a`v'_bcloselog = .
# g a`v'_ratfarSE = .
# g a`v'_ratcloseSE = .
# g a`v'_diffSE = .
#
# * Logits, panel A
# local a = 0
# forv x = 1/`bootstrapruns' {
# 	local a = `a' + 1
# 	cap drop `v'_sim
# 	preserve
# 		bsample, cluster(hhea)
# 		sort hhid
# 		g rand = uniform()
# 		g `v'_sim = (rand < `v')
# 		noi di "`a' -- `v', logit SE"
# 		version 10: logit `v'_sim close logc close_logc, cluster(hhea)
# 		matrix b = e(b)
#
# 	restore
# 	sort hhid
# 	replace a`v'_bclose = b[1,1] in `a'
# 	replace a`v'_blogc = b[1,2] in `a'
# 	replace a`v'_bcloselog = b[1,3] in `a'
# 	}
# sort hhid
# summ a`v'_bclose
# replace col_`v' = `r(sd)' in 2
# if `r(mean)' < 0 {
# 	count if a`v'_bclose > 0 & a`v'_bclose < .
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 1
# 	}
# else {
# 	count if a`v'_bclose < 0
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 1
# 	}
# summ a`v'_blogc
# replace col_`v' = `r(sd)' in 4
# if `r(mean)' < 0 {
# 	count if a`v'_blogc > 0 & a`v'_blogc < .
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 3
# 	}
# else {
# 	count if a`v'_blogc < 0
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 3
# 	}
# summ a`v'_bcloselog
# replace col_`v' = `r(sd)' in 6
# if `r(mean)' < 0 {
# 	count if a`v'_bcloselog > 0 & a`v'_bcloselog < .
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 5
# 	}
# else {
# 	count if a`v'_bcloselog < 0
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 5
# 	}
#
#
# * Ratios, panel C
# local a = 0
# forv x = 1/`bootstrapruns' {
# 	local a = `a' + 1
# 	cap drop `v'_sim
# 	preserve
# 		bsample, cluster(hhea)
# 		sort hhid
# 		g rand = uniform()
# 		g `v'_sim = (rand < `v')
# 		noi di "`a' -- `v', ratios SE"
# 		reg `v'_sim poor close closepoor, cluster(hhea)
# 		local bb1 = ((_b[_cons] + _b[poor]) / _b[_cons])
# 		local bb2 = ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 		local bb3 =  ((_b[_cons] + _b[poor]) / _b[_cons]) - ((_b[_cons] + _b[poor] + _b[close] + _b[closepoor]) / (_b[_cons] + _b[close]))
# 	restore
# 	sort hhid
# 	pause
# 	replace a`v'_ratfarSE = `bb1' in `a'
# 	replace a`v'_ratcloseSE = `bb2' in `a'
# 	replace a`v'_diffSE = `bb3' in `a'
# 	}
# sort hhid
# summ a`v'_ratfarSE
# replace col_`v' = `r(sd)' in 16
# summ a`v'_ratcloseSE
# replace col_`v'  = `r(sd)' in 18
# summ a`v'_diffSE
# replace col_`v' = `r(sd)' in 20
# if `r(mean)' < 0 {
# 	count if a`v'_diffSE > 0 & a`v'_diffSE!=.
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 19
# 	}
# else {
# 	count if a`v'_diffSE < 0
# 	replace sig_`v' = `r(N)'*2/`bootstrapruns' in 19
# 	}
# }
#
#
# ******** P-values
# foreach v of varlist showup_hat* {
# local a = (col_`v'[5] - col_showup_b[5])/(((col_`v'[6])^2 + (col_showup_b[6])^2)^.5)
# replace col_`v' = 2*(1-normal(abs(`a'))) in 8
# local a = (col_`v'[19] - col_showup_b[19])/(((col_`v'[20])^2 + (col_showup_b[20])^2)^.5)
# replace col_`v' = 2*(1-normal(abs(`a'))) in 21
# }
#
#
# ******** Asterisks & rounding
#
#  foreach v of varlist showup_a showup_b showup_hat* {
# replace col_`v' = round(col_`v',.001)
# tostring col_`v', replace force format(%11.3f)
# replace col_`v' = col_`v' + "*" if sig_`v' < .1
# replace col_`v' = col_`v' + "*" if sig_`v' < .05
# replace col_`v' = col_`v' + "*" if sig_`v' < .01
# foreach n in 2 4 6 16 18 20 {
# 	replace col_`v' = "(" + col_`v' + ")" in `n'
# 	}
# }
#
#
# ******** Save
#
# 	if(`smallsample'==0){
# 		outsheet col_* in 1/21 using "tables\Table 9.csv", replace comma
# 	}
# 	else{
# 		outsheet col_* in 1/21 using "tables\Online Appendix C19.csv", replace comma
# 	}
#
# }

###########################
# Table 10: Impact of alternative testing approaches on poverty gap (424)
###########################
