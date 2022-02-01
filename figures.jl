###########################
# Figure 1
###########################
function figure_1(; labels::Dict{String,String} = labels, bins::Int64 = 0)

    # Load Data
    df = DataFrame(load("input/matched_baseline.dta"))
    rename!(df, [:logconsumption  => :logc])

    bins = if bins == 0; size(df, 1) else bins end

    # Make group variable (urban x kabgroup)
    insertcols!(df, :kaburban =>
        categorical(string.(collect(zip(df.urban, df.kabgroup))), compress = true))
    all_groups = sort(unique(df.kaburban))

    # TODO: Drop groups 2 and 4 due to insufficient variation
    df = df[(df.kaburban .!= all_groups[2]) .& (df.kaburban .!= all_groups[4]), :]
    deleteat!(all_groups, [2, 4])

    # Clean data
    df = clean(df[df.showup .== 1, :], [:getbenefit, :logc, :kaburban, :PMTSCORE],
               out_t = Dict([:getbenefit, :logc, :PMTSCORE] .=> F64))

    Φ(x::Float64) = cdf(Normal(), x)

    _, r_A = glm_clust(@formula(getbenefit ~ logc + kaburban), df;
                       link = ProbitLink(), group = :kaburban)

    insertcols!(df, :lambda => Φ.(r_A.coef[1] .+ df.logc .* r_A.coef[2] .+
                [(df.kaburban[i] == all_groups[1]) ? 0.0 :
                 r_A.coef[1 + findfirst(isequal(df.kaburban[i]), all_groups)]
                 for i=1:size(df,1)]))

    _, r_B = glm_clust(@formula(getbenefit ~ PMTSCORE + kaburban), df;
                       link = ProbitLink(), group = :kaburban)

    insertcols!(df, :mu_pmt => Φ.(r_B.coef[1] .+ df.PMTSCORE .* r_B.coef[2] .+
                 [(df.kaburban[i] == all_groups[1]) ? 0.0 :
                  r_B.coef[1 + findfirst(isequal(df.kaburban[i]), all_groups)]
                  for i=1:size(df,1)]))

    # Build and save plots, with axes normalized to those in paper
    p1 = binscatter(df, @formula(lambda ~ logc), bins, legend = false,
                    xl = labels["logc"], xrange = (11,16), yrange = (0.,0.4),
                    yl = "(Pred.) Probability to Receive Benefits")
    Plots.savefig(p1, "output/plots/fig1A.png")

    p2 = binscatter(df, @formula(mu_pmt ~ PMTSCORE), bins, legend = false,
                    xl = "Pred. Log Consumption (PMT score)",
                    yl = "(Pred.) Probability to Receive Benefits",
                    xrange = (11,15), yrange = (0,1))
    Plots.savefig(p2, "output/plots/fig1B.png")

    return df, p1, p2
end

###########################
# Figure 2
###########################
function figure_2(; labels::Dict{String,String} = labels)
    # Load and clean data
    df = rename!(DataFrame(load("input/matched_baseline.dta")),
                 [:logconsumption  => :logc])
    df = clean(df[df.selftargeting .== 1, :], [:logc, :showup, :hhea],
               out_t = Dict([:logc, :showup] .=> F64))

    x0_grid = collect(range(11.25; stop = 15, length = 200))
    y_hat, ub, lb = fan_reg(@formula(showup ~ logc), df, x0_grid;
                            clust = :hhea, bw = :cross_val)

    p = plot(x0_grid, y_hat, legend = false, xrange = (11,15.25), yrange = (0,0.8),
             xl = labels["logc"], yl = "Show-up Probability", lw = 2.0, lc = :cyan3)
    plot!(p, x0_grid, [lb, ub], ls = :dash, lw = 1.5, lc = :cyan4)

    savefig(p, "output/plots/fig2.png")
    return p
end

###########################
# Figure 3
###########################
function figure_3(; labels::Dict{String,String} = labels)
    # Load and clean data
    df = rename!(DataFrame(load("input/matched_baseline.dta")),
                 [:logconsumption  => :logc])
    df = clean(df[df.selftargeting .== 1, :], [:logc, :PMTSCORE, :showup, :hhea],
               out_t = Dict([:showup, :logc, :PMTSCORE] .=> F64))

    # Observable component of log consumption
    x0_grid = collect(range(11.5; stop = 14.5, length = 100))
    y_hat, ub, lb = fan_reg(@formula(showup ~ PMTSCORE), df, x0_grid; clust = :hhea, bw=:cross_val)

    pA = plot(x0_grid, y_hat, legend = false, xrange = (11.4, 15), yrange = (0., 0.8),
              xl = "Observable Component of Log Consumption (PMT Score)",
              yl = "Show-up Probability", lw = 2, lc = :cyan3)
    plot!(pA, x0_grid, [lb, ub], ls = :dash, lw = 1.5, lc = :cyan4)
    Plots.savefig(pA, "output/plots/fig3A.png")

    # Compute residual unobservable consumption
    r = reg(df, @formula(logc ~ PMTSCORE), cluster(:hhea), save = :residuals)
    insertcols!(df, :eps => Vector{Float64}(r.residuals))

    # Unobservable component of log consumption
    x0_grid = collect(range(-1.75; stop = 1.75, length = 100))
    y_hat, ub, lb = fan_reg(@formula(showup ~ eps), df, x0_grid; clust = :hhea, bw=:cross_val)

    pB = plot(x0_grid, y_hat, legend = false, xrange = (-2, 2), yrange = (0., 0.8),
              xl = "Unobservable Component of Log Consumption",
              yl = "Show-up Probability", lw = 2, lc = :cyan3)
    plot!(pB, x0_grid, [lb, ub], ls=:dash, lw = 1.5, lc = :cyan4)
    Plots.savefig(pB, "output/plots/fig3B.png")

    return pA, pB
end

###########################
# Figure 4 (p. 399)
###########################
function figure_4(; labels::Dict{String,String} = labels)
    # Load, clean, separate data
    df = clean(rename!(DataFrame(load("input/matched_baseline.dta")),
                 [:logconsumption  => :logc]), Dict(:logc => F64))
    df1    = df.logc[(df.getbenefit.==1) .& (df.maintreatment.==1)]
    df2    = df.logc[(df.getbenefit.==1) .& (df.maintreatment.==2)]
    n1, n2 = length(df1), length(df2)

    # Panel A: Experimental Cumulative Distribution
    pA = plot(sort(df1), (1:n1)./n1, xrange = (11, 15), lw = 2, lc = :cyan3,
              title = "Experimental Cumuluative Distribution",
              xl = "Log Consumption", yl = "CDF",
              legend=:bottomright, label = "Automatic Screening")
    plot!(pA, sort(df2), (1:n2)./n2, lw = 2, lc = :cyan4, label="Self-Targeting")
    Plots.savefig(pA, "output/plots/fig4A.png")

    # Panel B: Probability receive benefits under either treatment, as fn of logc
    df = clean(df, [:getbenefit, :maintreatment, :hhea],
               out_t = Dict([:getbenefit, :maintreatment] .=> F64))

    # Unobservable component of log consumption
    x0_grid = collect(range(11.25; stop = 15.25, length = 100))

    # I did a bit of experimentation setting different bandwidths to back out
    # what was likely being used in the Stata code! Cross-validation chooses a band-
    # width which is pretty close, but doesn't match the figure in the paper as well.
    y1, u1, l1 = fan_reg(@formula(getbenefit ~ logc),
                         df[df.maintreatment.==1,:], x0_grid; clust = :hhea, bw = 0.2)
    y2, u2, l2 = fan_reg(@formula(getbenefit ~ logc),
                      df[df.maintreatment.==2,:],    x0_grid; clust = :hhea, bw = 0.2)

    pB = plot(x0_grid, y1, legend = :topright, xrange = (11, 15.4), yrange = (0., 0.4),
              xl = "Log Consumption", label="Automatic Screening",
              yl = "Probability to Receive Benefits", lw = 2, lc = :lightpink1)
    plot!(pB, x0_grid, [l1, u1], ls=:dash, lw = 1.5, lc = :palevioletred, label="")

    plot!(pB, x0_grid, y2,       lw = 2,   lc = :cyan3, label="Self-Targeting")
    plot!(pB, x0_grid, [l2, u2], lw = 1.5, lc = :cyan4, label="", ls=:dash)

    Plots.savefig(pB, "output/plots/fig4B.png")

    return pA, pB
end


###########################
# Figure 5 (p. 408)
###########################
function figure_5(; labels::Dict{String,String} = labels)
    # Load, clean, separate data
    df = clean(rename!(DataFrame(load("input/matched_baseline.dta")),
                 [:logconsumption  => :logc]), Dict(:logc => F64))
    insertcols!(df, :getbenefit_hyp => Vector{Float64}(
                (df.pcexppred_noise .< df.povertyline3_noise) .&
                .!((df.showup .== 0) .& (df.selftargeting .== 1))))

    df1    = df.logc[(df.getbenefit_hyp.==1) .& (df.maintreatment.==1)]
    df2    = df.logc[(df.getbenefit_hyp.==1) .& (df.maintreatment.==2)]
    n1, n2 = length(df1), length(df2)

    # Panel A: Experimental Cumulative Distribution
    pA = plot(sort(df1), (1:n1)./n1, xrange = (11, 15), lw = 2, lc = :cyan3,
              legend=:bottomright, xl = "Log Consumption", yl = "CDF",
              label = "Hypothetical Universal Automatic Targeting")
    plot!(pA, sort(df2), (1:n2)./n2, lw = 2, lc = :cyan4, label="Self-Targeting")
    Plots.savefig(pA, "output/plots/fig5A.png")

    # Panel B: Probability receive benefits under either treatment, as fn of logc
    df = clean(df, [:maintreatment, :hhea],
               out_t = Dict([:maintreatment] .=> F64))

    # Unobservable component of log consumption
    x0_grid = collect(range(11.2; stop = 15.5, length = 100))

    # Run both Fan regressions
    y1, u1, l1 = fan_reg(@formula(getbenefit_hyp ~ logc),
                         df[df.maintreatment.==1,:], x0_grid; clust = :hhea, bw = 0.18)
    y2, u2, l2 = fan_reg(@formula(getbenefit_hyp ~ logc),
                         df[df.maintreatment.==2,:], x0_grid; clust = :hhea, bw = 0.2)

    pB = plot(x0_grid, y1, legend = :topright, xrange = (11, 15.5), yrange = (0., 0.4),
              xl = "Log Consumption", yl = "Probability to Receive Benefits",
              label="Hypothetical Universal Automatic Targeting", lw = 2, lc = :lightpink1)
    plot!(pB, x0_grid, [l1, u1], lw = 1.5, lc = :palevioletred, label="", ls=:dash)
    plot!(pB, x0_grid, y2,       lw = 2,   lc = :cyan3, label="Self-Targeting")
    plot!(pB, x0_grid, [l2, u2], lw = 1.5, lc = :cyan4, label="", ls=:dash)

    Plots.savefig(pB, "output/plots/fig5B.png")
    return pA, pB
end

###########################
# Figure 6 (p. 413)
###########################
function figure_6(; labels::Dict{String,String} = labels)
    # Load + clean data
    df = clean(rename!(DataFrame(load("input/costs.dta")),
               [:consumption  => :c]), Dict([:c, :totcost_pc] .=> F64))
    # Confirm concavity of graph
    insertcols!(df, :c2 => df.c .^ 2)
    r = reg(df,                  @formula(totcost_pc ~ c + c2))
    r = reg(df[df.c .< 2000000,:], @formula(totcost_pc ~ c + c2))

    x0_grid = collect(range(0; stop = 4000000, length = 100))
    y_hat, ub, lb = fan_reg(@formula(totcost_pc ~ c), df, x0_grid; clust = :hhea, bw=3e5)

    # Plot Fan regression
    p = plot(x0_grid, y_hat, legend = false, xrange = (0, 4000000), yrange = (0, 60000),
             xl = "Per Capita Consumption", yl = "Total Costs per Capita",
             lw = 2, lc = :cyan3)
    plot!(p, x0_grid, [lb, ub], ls=:dash, lw = 1.5, lc = :cyan4)
    Plots.savefig(p, "output/plots/fig6.png")
    return p
 end
