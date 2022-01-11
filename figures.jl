###########################
# Figure 1
###########################
function figure_1(; labels::Dict{String,String} = labels, bins::Int64 = 0)

    # Load Data
    df = DataFrame(load("input_data/matched_baseline.dta"))
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
               out_type = Dict([:getbenefit, :logc, :PMTSCORE] .=> F64))

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
                    xlabel = labels["logc"], xrange = (11,16), yrange = (0.,0.4),
                    ylabel = "(Pred.) Probability to Receive Benefits")
    Plots.savefig(p1, "plots/fig1A.png")

    p2 = binscatter(df, @formula(mu_pmt ~ PMTSCORE), bins, legend = false,
                    xlabel = "Pred. Log Consumption (PMT score)",
                    ylabel = "(Pred.) Probability to Receive Benefits",
                    xrange = (11,15), yrange = (0,1))
    Plots.savefig(p2, "plots/fig1B.png")

    return df, p1, p2
end

###########################
# Figure 2
###########################
function figure_2(; labels::Dict{String,String} = labels)
    # Load and clean data
    df = rename!(DataFrame(load("input_data/matched_baseline.dta")),
                 [:logconsumption  => :logc])
    df = clean(df[df.selftargeting .== 1, :], [:logc, :showup, :hhea],
               out_type = Dict([:logc, :showup] .=> F64))

    x0_grid = collect(range(11.25; stop = 15, length = 200))
    y_hat, ub, lb = fan_reg(@formula(showup ~ logc), df, x0_grid; clust = :hhea)

    p = plot(x0_grid, y_hat, legend = false, xrange = (11,15.25), yrange = (0,0.8),
             xl = labels["logc"], yl = "Show-up Probability", lw = 2.0, lc = :cyan3)
    plot!(p, x0_grid, [lb, ub], ls=:dash, lw = 1.5, lc = :cyan4)

    savefig(p, "plots/fig2.png")
    return p
end

###########################
# Figure 3
###########################
function figure_3(; labels::Dict{String,String} = labels)
    # Load and clean data
    df = rename!(DataFrame(load("input_data/matched_baseline.dta")),
                 [:logconsumption  => :logc])
    df = clean(df[df.selftargeting .== 1, :], [:logc, :PMTSCORE, :showup, :hhea],
               out_type = Dict([:showup, :logc, :PMTSCORE] .=> F64))

    # Observable component of log consumption
    x0_grid = collect(range(11.25; stop = 14.5, length = 100))
    y_hat, ub, lb = fan_reg(@formula(showup ~ PMTSCORE), df, x0_grid; clust = :hhea)

    pA = plot(x0_grid, y_hat, legend = false, xrange = (11, 15), yrange = (0., 0.8),
              xl = "Observable Component of Log Consumption (PMT Score)",
              yl = "Show-up Probability", lw = 2, lc = :cyan3)
    plot!(pA, x0_grid, [lb, ub], ls = :dash, lw = 1.5, lc = :cyan4)
    Plots.savefig(pA, "plots/fig3A.png")
    @show pA

    r = reg(df, @formula(logc ~ PMTSCORE), cluster(:hhea), save = :residuals)
    insertcols!(df, :eps => Vector{Float64}(r.residuals))

    # Unobservable component of log consumption
    x0_grid = collect(range(-1.5; stop = 2, length = 100))
    y_hat, ub, lb = fan_reg(@formula(showup ~ eps), df, x0_grid; clust = :hhea)

    pB = plot(x0_grid, y_hat, legend = false, xrange = (-2, 2), yrange = (0.,0.8),
              xl = "Unobservable Component of Log Consumption",
              yl = "Show-up Probability", lw = 2, lc = :cyan3)
    plot!(pB, x0_grid, [lb, ub], ls=:dash, lw = 1.5, lc = :cyan4)
    Plots.savefig(pB, "plots/fig3B.png")

    return pA, pB
end

###########################
# Figure 6 (p. 413)
###########################
