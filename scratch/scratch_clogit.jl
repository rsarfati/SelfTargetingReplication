"""
```
function fan_reg(f::FormulaTerm, df::DataFrame; clust::Symbol=Symbol(),
                 N_grid = 50, N_bs = 100)
```
Implemented from Section 2 of Fan (1992). As discussed in paper, method assumes
the second derivative of m(x) exists, s.t. in small neighborhood of point x:

m(y) ≈ m(x) + m'(x)(y-x) ≡ a + b(y-x)

Hence, can equate estimation of m(x) to estimating intercept term a in a
local linear regression.
"""
function fan_reg(f::FormulaTerm, df::DataFrame; clust::Symbol=Symbol(),
                 N_grid::S = 50, N_bs::S = 100) where S <: Int64

    N = size(df, 1)
    X = df[:, Symbol(f.rhs)]
    Y = df[:, Symbol(f.lhs)]

    # Normal kernel
    K(z::Float64) = (2*pi)^(-0.5) * exp(-z^2 / 2)
    # Normal reference rule for bandwidth
    h0 = 1.059 * std(X) / (N^(1/5))
    # Normal kernel with bandwidth adjustment
    K_h(z::Float64; h::Float64 = h0) = K(z / h) / h

    """
    Local linear regression!
    """
    function m_local(x0::Float64; h::Float64 = h0)
        # Compute and insert population weights
        K_x0 = [K.((X[i] - x0) / h) for i=1:N]
        #df_w = insertcols(df, [:w_i => K_x0])
        # Estimate regression
        _, r = logistic_reg(f, df; clust = :hhea, weights = K_x0)
        b = r.coef#coef(reg(df_w, @formula(y ~ x); weights = :w_i))
            #@show b
        return b[1] #+ b[2] * x0
    end

    x0_grid = collect(range(11; stop=15, length=N_grid))
    local_lin_reg = m_local.(x0_grid)
    return local_lin_reg
    for i=1:points

    end
    return out_line, lower, upper
end

function logistic_reg(f::FormulaTerm, df::DataFrame; clust::Symbol = Symbol(),
                      wts::Vector{Float64} = Vector{Float64}())

    r = if isempty(wts)
        glm(f, df, Binomial(), LogitLink())
    else
        glm(f, df, Binomial(), LogitLink(), wts = wts)
    end

    vcov_i, r_fields = if clust == Symbol()
        vcov(r), reg(df, f)
    else
        clust_t = eltype(df[:, clust])
        vcov(r, CRHC0(convert(Array{clust_t}, df[:, clust]))),
             reg(df, f, Vcov.cluster(clust))
    end
    gen_fields = [getfield(r_fields, field) for field in fieldnames(typeof(r_fields))]

    # Output conformed to type FixedEffectModel in either case, for type stability!
    return mean(r.model.rr.y), FixedEffectModel(coef(r), vcov_i, gen_fields[3:end]...)
end
function clogistic_reg(f::FormulaTerm, df::DataFrame, group::Symbol; clust::Symbol=Symbol())

    r_i = glm(f, df, Binomial(), LogitLink(); contrasts = Dict(group => DummyCoding()))

    vcov_i, r_fields = if clust == Symbol()
        vcov(r_i), reg(df, f)
    else
        clust_t = eltype(df[:, clust])
        vcov(r_i, CRHC0(convert(Array{clust_t}, df[:, clust]))),
            reg(df, f, Vcov.cluster(clust); contrasts = Dict(group => DummyCoding()))
    end
    gen_fields = [getfield(r_fields, field) for field in fieldnames(typeof(r_fields))]

    inds = 1:(try length(f.rhs) catch; 2 end) # Error if query length of 1-element tuple.

    return mean(r_i.model.rr.y), FixedEffectModel(coef(r_i)[inds], vcov_i[inds,inds],
                        gen_fields[3:8]..., gen_fields[9][inds], gen_fields[10:end]...)
end
function probit_reg(f::FormulaTerm, df::DataFrame, group::Symbol; clust::Symbol=Symbol())

    r_i   = glm(f, df, Binomial(), ProbitLink(); contrasts = Dict(group => DummyCoding()))
    μ_r_y = mean(r_i.model.rr.y)

    vcov_i, r_fields = if clust == Symbol()
        # No need to custer SEs -- i.e. scatter plots
        vcov(r_i), reg(df, f)
    else
        # Convert SEs to be cluster-robust
        clust_t = eltype(df[:, clust])
        vcov(r_i, CRHC0(convert(Array{clust_t}, df[:, clust]))),
            reg(df, f, Vcov.cluster(clust))
    end
    gen_fields = [getfield(r_fields, field) for field in fieldnames(typeof(r_fields))]

    # Output conformed to type FixedEffectModel in either case, for type stability!
    return μ_r_y, FixedEffectModel(coef(r_i), vcov_i, gen_fields[3:end]...)
end

#####################

function clogistic_reg(f::FormulaTerm, df::DataFrame, group::Symbol, clust::Symbol)

    # all_terms = Symbol.([f.lhs; f.rhs...])
    # df = clean(df, all_terms)
    #
    # Build Strata
    G = countmap(df[:,group])
    N_g = length(G)
    N = size(df, 1)

    glm(@formula(getbenefit ~ kecagroup), df, Binomial(), LogitLink(); contrasts = Dict(group => DummyCoding()))
    #G_w = Dict{Int,Float64}(G.keys .=> G.vals/N) nonsense

    r_i = glm(f, df, Binomial(), LogitLink(); contrasts = Dict(group => DummyCoding()))

    beta = coef(r_i)

    coef_final = Dict(f.rhs .=> [sum([beta[1+(B)*N_g + g]/(G_w[g]) for g=1:N_g]) for B=2:length(f.rhs)])

    i=1

    i= 2
    i= N_G + 1

    i = 2 + N_G
    i = 1 + 2*N_G

    i= 2 + 2*N_G
    i = 1 + 3*N_G

    @show r_i

    # mean(Matrix{Float64}(df[df[:, group] .== 58., all_terms]))
    # g_means = Dict([g => mean(Matrix(df[df.kecagroup .== g, all_terms]), dims=1) for g in G])
    # @show g_means
    #                 # [:getbenefit, :selftargeting, :logc,
    #                 #                              :logcselftarg, :mistarget, :excl_error,
    #                 #                              :incl_error]
    #
    # # Demean
    # @show size(vcat([g_means[i] for i in df.kecagroup]...))
    # df[:, all_terms] .-= vcat([g_means[i] for i in df.kecagroup]...)




    # X = Matrix(df_subset[:, Symbol.([f.rhs...])])
    # y = Vector(df_subset[:, clust])#[clust, Symbol(f.lhs)]])
    r_dummy    = reg(df_subset, f, Vcov.cluster(clust); contrasts = Dict(group => DummyCoding()))
    @show r_dummy



    @show "yo"
    vcov_i  = vcov(r_i, CRHC0(convert(Array{clust_t}, df_subset[:, clust])))

    r_dummy    = reg(df_subset, f, Vcov.cluster(clust); contrasts = Dict(group => DummyCoding()))
    gen_fields = [getfield(r_dummy, field) for field in fieldnames(typeof(r_dummy))]

    return mean(r_i.model.rr.y), FixedEffectModel(coef(r_i), vcov_i, gen_fields[3:end]...)
end

clean(df::DataFrame, col::Symbol) = convert_types(df[.!ismissing.(df[:,col]), :], [col] .=> F64)
function clean(df::DataFrame, cols::Vector{Symbol})
    for col in cols
        df = clean(df, col)
    end
    return df
end
