"""
```
function glm_clust(f::FormulaTerm, df::DataFrame, link::Link; group::Symbol=Symbol(),
                   clust::Symbol=Symbol(), wts::Vector{F64} = Vector{F64}())
```
Helper function to make printable RegressionModel object with clustered SEs.

This is needlessly complicated only because existing Julia packages don't let
you run a regression with a link function AND cluster your standard errors.
* shakes head *
"""
function glm_clust(f::FormulaTerm, df::DataFrame; link::Link=LogitLink(),
                   group::Symbol=Symbol(), clust::Symbol=Symbol(),
                   wts::Vector{F64} = Vector{F64}(), glm_kwargs = Dict())
    # Input checks
    if link == ProbitLink()
        @assert group != Symbol() "The `group` kwarg is required for probit. Check inputs."
    end
    # Option to include weights for observations
    if !isempty(wts); glm_kwargs[:wts] = wts end
    # Specify strata for FEs in GLM.jl lingo
    if group != Symbol(); glm_kwargs[:contrasts] = Dict(group => DummyCoding()) end

    @show glm_kwargs

    # Run regression
    r = glm(f, df, Binomial(), link; glm_kwargs...)

    # Compute mean of dependent variable
    μ_r_y = mean(r.model.rr.y)

    # Here is the reason this function is complicated: clustering SEs...
    vcov_i, r_fields = if clust == Symbol()
        # Case 1: no need to custer SEs (i.e. scatter plots)
        vcov(r), reg(df, f)
    else
        # Case 2: convert SEs to be cluster-robust
        c_t = typeof(df[1, clust])
        vcov(CRHC0(convert(Array{c_t}, df[:, clust])), r),
        reg(df, f, Vcov.cluster(clust))
    end

    # Collect fields for automatic table production
    gen_fields = [getfield(r_fields, s) for s in fieldnames(typeof(r_fields))]

    # Output conformed to type FixedEffectModel in either case, for type stability
    if link == ProbitLink() || group == Symbol()
        # Case: {Logit, Probit}
        return μ_r_y, FixedEffectModel(coef(r), vcov_i, gen_fields[3:end]...)
    else
        # Case: {Conditional Logit}
        inds = 1:(try length(f.rhs) catch; 2 end) # Workaround in case of 1-element tuple.
        return μ_r_y, FixedEffectModel(coef(r)[inds], vcov_i[inds,inds],
                          gen_fields[3:8]..., gen_fields[9][inds], gen_fields[10:end]...)
    end
end

"""
```
bs_σ(V::Matrix{T}; conf::T = 0.975, draws::Int64 = 1000, ind::Int64 = 2) where T<:F64
```
Bootstrap confidence intervals!
"""
function bs_σ(V::Matrix{T}; conf::T = 0.975, draws::Int64 = 1000, ind::Int64 = 2) where T<:F64
    N = size(V, 1)
    @assert 0 < ind <= N "Provided `ind` doesn't correspond to valid explanatory variable; check input."
    return quantile(rand(MvNormal(zeros(N), V), draws)[ind,:], [1-conf, conf])
end
function bs_σ(V::T; conf::T = 0.975, draws::Int64 = 1000) where T<:F64
    return quantile(abs.(rand(Normal(0, V), draws)), conf)
end

"""
```
function sup_t(vcov::Matrix{T}, conf::F64; draws=10000) where T<:F64
```
Helper function to implement sup-t confidence bands.
"""
function sup_t(V::Matrix{T}; conf::T = 0.975, draws::Int64 = 1000) where T<:F64
    N = size(V, 1)
    d = MvNormal(zeros(N), V)
    return quantile(vec(maximum(abs.(rand(d, draws)), dims = 2)), conf)
end


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
function fan_reg(f::FormulaTerm, df::DataFrame, x0_grid::Vector{F64};
                 bw::Union{Symbol,F64} = :norm_ref, clust::Symbol = Symbol(),
                 compute_σ::Symbol = :analytic, N_bs::Int64 = 1000)

    X   = df[:, Symbol(f.rhs)]
    Y   = df[:, Symbol(f.lhs)]
    N   = size(df, 1)
    N_g = length(x0_grid)
    cluster_on = (clust == Symbol()) ? Vector() : df[:, clust]

    # Normal kernel:
    K_normal(z::F64) = (2*pi)^(-0.5) * exp(-z^2 / 2)

    """
    Local linear regression smoother (Fan 1992).
    Keyword option for computing boostrap confidence intervals.
    """
    function m_hat(X::Vector{F64}, Y::Vector{F64}, x0::F64, h::F64;
                   K::Function = K_normal, compute_σ::Symbol = :analytic,
                   cluster_on::Vector = Vector(), N_bs::Int64 = 1000,
                   save_w_ind::Int64 = 0)
        # Eq 2.4
        s_n(l::Int64) = sum(( K.((x0 .- X) ./ h) .* abs.(x0 .- X) .^ l))
        # Eq 2.3
        w = K.((x0 .- X) ./ h) .* (s_n(2) .- (x0 .- X) .* s_n(1))
        # Eq 2.2
        m_x = sum(w .* Y) / (sum(w) + N^(-2))

        # Alberto's lecture notes on local polynomial regression
        Kx1 = K.((x0 .- X) ./ h)
        i_sum = inv(sum([ Kx1[j] * (1 + X[j]^2) for j=1:N]))
        Kx  = [i_sum * Kx1[i] .* (1 + x0 * X[i]) for i=1:N]

        # Construct confidence bands
        if compute_σ != :none
            df_σ = DataFrame(:Y => Y, :X => X, :Kx => Kx)

            # Case: not clustering SEs
            r_σ = if isempty(cluster_on)
                reg(df_σ, @formula(Y ~ X), weights = :Kx)
            # Case: clustering SEs
            else
                df_σ = insertcols!(df_σ, clust => cluster_on)
                reg(df_σ, @formula(Y ~ X), cluster(clust), weights = :Kx)
            end

            bl, bu = if compute_σ == :bootstrap

                V = vcov(r_σ)
                bs_σ(V; draws = N_bs, ind = 2)

            elseif compute_σ == :analytic
                b = 1.96 * stderror(r_σ)[2] / 2
                -b, b
            else
                @error "Implemented options for CIs limited to :bootstrap and :analytic."
            end
            lb_i = m_x + bl
            ub_i = m_x + bu
            return m_x, lb_i, ub_i
        else
            if save_w_ind > 0
                m_x, Kx[save_w_ind]
            else
                return m_x, 0.0, 0.0
            end
        end
    end

    # Select bandwidth (if not given as input)
    h0 = if typeof(bw) <: F64
        bw
    else
        # Normal reference rule / Silverman's rule of thumb
        if bw == :norm_ref
            0.9 * min(std(X), iqr(X) / 1.349) * N^(-1/5)
        # 1-Nth of total distance (used by Stata command?)
        elseif bw == :tot_dist
            (maximum(X) - minimum(X)) / N
        # Choose h0 by cross-validation method (Stone 1977)
        elseif bw == :cross_val
            function CV(h_test)
                m_i, w_i = zeros(N), zeros(N)
                for i=1:N
                    m_i[i], w_i[i] = m_hat(Y, X, X[i], h_test; compute_σ = :none, save_w_ind = i)
                end
                return sum(((Y .- m_i) ./ (1 .- w_i)).^2)
            end
            h_grid = collect(range(0.35, stop = 2.0, length = 20))
            h_grid[argmin(CV.(h_grid))]
        else
            @error "Provided bandwidth selection method invalid -- check input!"
        end
    end

    # Compute m_hat over grid of x's
    y_hat, lb, ub = zeros(N_g), zeros(N_g), zeros(N_g)
    for i=1:N_g
        y_hat[i], lb[i], ub[i] = m_hat(X, Y, x0_grid[i], h0; cluster_on = cluster_on,
                                       compute_σ = compute_σ, N_bs = N_bs)
    end
    return y_hat, lb, ub
end

"""
Helper to coerce DataFrame types to desired format.
"""
function convert_types(df, colstypes)
    for (c, t) in colstypes
        df[!, c] = convert.(t, df[!, c])
    end
    return df
end

"""
```
clean(df::DataFrame, c::Symbol; out_t::Type = eltype(findfirst(!ismissing, df[:,c])))
clean(df::DataFrame, cols::Vector{Symbol}; out_t::Dict{Symbol,<:Type} = Dict())
clean(df::DataFrame, cols::Vector{Symbol}, out_t::Type)
```
Functions to convert & filter data so as to be consistently typed & non-missing.
"""
function clean(df::DataFrame, c::Symbol; out_t::Type = typeof(df[findfirst(!ismissing, df[:,c]), c]))
    return convert_types(df[.!ismissing.(df[:,c]), :], [c] .=> out_t)
end
function clean(df::DataFrame, cols::Vector{Symbol}; out_t::Dict = Dict())
    for c in cols
        df = haskey(out_t, c) ? clean(df, c; out_t = out_t[c]) : clean(df, c)
    end
    return df
end
clean(df::DataFrame, t::Dict{Symbol,<:Type}) = clean(df, collect(keys(t)); out_t = t)
clean(df::DataFrame, cols::Vector{Symbol}, out_t::Type) = clean(df, cols;
                     out_t = Dict([cols .=> repeat([out_t], length(cols))]...))
