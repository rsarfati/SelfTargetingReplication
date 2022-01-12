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
                   wts::Vector{F64} = Vector{F64}())

    # Input checks
    if link == ProbitLink()
        @assert group != Symbol() "Require `group` kwarg for probit. Check inputs."
    end

    # Option to include weights for observations
    if !isempty(wts); glm_kwargs[:wts] = wts end
    # Specify strata for FEs in GLM.jl lingo
    if group != Symbol(); glm_kwargs[:contrasts] = Dict(group => DummyCoding()) end

    # Run regression
    r = glm(f, df, Binomial(), link; glm_kwargs...)

    # Compute mean of dependent variable
    μ_r_y = mean(r.model.rr.y)

    # Reason this function is complicated: clustering SEs...
    vcov_i, r_fields = if clust == Symbol()
        # Case: no need to custer SEs (i.e. scatter plots)
        vcov(r), reg(df, f)
    else
        # Case: convert SEs to be cluster-robust
        c_t = typeof(df[1,clust])
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
    d = MvNormal(zeros(N), V)
    @assert 0 < ind <= N "Keyword ind does not correspond to a variable; check input."
    return quantile(rand(d, draws)[ind,:], [1-conf, conf])
end
function bs_σ(V::T; conf::T = 0.975, draws::Int64 = 1000) where T<:F64
    d = Normal(0, V)
    return quantile(abs.(rand(d, draws)), conf)
end

"""
```
function sup_t(vcov::Matrix{T}, conf::F64; draws=10000) where T<:F64
```
Helper function to implement sup-t confidence bands.
"""
function sup_t(V::Matrix{T}; conf::T = 0.95, draws::Int64 = 1000) where T<:F64
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
                 bw::Union{Symbol,F64} = :norm_ref,
                 bootstrap_SEs::Bool = true, clust::Symbol = Symbol(),
                 N_bs::Int64 = 1000)

    X   = df[:, Symbol(f.rhs)]
    Y   = df[:, Symbol(f.lhs)]
    N   = size(df, 1)
    N_g = length(x0_grid)
    cluster_on = (clust == Symbol()) ? Vector() : df[:, clust]

    # Normal kernel
    K_normal(z::F64) = (2*pi)^(-0.5) * exp(-z^2 / 2)

    # Select bandwidth:
    h0 = if typeof(bw) <: F64
        bw
    else
        # Normal reference rule / Silverman's rule of thumb (should be 0.9)
        if bw == :norm_ref
            1.9 * min(std(X), iqr(X) / 1.349) * N^(-1/5)
        # 1-Nth of total distance (used by Stata)
        elseif bw == :tot_dist
            (maximum(X) - minimum(X)) / N
        # Choose h0 by cross-validation method (Stone 1977)
        elseif bw == :cross_val
            #TODO
            0.0
        else
            @error "Bandwidth selection method invalid -- check input!"
        end
    end

    """
    Local linear regression smoother (Fan 1992).
    Keyword available for computing boostrap confidence intervals.
    """
    function m_hat(X::Vector{F64}, Y::Vector{F64}, x0::F64, h::F64;
                   K::Function = K_normal, bootstrap_SEs::Bool = bootstrap_SEs,
                   cluster_on::Vector = Vector(), N_bs::Int64 = 500)
        # Eq 2.4
        s_n(l::Int64) = sum(( K.((x0 .- X) ./ h) .* abs.(x0 .- X) .^ l))
        # Eq 2.3
        w = K.((x0 .- X) ./ h) .* (s_n(2) .- (x0 .- X) .* s_n(1))
        # Eq 2.2
        m_x = sum(w .* Y) / (sum(w) + N^(-2))

        # Construct sup-t confidence bands
        if bootstrap_SEs
            df_SEs = DataFrame(:Y => Y, :X => X, :Kx => 1 ./ K.((x0 .- X) ./ h))

            # Case: not clustering SEs
            V = if isempty(cluster_on)
                vcov(reg(df_SEs, @formula(Y ~ X), weights=:Kx))
            # Case: clustering SEs
            else
                df_SEs = insertcols!(df_SEs, clust => cluster_on)
                vcov(reg(df_SEs, @formula(Y ~ X), cluster(clust), weights=:Kx))
            end
            bl, bu = bs_σ(V; draws = N_bs, ind=2)
            #b = sup_t(V; draws = N_bs)
            lb_i = m_x + bl
            ub_i = m_x + bu

            return m_x, lb_i, ub_i
        else
            return m_x, 0.0, 0.0
        end
    end
    # Compute m_hat over grid of x's
    y_hat, lb, ub = zeros(N_g), zeros(N_g), zeros(N_g)
    for i=1:N_g
        y_hat[i], lb[i], ub[i] = m_hat(X, Y, x0_grid[i], h0;
                                       bootstrap_SEs = bootstrap_SEs,
                                       cluster_on = cluster_on, N_bs = N_bs)
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
clean(df::DataFrame, col::Symbol;
      out_type::T = eltype(findfirst(!ismissing, df[:,col]))) where T<:Type
clean(df::DataFrame, cols::Vector{Symbol}; out_type::Dict{Symbol,<:Type} = Dict())
clean(df::DataFrame, cols::Vector{Symbol}, out_type::Type)
```
Functions to convert & filter data so as to be consistently typed & non-missing.
"""
function clean(df::DataFrame, col::Symbol;
               out_type::Type = typeof(df[findfirst(!ismissing, df[:,col]), col]))
    return convert_types(df[.!ismissing.(df[:,col]), :], [col] .=> out_type)
end
function clean(df::DataFrame, cols::Vector{Symbol};
               out_type::Dict{Symbol,<:Type} = Dict{Symbol,Type}())
    for col in cols
        df = haskey(out_type, col) ? clean(df, col; out_type = out_type[col]) :
                                     clean(df, col)
    end
    return df
end
clean(df::DataFrame, t::Dict{Symbol,<:Type}) = clean(df, collect(keys(t));
                                                     out_type = t)
clean(df::DataFrame, cs::Vector{Symbol}, out_type::Type) = clean(df, cs;
                     out_type = Dict([cs .=> repeat([out_type], length(cs))]...))
