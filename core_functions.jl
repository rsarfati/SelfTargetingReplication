"""
Shortcut identity matrix.
"""
function eye(N::Int64)
    return Matrix{F64}(I, N, N)
end


"""
Trapezoidal integration w/ uniform grid!
"""
function trapz(f::Function, a::F64, b::F64, N_g::Int64)
    integ = f(a) + f(b)
    N_g -= 1
    h = (b-a) / N_g
    x = a
    for j = 2:N_g
        x     += h
        integ += 2 * f(x)
    end
    return integ * h / 2
end

"""
```
glm_clust(f::FormulaTerm, df::DataFrame; link::Link=LogitLink(),
          group::Symbol=Symbol(), clust::Symbol=Symbol(),
          wts::Vector{F64} = Vector{F64}(), glm_kwargs = Dict())
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
        @assert group != Symbol() "The `group` keyword is required for the " *
                                  "probit model. Check your inputs."
    end

    # Option to include weights for observations
    if !isempty(wts); glm_kwargs[:wts] = wts end

    # Specify strata for FEs in GLM.jl lingo
    if group != Symbol(); glm_kwargs[:contrasts]=Dict(group=>DummyCoding()) end

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

    # Output conformed to FixedEffectModel in both cases, for type stability
    if link == ProbitLink() || group == Symbol()
        # Case: {Logit, Probit}
        return μ_r_y, FixedEffectModel(coef(r), vcov_i, gen_fields[3:end]...)
    else
        # Case: {Conditional Logit}
        inds = 1:(try length(f.rhs) catch; 2 end) # Fix for if 1-elmt tuple
        return μ_r_y, FixedEffectModel(coef(r)[inds], vcov_i[inds,inds],
               gen_fields[3:8]..., gen_fields[9][inds], gen_fields[10:end]...)
    end
end

"""
```
bs_σ(V::Matrix{T}; conf::T = 0.975, draws::Int64=1000, ind = 2) where T<:F64
```
Bootstrap confidence intervals!
"""
function bs_σ(V::Matrix{T}; conf::T = 0.975, draws::Int64 = 1000,
              ind::Int64 = 2) where T<:F64
    # Define constants, gut-check inputs
    N = size(V, 1)
    @assert 0 < ind <= N "Provided `ind` doesn't correspond to valid " *
                          "explanatory variable; check input."
    return quantile(rand(MvNormal(zeros(N), V), draws)[ind,:], [1-conf, conf])
end
function bs_σ(V::T; conf::T = 0.975, draws::Int64 = 1000) where T<:F64
    return quantile(abs.(rand(Normal(0, V), draws)), conf)
end

"""
```
bootstrap(df::DataFrame, f::Function; N_bs = 1000, α::F64 = 0.05,
          clust::Symbol = Symbol(), domain::Vector{F64} = Vector(),
          id::String = "")
```
"""
function bootstrap(df::DataFrame, f::Function; N_bs::Int64 = 1000, α::T = 0.05,
                   clust::Symbol = Symbol(), domain::Vector{T} = Vector(),
                   id::String = "") where T<:F64

    # If pass symbol to cluster on, extract values on which to cluster
    clustering = (clust != Symbol())
    clust_set  = clustering ? unique(df[:, clust]) : Vector()

    # Is evaluated function applied to domain? (e.g. Fan regression grid)
    N_x = length(domain)

    # Store bootstrapped output
    all_bs = (N_x==0) ? Vector{F64}(undef, N_bs) :
                         Array{F64}(undef, N_bs, N_x)

    (id!="") ? println("$(id)Running $(N_bs) boostrap iterations...") : nothing

    ## Run bootstrap iterations!
    for i=1:N_bs
        # Sample at cluster-level, include all obs. w/in cluster in sample
        idx_bs = if clustering
            bs_clust = sample(clust_set, length(clust_set); replace = true)
            vcat([findall(isequal(c), df[:,clust]) for c in bs_clust]...)
        # No clustering
        else
            sample(1:N, N; replace = true)
        end
        # Evaluate directly OR ...
        (N_x == 0) ?        all_bs[i]   = f(df[idx_bs,:]) :
            # ... evaluate over provided domain
            for j=1:N_x;    all_bs[i,j] = f(df[idx_bs,:], domain[j]) end
    end
    if N_x == 0
        return quantile(all_bs, α/2), quantile(all_bs, 1 - (α/2))
    else
        return quantile.([all_bs[:,j] for j=1:N_x],    α/2),
               quantile.([all_bs[:,j] for j=1:N_x], 1-(α/2))
    end
end

"""
Manually drop groups which fail positivity!
"""
G_drop(d::DataFrame, v::Symbol) = findall(
                         g->(prod(d[d.kecagroup .== g, v] .== 0.0) ||
                             prod(d[d.kecagroup .== g, v] .== 1.0)),
                             unique(d.kecagroup))
df_drop(v::Symbol, d::DataFrame) = d[d.kecagroup .∉ (G_drop(d, v),), :]


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
                 run_bootstrap = true, b_ind=2, N_bs = 1000, α = 0.05, caller_id = "")

    id  = (caller_id != "") ? "($(caller_id)) " : ""
    X   = df[:, Symbol(f.rhs)]
    Y   = df[:, Symbol(f.lhs)]
    N   = size(df, 1)
    N_g = length(x0_grid)

    # Normal kernel:
    K_normal(z::F64) = (2*pi)^(-0.5) * exp(-z^2 / 2)

    ########################################################
    # Local linear regression smoother (Fan, 1992)
    ########################################################
    function m_hat(X::Vector{F64}, Y::Vector{F64}, x0::F64, h::F64;
                   K::Function = K_normal, save_w_ind = 0, return_CI = false)
        N = length(X)
        # Eq 2.4
        s_n(l::Int64) = sum(( K.((x0 .- X) ./ h) .* abs.(x0 .- X) .^ l))
        # Eq 2.3
        w = K.((x0 .- X) ./ h) .* (s_n(2) .- (x0 .- X) .* s_n(1))
        # Eq 2.2
        m_x = sum(w .* Y) / (sum(w) + N^(-2))
        # Per Alberto's lecture notes on local polynomial regression
        Kx1   = K.((x0 .- X) ./ h)
        i_sum = inv(sum([ Kx1[j] * (1 + X[j]^2)    for j=1:N]))
        Kx    = [i_sum * Kx1[i] .* (1 + x0 * X[i]) for i=1:N]
        # Flag used when computing optimal bandwidth
        if save_w_ind > 0
            return m_x, Kx[save_w_ind]
        # Returns analytically evaluated confidence bounds
        elseif return_CI
            df_σ = DataFrame(:Y => Y, :X => X, :Kx => w)
            # Case: not clustering SEs
            r_σ = if (clust == Symbol())
                reg(df_σ, @formula(Y ~ X), weights = :Kx)
            # Case: clustering SEs
            else
                df_σ = insertcols!(df_σ, clust => df[:, clust])
                reg(df_σ, @formula(Y ~ X), cluster(clust), weights = :Kx)
            end
            b = stderror(r_σ)[b_ind]
            bl, bu = (b !== 0) ? (m_x-b, m_x+b) : (NaN, NaN)
            return m_x, bl, bu
        else
            return m_x, 0., 0.
        end
    end
    ########################################################
    # Select bandwidth (if not given as input)
    ########################################################
    h0 = if typeof(bw) <: F64
        bw
    else
        println("$(id)Selecting optimal bandwidth via cross-validation...")
        # Normal reference rule / Silverman's rule of thumb
        if     bw == :norm_ref
            min(std(X), iqr(X) / 1.349) * N^(-1/5)
        # 1/Nth of total distance (used by Stata function)
        elseif bw == :tot_dist
            (maximum(X) - minimum(X)) / N
        # Choose h0 by cross-validation method (Stone 1977)
        elseif bw == :cross_val
            # Function computes cross-validated MSE
            function CV(h_test)
                m_i, w_i = zeros(N), zeros(N)
                for i=1:N
                    m_i[i], w_i[i] = m_hat(X, Y, X[i], h_test; save_w_ind = i)
                end
                return sum(((Y .- m_i) ./ (1 .- w_i)).^2)
            end
            # Enumerate grid of possible values for bandwidth
            h_grid = collect(range(0.25, stop = x0_grid[end]/5, length = 30))
            # Choose bandwidth which minimizes cross-validated MSE
            h_grid[argmin(CV.(h_grid))]
        else
            @error "$(id)Bandwidth selection method invalid -- check argument!"
        end
    end
    ########################################################
    # Run Fan regression (computes m_hat over grid of x's)
    ########################################################
    y_hat, lb, ub = zeros(N_g), zeros(N_g), zeros(N_g)
    for i=1:N_g
        y_hat[i], lb[i], ub[i] = m_hat(X, Y, x0_grid[i], h0; return_CI = !run_bootstrap)
    end
    ########################################################
    # Bootstrap confidence bands
    ########################################################
    if run_bootstrap
        println("$(id)Bootstrapping standard errors! " *
                "Default N_bs = 1000 takes ~1 min; think of a happy memory " *
                "while you're waiting.")

        # Define function which one intends to bootstrap
        bs_fun(df0::DataFrame, x0::Float64) = m_hat(df0[:,:X], df0[:,:Y], x0, h0)[1]
        # Construct a minimally-sized DataFrame
        df_bs = DataFrame(:X => X, :Y => Y, clust => df[:,clust])
        # Run bootstrap!
        lb, ub = bootstrap(df_bs, bs_fun; N_bs = N_bs, α = α, clust = clust,
                            domain = x0_grid, id=id)
        # # Store bootstrapped Fan regressions
        # m_hat_bs = Matrix{F64}(undef, N_bs, N_g)
        # # Extract values to cluster on
        # clust_on = (clust == Symbol()) ? Vector() : df[:, clust]
        # N_c = unique(clust_on)
        # # Bootstrap!!!!!!!!
        # for i=1:N_bs
        #     # Case: Clustering
        #     idx_bs = if clust != Symbol()
        #         bs_clust = sample(N_c, length(N_c); replace = true)
        #         vcat([findall(isequal(c), df[:,clust]) for c in bs_clust]...)
        #     # Case: No clustering
        #     else
        #         sample(1:N, N; replace = true)
        #     end
        #     # Evaluate Fan regression for each bootstrapped sample
        #     for j=1:N_g
        #         m_hat_bs[i,j],_,_ = m_hat(X[idx_bs], Y[idx_bs], x0_grid[j], h0)
        #     end
        # end
        # lb = quantile.([m_hat_bs[:,j] for j=1:N_g],    α/2)
        # ub = quantile.([m_hat_bs[:,j] for j=1:N_g], 1-(α/2))

    end
    return y_hat, lb, ub
end

########################################
# Data Cleaning
########################################
Base.parse(t::Type{String}, str::String) = str

"""
Helper to coerce DataFrame types to desired format.
"""
function convert_types(df, colstypes)
    for (c, t) in colstypes
        df[!, c] = if typeof(df[findfirst(!ismissing,df[:,c]),c]) == String
            convert.(t, parse.(t, df[!, c]))
        else
            convert.(t, df[!, c])
        end
    end
    return df
end

"""
```
clean(df::DataFrame, c::Symbol;
      out_t::Type = eltype(findfirst(!ismissing, df[:,c])))
clean(df::DataFrame, cols::Vector{Symbol}; out_t::Dict{Symbol,<:Type} = Dict())
clean(df::DataFrame, cols::Vector{Symbol}, out_t::Type)
```
Functions to convert & filter data so as to be consistently typed & non-missing.
"""
function clean(df::DataFrame, c::Symbol;
               out_t::Type = typeof(df[findfirst(!ismissing, df[:,c]), c]))
    return convert_types(df[.!ismissing.(df[:,c]), :], [c] .=> out_t)
end
function clean(df::DataFrame, cols::Vector{Symbol}; out_t::Dict = Dict())
    for c in cols
        df = haskey(out_t, c) ? clean(df, c; out_t = out_t[c]) : clean(df, c)
    end
    return df
end
clean(df::DataFrame, t::Dict{Symbol,<:Type})=clean(df,collect(keys(t));out_t=t)
clean(df::DataFrame, cols::Vector{Symbol}, out_t::Type) = clean(df, cols;
                     out_t = Dict([cols .=> repeat([out_t], length(cols))]...))
