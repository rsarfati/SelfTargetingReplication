# Saved helpers of 6 PM, Jan 4

"""
```
function glm_clust(f::FormulaTerm, df::DataFrame, link::Link; group::Symbol=Symbol(),
                   clust::Symbol=Symbol(), wts::Vector{Float64} = Vector{Float64}())
```
Helper function to make printable RegressionModel object with clustered SEs.

This is needlessly complicated only because existing Julia packages don't let
you run a regression with a link function AND cluster your standard errors.
* shakes head *
"""
function glm_clust(f::FormulaTerm, df::DataFrame, link::Link; group::Symbol=Symbol(),
                   clust::Symbol=Symbol(), wts::Vector{Float64} = Vector{Float64}())

    # Input checks
    if link == ProbitLink()
        @assert group != Symbol() "Must provide `group` kwarg for probit. Check inputs."
    end

    # Option to include weights for observations
    if !isempty(wts);     glm_kwargs[:wts] = wts end
    # Specify strata for FEs in GLM.jl lingo
    if group != Symbol(); glm_kwargs[:contrasts] = Dict(group => DummyCoding()) end

    # Run regression
    r = glm(f, df, Binomial(), link; glm_kwargs...)
    # Compute mean of dependent variable
    μ_r_y = mean(r.model.rr.y)

    # Reason this function is complicated: clustering SEs...
    vcov_i, r_fields = if clust == Symbol()
        # No need to custer SEs (i.e. scatter plots)
        vcov(r), reg(df, f)
    else
        # Convert SEs to be cluster-robust
        c_t = typeof(df[1,clust])
        vcov(CRHC0(convert(Array{c_t}, df[:, clust])), r), reg(df, f, Vcov.cluster(clust))
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
        return μ_r_y, FixedEffectModel(coef(r)[inds], vcov_i[inds,inds], gen_fields[3:8]...,
                                       gen_fields[9][inds], gen_fields[10:end]...)
    end
end

"""
```
function sup_t(vcov::Matrix{T}, conf::Float64; draws=10000) where T<:Float64
```
Helper function to implement sup-t confidence bands.
"""
function sup_t(V::Matrix{T}; conf::T=0.95, draws::Int64=1000) where T<:Float64
    d = MvNormal(zeros(size(V, 1)), V)
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
function fan_reg(f::FormulaTerm, df::DataFrame; clust::Symbol=Symbol(),
                 N_grid::S = 50, N_bs::S = 100) where S <: Int64

    N = size(df, 1)
    X = df[:, Symbol(f.rhs)]
    Y = df[:, Symbol(f.lhs)]

    # Normal kernel
    function gaussiankernel(x::Real, xdata::Vector{Float64}, h::Real, w::Vector, n::Int)
        h1= 1.0/h
        tmp = log(h) + log(2π)/2
        for ind in 1:n
            @inbounds w[ind] = -0.5 * abs2((x - xdata[ind])*h1) - tmp
        end
        w .= exp.(w)
    end
    function bwnormal(xdata::Vector{Float64})
        0.9 * min((quantile(xdata, .75) - quantile(xdata, .25)) / 1.34, std(xdata)) * length(xdata) ^ (-0.2)
    end
    # Reference: Smoothing Parameter Selection in Nonparametric Regression Using an Improved Akaike Information Criterion
    # Clifford M. Hurvich, Jeffrey S. Simonoff and Chih-Ling Tsai
    # Journal of the Royal Statistical Society. Series B (Statistical Methodology), Vol. 60, No. 2 (1998), pp. 271-293
    # http://www.jstor.org/stable/2985940
    function AIClocallinear(xdata::Vector{Float64}, ydata::Vector{Float64}, kernel::Function, h::Real, w::Vector, n::Int)
        tmp = 0.0
        traceH = 0.0
        ind = 1
        ind_end = 1+n
        @inbounds while ind < ind_end
              kernel(xdata[ind], xdata, h, w, n)
              s0 = sum(w)
              s1 = s0 * xdata[ind] - sum(w .* xdata)
              s2 = sum([w[j]*(xdata[j]-xdata[ind]).^2 for j=1:N]) #s2 = wsumsqdiff(w, xdata, xdata[ind], n)
              sy0 = sum(w .* ydata)
              sy1 = sum([w[j] * ydata[j] * (xdata[ind] - xdata[j]) for j=1:N]) #wsumyxdiff(w, xdata, xdata[ind], ydata, n)
              tmp += abs2((s2 * sy0 - s1 * sy1) /(s2 * s0 - s1 * s1) - ydata[ind])
              traceH += s0*w[ind]/(s2 * s0 - s1 * s1)
              ind += 1
        end
        tmp/n  + 2*(traceH+1)/(n-traceH-2)
    end
    function bwlocallinear(xdata::Vector{Float64}, ydata::Vector{Float64}, kernel::Function=gaussiankernel)
        n = length(xdata)
        length(ydata)==n || error("length(ydata) != length(xdata)")
        w = ones(n)
        if kernel == gaussiankernel
            h0 = bwnormal(xdata)
            hlb = 0.1 * h0
            hub = 10. * h0
        elseif kernel == betakernel
            h0 = midrange(xdata)
            hlb = h0 / n
            hub = 0.25
        elseif kernel == gammakernel
            h0 = midrange(xdata)
            hlb = h0/n
            hub = h0
        end
        Optim.minimizer(Optim.optimize(h -> AIClocallinear(xdata, ydata, kernel,h, w, n), hlb, hub))
    end

    function locallinear(xdata::Vector{Float64}, ydata::Vector{Float64}; xeval::Vector{Float64}=xdata, kernel::Function=gaussiankernel, h::Real=bwlocallinear(xdata, ydata, kernel))
        N = length(xdata)
        length(ydata) == N || error("length of ydata not the same with xdata")
        w = ones(N)
        pre = zeros(length(xeval))
        for i in 1:length(xeval)
            kernel(xeval[i], xdata, h, w, N)
            s0 = sum(w)
            s1 = s0*xeval[i] - sum(w .* xdata)
            s2 = sum([w[j]*(xdata[j]-xeval[i]).^2 for j=1:N]) #wsumsqdiff(w, xdata, xeval[i], n)
            sy0 = sum(w .* ydata)
            sy1 = sum([w[j] * ydata[j] * (xeval[i] - xdata[j]) for j=1:N]) #wsumyxdiff(w, xdata, xeval[i], ydata, n)
            pre[i] = (s2 * sy0 - s1 * sy1) /(s2 * s0 - s1 * s1)
        end
        pre
    end
    # locallinear(xdata::Vector{Float64}, ydata::Vector{Float64}, xeval::Real;
    #     kernel::Function = gaussiankernel, h::Real = bwlocallinear(xdata,ydata,kernel)) =
    # locallinear(xdata, ydata, xeval=[xeval;], kernel=kernel, h=h)


    # K(z::Float64) = (abs(z) <= 1) ? 0.5 : 0.0#(2*pi)^(-0.5) * exp(-z^2 / 2)
    # # Set Bandwidth
    # h0 = (maximum(X) - minimum(X)) / N
    # # Normal kernel with bandwidth adjustment
    # K_h(z::Float64; h::Float64 = h0) = K(z / h) / h
    #
    # """
    # Local linear regression!
    # """
    # function m_hat(x0::Float64; h::Float64 = h0)
    #     @show "1"
    #     # Eq 2.4
    #     s_n(l::Int64) = sum(( K.( (x0 .- X) ./ h) .* (x0 .- X) .^ l))
    #     # Eq 2.3
    #     w = K.( (x0 .- X) ./ h) .* (s_n(2) .- (x0 .- X) .* s_n(1))
    #     # Eq 2.2
    #     return sum(w .* Y) / sum(w)
    # end

    x0_grid = collect(range(11.0; stop=15.0, length=N_grid))
    local_lin_reg = locallinear(X, Y, xeval = x0_grid)#npr(X, Y, xeval=X, B=N_bs, reg=locallinear, kernel=gaussiankernel)#m_hat.(x0_grid)
    return local_lin_reg

    for i=1:points

    end
    return out_line, lower, upper
end
function fan_reg(f::FormulaTerm, df::DataFrame; clust::Symbol = Symbol(),
                 N_grid::S = 50, N_bs::S = 100) where S <: Int64

    N = size(df, 1)
    X = df[:, Symbol(f.rhs)]
    Y = df[:, Symbol(f.lhs)]

    # Normal kernel
    function gaussiankernel(x::Real, xdata::Vector{F64}, h::Real, w::Vector, n::Int)
        h1= 1.0/h
        tmp = log(h) + log(2π)/2
        for ind in 1:n
            @inbounds w[ind] = -0.5 * abs2((x - xdata[ind])*h1) - tmp
        end
        w .= exp.(w)
    end
    function bwnormal(xdata::Vector{F64})
        0.9 * min((quantile(xdata, .75) - quantile(xdata, .25)) / 1.34, std(xdata)) * length(xdata) ^ (-0.2)
    end

    # Reference: Smoothing Parameter Selection in Nonparametric Regression Using an Improved Akaike Information Criterion
    # Clifford M. Hurvich, Jeffrey S. Simonoff and Chih-Ling Tsai
    # Journal of the Royal Statistical Society. Series B (Statistical Methodology), Vol. 60, No. 2 (1998), pp. 271-293
    # http://www.jstor.org/stable/2985940
    function AIClocallinear(xdata::Vector{F64}, ydata::Vector{F64}, kernel::Function, h::Real, w::Vector, n::Int)
        tmp = 0.0
        traceH = 0.0
        ind = 1
        ind_end = 1+n
        @inbounds while ind < ind_end
              kernel(xdata[ind], xdata, h, w, n)
              s0 = sum(w)
              s1 = s0 * xdata[ind] - sum(w .* xdata)
              s2 = sum([w[j]*(xdata[j]-xdata[ind]).^2 for j=1:N]) #s2 = wsumsqdiff(w, xdata, xdata[ind], n)
              sy0 = sum(w .* ydata)
              sy1 = sum([w[j] * ydata[j] * (xdata[ind] - xdata[j]) for j=1:N]) #wsumyxdiff(w, xdata, xdata[ind], ydata, n)
              tmp += abs2((s2 * sy0 - s1 * sy1) /(s2 * s0 - s1 * s1) - ydata[ind])
              traceH += s0*w[ind]/(s2 * s0 - s1 * s1)
              ind += 1
        end
        tmp/n  + 2*(traceH+1)/(n-traceH-2)
    end
    function bwlocallinear(xdata::Vector{F64}, ydata::Vector{F64}, kernel::Function=gaussiankernel)
        n = length(xdata)
        length(ydata)==n || error("length(ydata) != length(xdata)")
        w = ones(n)
        if kernel == gaussiankernel
            h0 = bwnormal(xdata)
            hlb = 0.1 * h0
            hub = 10. * h0
        elseif kernel == betakernel
            h0 = midrange(xdata)
            hlb = h0 / n
            hub = 0.25
        elseif kernel == gammakernel
            h0 = midrange(xdata)
            hlb = h0/n
            hub = h0
        end
        Optim.minimizer(Optim.optimize(h -> AIClocallinear(xdata, ydata, kernel, h, w, n), hlb, hub))
    end

    function locallinear(xdata::Vector{F64}, ydata::Vector{F64};
                         xeval::Vector{F64}=xdata, kernel::Function=gaussiankernel,
                         h::Real=bwlocallinear(xdata, ydata, kernel))
        N = length(xdata)
        length(ydata) == N || error("length of ydata not the same with xdata")
        w = ones(N)
        pre = zeros(length(xeval))
        for i in 1:length(xeval)
            kernel(xeval[i], xdata, h, w, N)
            s0 = sum(w)
            s1 = s0*xeval[i] - sum(w .* xdata)
            s2 = sum([w[j]*(xdata[j]-xeval[i]).^2 for j=1:N]) #wsumsqdiff(w, xdata, xeval[i], n)
            sy0 = sum(w .* ydata)
            sy1 = sum([w[j] * ydata[j] * (xeval[i] - xdata[j]) for j=1:N]) #wsumyxdiff(w, xdata, xeval[i], ydata, n)
            pre[i] = (s2 * sy0 - s1 * sy1) /(s2 * s0 - s1 * s1)
        end
        pre
    end
    # locallinear(xdata::Vector{F64}, ydata::Vector{F64}, xeval::Real;
    #     kernel::Function = gaussiankernel, h::Real = bwlocallinear(xdata,ydata,kernel)) =
    # locallinear(xdata, ydata, xeval=[xeval;], kernel=kernel, h=h)


    # K(z::F64) = (abs(z) <= 1) ? 0.5 : 0.0#(2*pi)^(-0.5) * exp(-z^2 / 2)
    # # Set Bandwidth
    # h0 = (maximum(X) - minimum(X)) / N
    # # Normal kernel with bandwidth adjustment
    # K_h(z::F64; h::F64 = h0) = K(z / h) / h
    #
    # """
    # Local linear regression!
    # """
    # function m_hat(x0::F64; h::F64 = h0)
    #     @show "1"
    #     # Eq 2.4
    #     s_n(l::Int64) = sum(( K.( (x0 .- X) ./ h) .* (x0 .- X) .^ l))
    #     # Eq 2.3
    #     w = K.( (x0 .- X) ./ h) .* (s_n(2) .- (x0 .- X) .* s_n(1))
    #     # Eq 2.2
    #     return sum(w .* Y) / sum(w)
    # end

    x0_grid = collect(range(11.0; stop=15.0, length=N_grid))
    local_lin_reg = locallinear(X, Y, xeval = x0_grid)
    #npr(X, Y, xeval=X, B=N_bs, reg=locallinear, kernel=gaussiankernel)#m_hat.(x0_grid)
    return local_lin_reg

    for i=1:points

    end
    return out_line, lower, upper
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
function clean(df::DataFrame, col::Symbol;
               out_type::T = eltype(findfirst(!ismissing, df[:,col]))) where T<:Type
function clean(df::DataFrame, cols::Vector{Symbol}; out_type::Dict{Symbol,<:Type} = Dict())
function clean(df::DataFrame, cols::Vector{Symbol}, out_type::Type)
```
Variations on functions which convert data to consistent, usable, non-missing type.
"""
function clean(df::DataFrame, col::Symbol;
               out_type::Type = typeof(df[findfirst(!ismissing, df[:,col]), col]))
    # if (typeof(df[findfirst(!ismissing, df[:,col]), col]) == String) && (out_type != String)
    #     return transform!(df[.!ismissing.(df[:,col]), :], col => ByRow(s -> parse(out_type, s)) => col)
    # end
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
clean(df::DataFrame, t::Dict{Symbol,<:Type}) = clean(df, collect(keys(t)); out_type = t)
clean(df::DataFrame, cs::Vector{Symbol}, out_type::Type) = clean(df, cs;
                            out_type = Dict([cs .=> repeat([out_type], length(cs))]...))
