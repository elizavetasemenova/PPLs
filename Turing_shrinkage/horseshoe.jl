#----------------------------
#  Horseshoe
#----------------------------

# include libraries
using Distributions
using GLM
using Turing
using StatsFuns: logistic, logsumexp
using CPUTime

# simulate fake data
n=100
p=4
X = rand(Float64, (n,p))
beta=[2.0 .^ (-i) for i in 0:(p-1)]
alpha=0
sigma=0.7
eps=rand(Normal(0, sigma), n)
y = alpha .+ X * beta + eps;
#lm(X, y)


# define Turing model
@model tmodel(X, y, ::Type{T}=Vector{Float64}) where {T} = begin

    n,p = size(X)

    beta = T(undef, p)
    lambda = T(undef, p)

    alpha ~ Normal(0,1)
    sigma ~ Truncated(Cauchy(0,1),0,Inf)
    tau ~ Truncated(Cauchy(0, 1), 0, +Inf)

    for i = 1:p
        lambda[i] ~ Truncated(Cauchy(0, 1), 0, +Inf)
        beta[i] ~ Normal(0, tau * lambda[i])
    end

    mu = alpha .+ X * beta

    y ~ MvNormal(mu, sigma)

end

# run inference and time it
steps = 2000
@time @CPUtime chain = sample(tmodel(X,y), NUTS(steps, 0.65))
