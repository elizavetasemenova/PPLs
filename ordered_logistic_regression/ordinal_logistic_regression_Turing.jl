
using CSV
using DataFrames
using Turing
using Plots
using StatsBase
using StatsFuns: logistic
using MLBase

df = CSV.read("WVS.csv")
deletecols!(df, :Column1)
y = convert(Array, df[:y])
X = convert(Matrix, df[setdiff(names(df), [:y])]);
println(typeof(X))
println(typeof(y))
println(countmap(y))

struct OrderedLogistic{T1, T2} <: DiscreteUnivariateDistribution
   η::T1
   cutpoints::Vector{T2}
end

function Distributions.logpdf(d::OrderedLogistic, k::Int)
   
    K = length(d.cutpoints)+1

    c =  d.cutpoints
    
    if k==1
        logp= log(logistic(c[k]-d.η))
    elseif k<K
        logp= log(logistic(c[k]-d.η) - logistic(c[k-1]-d.η))
    else
        logp= log(1-logistic(c[k-1]-d.η))
    end
    
    return(logp)
end

### Turing model
@model m(X, y) = begin

    D = size(X, 2)

    # priors
    sigma ~ TruncatedNormal(0,1,0,Inf)
    
    c1 ~ Normal(0, 20)
    log_diff_c ~ Normal(0, 2)
    c2 = c1 + exp(log_diff_c)
    c = [c1, c2]
    
    beta ~ MvNormal(zeros(D), sigma * ones(D))

    lp = X * beta

    # likelihood
    for i = 1:length(y)
        y[i] ~ OrderedLogistic(lp[i], c)
    end
end

steps = 10000
#chain = sample(m(X, y), NUTS(steps, 0.65));
chain = sample(m(X, y), HMC(steps, 5e-3, 5));

show(chain)

e_log_diff_c = exp.(chain[:log_diff_c].value.data)[:,1,1]
c1_est = chain[:c1].value.data[:,1,1]
c2_est = c1_est + e_log_diff_c;
println(mean(c1_est))
println(mean(c2_est))

function Distributions.rand(d::OrderedLogistic)
    cutpoints = d.cutpoints
    η = d.η  
    
    if !issorted(cutpoints)
        error("cutpoints are not sorted")
    end

    K = length(cutpoints)+1
    c = vcat(-Inf, cutpoints, Inf)
    l = [i for i in zip(c[1:(end-1)],c[2:end])]
    ps = [logistic(η - l[i][1]) - logistic(η - l[i][2]) for i in 1:K]
    k = findall(ps.== maximum(ps))[1]
    
    if all(ps.>0)
        return(k)
    else
        return(-Inf)
    end
end

beta_est = chain[:beta].value.data[:,:,1]';
lp_post = X * beta_est;

y_pred_samps = zeros(size(lp_post));

for i in 1:size(y_pred_samps,1)
    for j in 1:size(y_pred_samps,2)
        
        c1 = c1_est[j,1,1]
        c2 = c2_est[j,1,1]
        c = [c1, c2]
        
        dist = OrderedLogistic(lp_post[i,j], c)
        
        y_pred_samps[i,j] = rand(dist)
    end
end

y_pred = zeros(size(y_pred_samps, 1));

for i in 1:length(y_pred)
    
    p1 = mean(y_pred_samps[i,:] .== 1)
    p2 = mean(y_pred_samps[i,:] .== 2)
    p3 = mean(y_pred_samps[i,:] .== 3)
    probs = [p1, p2, p3]
    
    y_pred[i] = sum((probs .== maximum(probs)) .* [1, 2, 3])
end

y_pred = convert(Array{Int64,1}, y_pred);
countmap(y_pred)

mean(y .== y_pred)

C = confusmat(3, y, y_pred)


