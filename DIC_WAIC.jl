# functions to compute Deviance Information Criterion and Watanabe-Akaike Information Criterion based on the 
# log-pdf[S, N] matrix, where S is the number of samples and N is the number of observations.

function DIC_logpfd(logpdf_mat)
    Dev = -2 .* sum(logpdf_mat, dims=2)
    DIC = mean(Dev) + var(Dev)/2
    return DIC
end

function WAIC_logpfd(logpdf_mat)
    lppd = sum(log.(mean(exp.(logpdf_mat), dims = 1)))
    pWAIC1 = 2 * sum(log.(mean(exp.(logpdf_mat), dims=1))  - mean(exp.(logpdf_mat), dims = 1))
    pWAIC2 = sum(var(logpdf_mat, dims=1))
    WAIC = -2 * lppd + 2* pWAIC2
    return WAIC
end
