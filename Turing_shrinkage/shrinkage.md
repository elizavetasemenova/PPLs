## Shrinkage and sparcity-inducing priors in Turing.

Often we need to estimate a set of coefficients $\beta$ which determine some functional relationship between a set of
inputs $x_i$ and a target variable $y$. Bayesian priors are known to have regularizing effect which helps to prevent over-fitting. Sparcity-inducing priors help to perform shrinkgage on $\beta$. In this article we demonstrate how some of them can be implemented and used in Turing. A range of both continuous and discrete shrinkage priors is being widerly adopted in the literature. However, not all probabilistic programming languages can handle both groups. Due to its flexibility, Turing allows both continuous and discrete parameters, and, hence, prios of both kinds can be implemented.

Continuous:
- Normal,
- Laplace,
- Student’s t,
- horseshoe,
- Finninsh horseshoe,

Discrete:
- spike-and-slab.



### Horseshoe and Finnish horseshoe

The horseshoe prior, initially proposed by Carvalho et al (2009), is symmetric around zero with fat tails and an infinitely large spike at zero. The prior is well suited for sparce models, where only a few of many regression coefficients are non-zero.

The original hosseshoe prior is driven by a global shrinkage parameter $\tau,$ and local parameters $\lambda_i:$

\begin{align}
\beta_i &\sim N(0, \tau \lambda_i), \\
\lambda_i &\sim C^+(0,1),\\
\tau &\sim C^+(0,\tau_0).
\end{align}


Piironen and Vehtari (2017) suggest an additional regularization of the non-zero coefficients via the Finnish horseshoe:

\begin{align}
\beta_i &\sim N(0, \tau \widetilde{\lambda}_i), \\
\widetilde{\lambda}_i &= \frac{c \lambda_i}{\sqrt(c^2 + \tau^2 \lambda^2_i)}, \\
\lambda_i &\sim C^+(0,1),\\
c^2 &\sim \text{InvGamma}(\nu/2, \nu/2s^2),\\
\tau &\sim C^+(0,\tau_0).
\end{align}


## References

- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2009). Handling sparsity via the horseshoe. In International Conference on Artificial Intelligence and Statistics (pp. 73-80).

- Piironen J. & Vehtari A. (2016). On the Hyperprior Choice for the Global Shrinkage Parameter in the Horseshoe Prior. https://arxiv.org/pdf/1610.05559v1.pdf

- Piironen, J., and Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. https://arxiv.org/abs/1707.01694

- R. B. O'Hara and M. J. Sillanpää (2009), A review of Bayesian variable selection methods: what, how and which, Bayesian Analysis, Volume 4, Number 1, 85-117.
