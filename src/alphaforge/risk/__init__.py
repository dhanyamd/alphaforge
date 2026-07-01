"""(III) RISK MODEL — what the bet could cost you in volatility.

The alpha says what you hope to make; the risk model says what it could cost.
The key move (and the only reason the math is computable): describe a stock's
return as exposure to a handful of COMMON FACTORS plus its own specific piece,

        r_i = Σ_k  B_{i,k} f_k  +  u_i

so portfolio variance is  V = B Σ_f B' + D, where Σ_f is the K x K factor
covariance and D is the diagonal of specific variances. For 1,400 names you'd
need ~980,000 pairwise covariances; with ~65 factors you need ~2,000 numbers.
An impossible estimation problem becomes a tractable one.
"""

from alphaforge.risk.model import RiskModel, RiskModelSnapshot

__all__ = ["RiskModel", "RiskModelSnapshot"]
