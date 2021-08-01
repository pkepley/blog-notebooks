# Infinite Chain Riddler

This week's
[Riddler](https://fivethirtyeight.com/features/will-riddler-nation-win-gold-in-archery/)
asks us to consider an infinitely long chain whose link lengths decrease as a
fixed proportion $f \in (0, 1)$ of each preceding link (i.e if $\ell_n$ is the
length of the $n^{th}$ link, then $\ell_{n+1} = f \cdot \ell_n).$ This chain
also has the property that whenever you adjust the angle $\theta$ between any
pair of successive links, then every other pair of sequential links will bend to
form the same angle (i.e. $\angle(\text{link}_n, \text{link}_{n+1}) = \theta$
for all $n$).

Since $f < 1$, the chain has a finite length, and the links tend toward a
limiting point. We will refer to this limit point as the chain's end point.
Assuming that the chain starts out in a straight line (i.e. $\theta = 0)$, we
are asked: what curve does the chain's end point sweep out as we vary $\theta$?
