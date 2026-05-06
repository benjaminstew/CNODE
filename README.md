# CNODE  
This repo contains my implementation of:  
- the Neural ODE training algorithm outlined in the 2025 paper ["Training Neural ODEs Using Fully Discretized Simultaneous Optimization"](https://arxiv.org/abs/2502.15642) (Mariia Shapovalova, Calvin Tsay). It reframes Neural ODE training as a NLP based on a direct collocation scheme, which is solved using IPOPT. 
- an updated version of the algorithm using the integrated residual regularised direct collocation (IRR-DC) problem formulation outlined in the 2025 paper ["Reliable Solution to Dynamic Optimization
Problems using Integrated Residual Regularized
Direct Collocation"](https://arxiv.org/pdf/2503.09123) (Yuanbo Nie, Eric C. Kerrigan).  
- parallelised/batched Neural ODE training with the old-but-gold Alternating Direction Method of Multipliers (ADMM) algorithm introduced in the 1976 paper ["A DUAL ALGORITHM FOR THE SOLUTION OF
NONLINEAR VARIATIONAL PROBLEMS VIA FINITE ELEMENT APPROXIMATION"](https://www.sciencedirect.com/science/article/pii/0898122176900031?via%3Dihub) (Daniel Gabay, Bertrand Mercier) and outlined in a [textbook](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) (Stephen Boyd et al.).

The Neural ODE model is trained on synthetic Van der Pol Oscillator dataset (src/dataloaders/vdpo_loader), an experimental catechol-based chemical reaction dataset (data/catechol_single_solvent) and a synthetic cell culture batch reaction dataset (data/bioreactor_sim_by_noise). 

