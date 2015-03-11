# fdmb
Toolbox for estimating parameters in stochastic linear systems using the EM-algorithm

This is software accompanying the publication [1,2]
   
- A numerically efficient implementation of the Expectation Maximization algorithm for state space
models. Applied Mathematics and Computation 241, 2014, 222-232

Since publication, the package has evolved into a shared C library with Python bindings. The
following functions are already ported

- arfit: Fit parameters of in a VAR model.
- emfit: Fit maximum likelihood parameters in the state space model (EM algorithm)
         -> Basic functionality ported

A package of the toolbox for Arch Linux is available from the AUR [3].
The source code as mentioned in the publication can be requested from the author [4].


[1] http://www.sciencedirect.com/science/article/pii/S0096300314006869

[2] http://webber.physik.uni-freiburg.de/~jeti/papers/1-s2.0-S0096300314006869-main.pdf

[3] https://aur.archlinux.org/packages/fdmb-git/

[4] mailto:Wolfgang.Mader@fdm.uni-freiburg.de