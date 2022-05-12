# smoothing
Smoothing functions for Regression and Density Estimation

This is a Nim port of the [R _sm_ library](https://rdrr.io/cran/sm/) to use with Arraymancer.  If you wish to estimate a kernel bandwidth for non-normal distributions when using [the kde function](https://mratsim.github.io/Arraymancer/kde.html#kde%2CTensor%5BT%3A%20SomeNumber%5D%2CstaticKernelFunc%2Cfloat%2CU%2Cfloat%2Cfloat%2CTensor%5BT%3A%20SomeNumber%5D) in Arraymancer, there are some options:

- Use Silvermans rule of thumb, which Arraymancer handles
- [x] Sheather and Jones method in this library
- [x] Normal optimal choice in this library
- [ ] Cross-validation (not yet done)
