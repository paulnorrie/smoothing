# smoothing
Smoothing functions for Regression and Density Estimation

This is a Nim port of the [R _sm_ library](https://rdrr.io/cran/sm/) to use with Arraymancer.  If you wish to estimate a kernel bandwidth for non-normal distributions when using [the kde function](https://mratsim.github.io/Arraymancer/kde.html#kde%2CTensor%5BT%3A%20SomeNumber%5D%2CstaticKernelFunc%2Cfloat%2CU%2Cfloat%2Cfloat%2CTensor%5BT%3A%20SomeNumber%5D) in Arraymancer, there are some options:

- Use Silvermans rule of thumb, which Arraymancer handles

or use this library for:

- [x] Sheather and Jones method
- [x] Normal optimal choice
- [ ] Cross-validation (not yet done)

## Example

```nim
import arraymancer
import smoothing/sj

let x = randomTensor(1, 100, 255)

# x isn't likely to be normal, so use the kernel bandwidth given by the
# Sheather Jones method instead of Silvermans rule of thumb (the default)
let sjbw = hsj(x) 

# smooth our tensor
x.kde("gauss", adjust = 1.0, samples = 255, bw = sjbw, normalize = false)
```