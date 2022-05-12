# Smoothing library for Nim
# Copyright (C) 2022  Paul Norrie

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Sheather and Jones Bandwidth estimation for a Kernel Density Estimator

import arraymancer
import arraymancerExtras/[fold_reduce, shapeshifting]
import std/math
import norm

  
proc phi6(x: Tensor) : Tensor =
  (x^.6.0 - 15.0 * x^.4.0 + 45.0 * x^.2.0 -. 15.0) *. gauss(x, 0.0, 1.0, norm = true)



proc phi4(x: Tensor) : Tensor =
  (x ^. 4 - 6.0 * x ^. 2 +. 3) *. gauss(x, 0.0, 1.0, norm = true)



proc sj[T](x: Tensor[T], h: float64) : float64 =
  ## Equation 12 of Sheather and Jones, i.e. calculating the bandwidth 
  ## selector
  
  let n = x.shape[0].float64
  let one = ones[float64](1, n.int)

  let lam = x.iqr()
  let a = 0.92 * lam * pow(n, -1 / 7.0) 
  let b = 0.912 * lam * pow(n, -1 / 9.0)

  var W = x.tile(n.int, 1).astype(float64)
  W = W - W.transpose()
  
  var W1 = phi6(W / b)
  var tdb = (squeeze(one * W1)).dot(squeeze(one))
  tdb = -tdb / ( (n * (n - 1)) * b.pow(7.0) )
  
  W1 = phi4(W / a)
  var sda = (squeeze(one * W1)).dot(squeeze(one))
  sda = sda / ( (n * (n - 1)) * a.pow(5.0))

  let alpha2 = 1.357 * (abs(sda / tdb)).pow(1 / 7.0) * h.pow(5 / 7.0)
  W1 = phi4(W / alpha2)
  var sdalpha2 = (squeeze(one * W1)).dot(squeeze(one))
  sdalpha2 = sdalpha2 / ((n * (n - 1)) * alpha2 ^ 5)

  let zero = [0'f64].toTensor()
  let sigma = sqrt(2.0)
  let d = gauss(zero, mean = 0.0, sigma, true)[0]
  result = (d / (n * abs(sdalpha2))).pow(0.2) - h




proc hsj*[T](x: Tensor[T]) : float64 =
  ## Sheather and Jones [1]_ bandwidth estimation.
  ## 
  ## Calculates the bandwidth to use for a Kernel Density Estimation on a
  ## distribution of values, `x`, which must be a vector (1-dimensional).
  ## 
  ## 
  ## References
  ##  ----------
  ##  .. [1] A reliable data-based bandwidth selection method for kernel
  ##      density estimation. Simon J. Sheather and Michael C. Jones.
  ##      Journal of the Royal Statistical Society, Series B. 1991
  
  assert x.rank == 1
  var h0 = hnorm(x)
  var v0 = sj(x, h0)

  var hstep = 1.1
  if v0 <= 0:
    hstep = 0.9

  var h1 = h0 * hstep
  var v1 = sj(x, h1)

  # converge
  while v1 * v0 > 0:
    h0 = h1
    v0 = v1
    h1 = h0 * hstep
    v1 = sj(x, h1)

  result = h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))

