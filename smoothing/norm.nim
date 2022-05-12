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

## Bandwidth for normal distribution
import arraymancer
import arraymancerExtras/fold_reduce
import std/math

proc wmean*[T: SomeNumber](x: Tensor[T], weights: Tensor[SomeNumber]) : float64 = 
  ## Weighted mean of a tensor.
  ## 
  ## Parameters:
  ## -----------
  ## `x` :      tensor to take the mean of
  ## 
  ## `weights` : non-negative weights for each element of `x`.  If `weights` 
  ##             does not have the same size as `x`, a `AssertionError` is raised.  
  
  assert weights.size() == x.size()
  let x = x.astype(float64)
  let weights = weights.astype(float64)
  result = sum(x *. weights) / sum(weights)



proc wvar*[T](x: Tensor[T], weights: Tensor[SomeNumber]) : float64 =
  ## Weighted variance of `x`
  ##
  ## Parameters:
  ## -----------
  ## `x` :      tensor to take the mean of
  ## 
  ## `weights` : non-negative weights for each element of `x`.  If `weights` 
  ##             does not have the same size as `x`, a `AssertionError` is raised.
  let x = x.astype(float64)
  let weights = weights.astype(float64)

  let meanDifferences = x -. x.wmean(weights)
  let squaredMeanDifferences = meanDifferences *. meanDifferences 
  var sumOfWeights = sum(weights)
  if sumOfWeights < 1:
    sumOfWeights = 2
  result = sum(weights *. squaredMeanDifferences) / (sumOfWeights - 1)



proc hnorm*[T,U: SomeNumber](x: Tensor[T], weights: Tensor[U]) : float64 =
  ## Evaluates the smoothing parameter (bandwidth) which is asymptotically optimal
  ## for estimating a density function when the underlying distribution is Normal.
  ## See paragraph 2.4.2 of
  ## Bowman and Azzalini[1]_ for details.
  ## 
  ## Parameters:
  ## -----------
  ## `x` :      1d tensor to evaluate bandwidth of
  ## 
  ## `weights` : an optional vector which allows the kernel functions over the 
  ##             observations to take different weights when they are averaged 
  ##             to produce a density estimate. This is useful, in particular, 
  ##             for censored data and to construct an estimate from binned data.
  ## 
  ##   References
  ##   ----------
  ##   .. [1] Applied Smoothing Techniques for Data Analysis: the
  ##       Kernel Approach with S-Plus Illustrations.
  ##       Bowman, A.W. and Azzalini, A. (1997).
  ##       Oxford University Press, Oxford

  # only works for 1d tensors
  assert x.rank <= 1

  let n = weights.sumToInt().float64  # sum returns T but if T is a byte, this will give an incorrect result

  result = 1.0
  if x.rank == 1:
    let sd = sqrt(x.wvar(weights))
    result = sd * pow(4 / (3 * n ), 1 / 5.0)



proc hnorm*[T](x: Tensor[T]) : float64 =
  ## Evaluates the smoothing parameter (bandwidth) which is asymptotically optimal
  ## for estimating a density function when the underlying distribution is Normal.
  ## See paragraph 2.4.2 of
  ## Bowman and Azzalini[1]_ for details.
  ## 
  ## Parameters:
  ## -----------
  ## `x` :      1d tensor to evaluate bandwidth of
  ## 
  ##   References
  ##   ----------
  ##   .. [1] Applied Smoothing Techniques for Data Analysis: the
  ##       Kernel Approach with S-Plus Illustrations.
  ##       Bowman, A.W. and Azzalini, A. (1997).
  ##       Oxford University Press, Oxford
  
  result = hnorm(x, ones[float64](x.shape[0]))

