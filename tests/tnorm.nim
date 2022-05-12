import unittest
import smoothing/norm
import arraymancer

test "wmean with 2d unweighted tensors":
  let x = [[1,1], [2,2]].toTensor()
  let weights = [[1,1], [1,1]].toTensor()
  check 1.5 == x.wmean(weights)

test "wmean with weighted tensors":
  let x = [6,6,6].toTensor()
  let weights = [0.5, 1.5, 2.0].toTensor() # (3 + 9 + 12) / 4
  check 6 == x.wmean(weights)


test "wvar":
  let x = [1,2,4,8,4,2,1].toTensor()
  let weights = [1,1,1,1,1,1,1].toTensor()
  check 6.1429 == round(x.wvar(weights), 4)


test "hnorm":
  let x = [1,2,4,8,4,2,1].toTensor()
  check 1.7789 == round(x.hnorm(), 4)