import unittest
import smoothing/sj
import arraymancer

test "hsj":
  let x = [1,2,4,8,4,2,1,8,16,12].toTensor() # not normal
  let expected = 1.9747
  check expected == round(hsj(x), 4)
        