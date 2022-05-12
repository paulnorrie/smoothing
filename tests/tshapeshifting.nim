import std/unittest
import arraymancer
import arraymancerExtras/shapeshifting

test "repeat":
  check [3,3,3,3].toTensor() == repeat([3].toTensor(), 4, axis = 0)
  
  let x = [[1,2],[3,4]].toTensor()
  
  check [[1,1,1,2,2,2], [3,3,3,4,4,4]].toTensor() == x.repeat(3, axis=1)
  
  check [[1,2], [1,2], [3,4], [3,4]].toTensor() == x.repeat(2, axis = 0)


test "repeat for 3D tensors":
  let c = [
    [ [0,1,2], [3,4,5] ],
    [ [6,7,8], [9,10,11] ]
  ].toTensor()

  let c0 = [
    [ [0,1,2], [3,4,5] ],
    [ [0,1,2], [3,4,5] ],
    [ [6,7,8], [9,10,11] ],
    [ [6,7,8], [9,10,11] ]
  ].toTensor()
  check c.repeat(2, axis = 0) == c0

  let c1 = [
    [ [0,1,2], [0,1,2], [3,4,5], [3,4,5] ],
    [ [6,7,8], [6,7,8], [9,10,11], [9,10,11] ]
  ].toTensor()
  check c.repeat(2, axis = 1) == c1

  let c2 = [
    [ [0,0,1,1,2,2], [3,3,4,4,5,5]],
    [ [6,6,7,7,8,8], [9,9,10,10,11,11]]
  ].toTensor()
  check c.repeat(2, axis = 2) == c2




test "tile 1d tensor along 1 axis":
  let a = [0, 1, 2].toTensor()
  let expected = [0, 1, 2, 0, 1, 2].toTensor()
  let actual = a.tile(2) 

  check expected == actual


test "tile 1d tensor along 2 axes":
  let a = [0, 1, 2].toTensor()
  let expected = [
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2]
  ].toTensor()
  let actual = a.tile(2, 2) 

  check expected == actual

test "tile 1d tensor along 3 axes":
  let a = [0, 1, 2].toTensor()
  let expected = [
    [
      [0, 1, 2, 0, 1, 2]
    ],
    [
      [0, 1, 2, 0, 1, 2]
    ]
  ].toTensor()
  let actual = a.tile(2, 1, 2)

  check expected == actual


test "tile 2d tensor along 2 axes":
  let a = [[1, 2], [3, 4]].toTensor()
  let expected = [[1, 2], [3, 4], [1, 2], [3, 4]].toTensor()
  let actual = a.tile(2,1)  
  check expected == actual


# fails when 2d tensor tile along the 2nd dimension greater than 1
test "tile 2d tensor along 1 axis":
  let a = [[1, 2], [3, 4]].toTensor()
  let expected = [[1, 2, 1, 2], [3, 4, 3, 4]].toTensor()
  let actual = a.tile(2)  # same as a.tile(1, 2)

  check expected == actual



test "tile 3d tensor along 3 axes":
  let a = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
  ].toTensor()
  
  let expected = [
    [[1,2,3,1,2,3], [4,5,6,4,5,6]],
    [[7,8,9,7,8,9], [10,11,12,10,11,12]]
  ].toTensor()

  check expected == a.tile(1,1,2)
