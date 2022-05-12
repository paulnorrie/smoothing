## Higher order Fold and Reduce and aggregate functions that return integers
## 
## e.g. sum of Tensor[byte] can exceed 255
import arraymancer, arraymancer/tensor/backend/openmp

template reduce_inline_int*[T](t: Tensor[T], op: untyped): untyped =
  ## same as reduce_inline but returns integer instead of T
  let z = t # ensure that if t is the result of a function it is not called multiple times
  var reduced: int
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, z.size, 1, op) do:
    x = z.atContiguousIndex(block_offset).int
  do:
    for y {.inject.} in z.items(block_offset, block_size):
      op
  reduced


proc sumToInt*[T](t: Tensor[T]): int =
  ## Compute the sum of all elements
  t.reduce_inline_int():
    x+=y.int
