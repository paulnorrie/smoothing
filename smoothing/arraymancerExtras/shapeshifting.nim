import arraymancer
import std/[algorithm, sequtils]
import dynamic_stack_arrays


proc size(m: Metadata, first: int, last: int = -1) : int =
  ## The number of elements in a shapes sub-dimensions from first to last (inclusive)
  ## if `last < 0` then size to last dimension
  let newLast = if last < 0: m.len - 1 else: last 

  assert newLast < m.len, "Index " & $newLast & " out of range 0.." & $(m.len - 1)
  
  result = 1
  if newLast >= first and first >= 0:
    for i in first .. newLast:
      result *= m[i]



proc repeat*[T](t: Tensor[T], repeats: Positive, axis: Natural) : Tensor[T] =
  ## Repeat each element along an axis. 
  ## 
  ## Parameters:
  ## -----------
  ## `t` :      Repeat elements in this tensor.  Must be C Contiguous.
  ## 
  ## `repeats`: Number of times each element will be repeated.  A value of 1
  ##            indicates no repeats (as there is 1 element already).  
  ##          
  ## `axis`:    The axis to repeat along.  
  ## 
  ## Returns
  ## -------
  ## Tensor of the same shape except on the `axis` dimension, which will be
  ## `axis * repeats`.
  runnableExamples:
    assert @[3,3,3,3] == repeat(3, 4).toSeq1D()
    let x = [[1,2],[3,4]].toTensor()
    assert @[1,1,2,2,3,3,4,4] == x.repeat(2).toSeq1D()
    assert @[[1,1,1,2,2,2], [3,3,3,4,4,4]] == x.repeat(3, axis=1)
    assert @[[1,2], [1,2], [3,4], [3,4]] == x.repeat(2)

  assert axis < t.rank, "given axis " & $axis & 
                        " does not exist in tensor of " & $t.rank & " dimensions "

  # create destination tensor 
  var newShape = t.shape
  newShape[axis] *= repeats
  result = newTensorUninit[T](newShape)

  # c contigous means we can copy as many elements as 
  # the multiple of the subsequent dimension sizes
  # e.g. t.shape = (10, 5, 3) and axis = 0 => copy 5*3 = 15 elements at time
  assert t.isCContiguous(), "Expected C Contiguous Tensor"
  var chunkSize = 1
  if axis + 1 < t.shape.len:
    chunkSize = t.shape.size(axis + 1)

  var n_outer = 1
  if axis - 1 > -1:
    n_outer = t.shape.size(0, axis - 1)

  let axisLen = t.shape[axis]

  var src = 0
  var dst = 0
  
  # TODO: //ise but the inner loops modify src and dst
  for i in 0 ..< n_outer:
    for j in 0 ..< axisLen: # each source set of elements (chunkSize)
      for k in 0 ..< repeats: # copy chunkSize k times
        when T is KnownSupportsCopyMem:
          let srcPtr = t.unsafe_raw_buf[src].addr()
          let dstPtr = result.unsafe_raw_buf[dst].addr()
        else:
          let srcPtr = t.storage.raw_buffer[src].addr()
          let dstPtr = result.storage.raw_buffer[dst].addr()
            
        copyMem(dstPtr, srcPtr, chunkSize * sizeof(T))
        dst += chunkSize 
        
      src += chunkSize
  


proc prepend[T](vector: openarray[T], value: T, n: int) : seq[int] {.inline.} = 
  ## Creates a sequence containing `n` x `value` prepended to `vector`.  
  ## 
  ## e.g.
  ## ```
  ## let vector = @[20, 21]
  ## assert @[1, 1, 20, 21] == vector.prepend(1, 2)
  ## ```
  if n > 0:
    result.setLen(n + vector.len) 
    result.fill(0, n - 1, value)
    result[n..<result.len] = vector
  else:
    result = @vector



proc tile*[T](t: Tensor[T], reps: varargs[int]) : Tensor[T] = 
  ## Construct a new Tensor by repeating `t` the number of times
  ## given by `reps`.
  ## 
  ## Parameters:
  ## -----------
  ## `t` :      Tile elements in this tensor.  Must be C Contiguous.
  ## 
  ## `reps`:    How many times to repeat elements along each dimension in `t`.  
  ##            E.g. `2,3` repeats the first dimension and the second dimension
  ##            thrice.   If `reps` is shorter than the dimensions of `t`, then
  ##            it is treated as the first dimensions not being repeated.
  ##            e.g. if `t.rank == 3` and `reps = 2,2` then
  ##            reps is treated as `1,2,2`. 
  ##            If `reps` is longer than the dimensions of `t`, then `t` is 
  ##            treated as having additional dimensions of shape 1 prepended.
  ##          
  ## 
  ## Returns
  ## -------
  ## Tiled tensor
  
  # result to same dimensions as reps by prepending dimensions of size 1
  result = t
  let promotedShape = prepend(@(result.shape), 1, reps.len - result.rank)
  result = result.reshape(promotedShape)

  # reps to have same dimensions 
  var tup = @reps
  if reps.len < result.rank:
    tup = tup.prepend(1, result.rank - reps.len)

  # new shape
  var shape_out: seq[int]
  for s, t in zip(result.shape, tup):
    shape_out.add(s * t)

  var n = result.size.int

  if n > 0:
    let origShape = result.shape
    for dim_in, nrep in zip(origShape, tup):
      if nrep != 1:
        let firstDim = result.size div n
        result = result.reshape(firstDim ,n)
        result = result.repeat(nrep, 0)

      n = n div dim_in

  result = result.reshape(shape_out)
  