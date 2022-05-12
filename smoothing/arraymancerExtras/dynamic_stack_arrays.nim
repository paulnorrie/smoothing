import arraymancer


iterator zip*[T, U](a: DynamicStackArray[T], b: openarray[U]): (T, U) =
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])
