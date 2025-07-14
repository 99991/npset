# npset
Vectorized NumPy implementation of a sparse set data structure.
Supports most operations that `set()` supports, in addition to vectorized update
functions.

# Example

```python
from npset import npset
import numpy as np

# All values added to the set must be non-negative and smaller than `limit`.
limit = 1000000

s = npset(limit)

# Add one million random values
s |= np.random.randint(limit, size=1000000)

# Remove 1000 random values
s -= np.random.randint(limit, size=1000000)

# As you might have noticed, this went very fast. Much faster than set()

# Another example with smaller values:
s = npset(limit, [1, 2, 3, 4])
t = npset(limit, [3, 4, 5, 6])

# {1, 2, 5, 6}
print(s ^ t)

s.add(0)

# {1, 2, 3, 4, 0}
print(s)

# Remove last value from set
t.pop()

# {3, 4, 5}
print(t)

# [True, True, False] (we just removed 6 with .pop())

print(t.contains([4, 5, 6]))

```
