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

#### Why should I use this over something like `numpy.union1d`?

<table>
    <tr>
        <td>
            <img width="1280" height="960" alt="cumulative_time" src="https://github.com/user-attachments/assets/15cead4e-a855-44ab-93e4-ee1b5ff085aa" />
        </td>
        <td>
            <img width="1280" height="960" alt="time" src="https://github.com/user-attachments/assets/89a3dbee-d579-49d4-a1b4-1a1b3cd57c1d" />
        </td>
    </tr>
</table>

When values are repeatedly added to a set with [`numpy.union1d`](https://numpy.org/doc/2.1/reference/generated/numpy.union1d.html), the computational complexity will be quadratic, because a new array is created every time. The following code adds 1000 chunks with 1000 random values each to a set and takes about 5 seconds.

```python
import numpy as np

limit = 10**6
values = []
for _ in range(1000):
    new_values = np.random.randint(limit, size=1000)
    values = np.union1d(values, new_values)
```

In contrast, `npset` does not have that flaw. Adding (or removing) $k$ values has a computational complexity of $O(k)$, regardless of the size of the set. The same code with `npset` takes about 50 milliseconds.

```python
import numpy as np
from npset import npset

limit = 10**6
values = npset(limit)
for _ in range(1000):
    new_values = np.random.randint(limit, size=1000)
    values |= new_values
```

#### When should I *NOT* use `npset`?

* When your values have large or unlimited range. In this case, a sparse set would waste a lot of memory.
* When your values are not integers. Sparse sets rely on indexing, which is difficult with data types such as floats.
