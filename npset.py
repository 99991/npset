import numpy as np

class NPSet:
    def __init__(self, capacity, values=None, dtype=np.int64, indices_dtype=np.int64):
        assert dtype in [np.int8, np.int16, np.int32, np.int64], f"Unsupported dtype {dtype}"
        self._indices = np.full(capacity, -1, indices_dtype)
        self._values = np.full(capacity, -1, dtype)
        self._size = 0

        if values is not None:
            self.update(values)

    def capacity(self):
        return self._indices.size

    def values(self):
        values = self._values[:self._size]
        values.setflags(write=False)
        return values

    def __iter__(self):
        yield from self._values[:self._size]

    def __len__(self):
        return self._size

    def add(self, value):
        if self._indices[value] != -1: return

        index = self._size
        self._indices[value] = index
        self._values[self._size] = value
        self._size += 1

    def update(self, values, deduplicate=True):
        values = np.asanyarray(values, dtype=self._values.dtype).ravel()

        if deduplicate:
            values = np.unique(values)

        indices = self._indices[values]

        values = values[indices == -1]

        new_size = self._size + values.size
        indices = np.arange(self._size, new_size)
        self._indices[values] = indices
        self._values[self._size:new_size] = values
        self._size = new_size

    def remove(self, value):
        index = self._indices[value]

        if index == -1:
            raise KeyError(f"Value {value} not found in set")

        # Swap-and-pop idiom, replace current value with last value
        replacement_value = self._values[self._size - 1]
        self._values[self._size - 1] = -1
        self._values[index] = replacement_value
        self._size -= 1

        # Update indices
        self._indices[replacement_value] = index
        self._indices[value] = -1

    def discard(self, value):
        try:
            self.remove(value)
        except KeyError:
            pass

    def difference_update(self, values, deduplicate=True):
        values = np.asanyarray(values, dtype=self._values.dtype).ravel()

        if deduplicate:
            values = np.unique(values)

        indices = self._indices[values]

        # Remove values that are not in the set
        valid = indices != -1
        values = values[valid]
        indices = indices[valid]

        cutoff = self._size - len(values)

        replacements = self._values[cutoff:self._size]

        # Replacements are values after cutoff not in values to be removed, as defined below:
        #
        # values_after_cutoff = set(values[indices >= cutoff])
        # replacements = [value for value in replacements if value not in values_after_cutoff]
        #
        # The following code does the same thing, but vectorized.
        # We filter replacements by temporarily using self._indices to store whether a value is to be removed.
        # The values after the cutoff contain exactly enough replacements
        # to override the removed values before the cutoff.

        values_after_cutoff = values[indices >= cutoff]

        # Backup original indices to restore later
        indices_backup = self._indices[values_after_cutoff]

        # Mark indices to be removed
        TO_BE_REMOVED = -2
        self._indices[values_after_cutoff] = TO_BE_REMOVED

        # Filter replacements to exclude values that will be removed
        replacements = replacements[self._indices[replacements] != TO_BE_REMOVED]

        # Restore original indices
        self._indices[values_after_cutoff] = indices_backup

        # Override indices before the cutoff with replacements
        indices_before_cutoff = indices[indices < cutoff]
        self._values[indices_before_cutoff] = replacements

        self._indices[replacements] = indices_before_cutoff
        self._indices[values] = -1

        self._size = cutoff

    def clear(self):
        self.indices[self.values()] = -1
        self._size = 0

    def contains(self, value):
        return self._indices[value] != -1

    def contains_all(self, values):
        return self.contains(values).all()

    def assert_compatible(self, other):
        assert self._values.dtype == other._values.dtype
        assert self._indices.dtype == other._indices.dtype
        assert self.capacity() == other.capacity(), f"Sets must have the same capacity, but have capacity {self.capacity()} and {other.capacity()}"

    def union(self, other):
        self.assert_compatible(other)

        new_set = self.copy()
        new_set.update(other.values())
        return new_set

    def difference(self, other):
        self.assert_compatible(other)

        new_set = self.copy()
        new_set.difference_update(other.values())
        return new_set

    def intersection(self, other):
        return self - (self - other)

    def copy(self):
        new_set = NPSet(
            len(self._indices),
            dtype=self._values.dtype,
            indices_dtype=self._indices.dtype)
        new_set._indices = self._indices.copy()
        new_set._values = self._values.copy()
        new_set._size = self._size
        return new_set

    def symmetric_difference(self, other):
        return (self - other) | (other - self)

    def issubset(self, other):
        self.assert_compatible(other)

        return other.contains_all(self.values())

    def __eq__(self, other):
        self.assert_compatible(other)

        return len(self) == len(other) and self.contains_all(other.values())

    def __str__(self):
        return "{" + ", ".join(str(x) for x in self._values[:self._size]) + "}"

    __repr__ = __str__
    __contains__ = contains
    __and__ = intersection
    __or__ = union
    __xor__ = symmetric_difference
    __sub__ = difference
