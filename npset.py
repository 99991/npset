import numpy as np


class NPSet:
    def __init__(
        self,
        capacity,
        values=None,
        dtype=np.int64,
        indices_dtype=np.int64,
    ):
        assert dtype in [np.int8, np.int16, np.int32, np.int64]

        self._indices = np.full(capacity, -1, indices_dtype)
        self._values = np.full(capacity, -1, dtype)
        self._size = 0

        if values is not None:
            self.update(values)

    @property
    def capacity(self):
        return self._indices.size

    def values(self):
        values = self._values[:self._size]
        values.setflags(write=False)
        return values

    def __iter__(self):
        return iter(self._values[:self._size])

    def __len__(self):
        return self._size

    def clear(self):
        self._indices[self.values()] = -1
        self._size = 0

    def contains(self, value):
        return self._indices[value] != -1

    __contains__ = contains

    def contains_any(self, values):
        return self.contains(values).any()

    def contains_all(self, values):
        return self.contains(values).all()

    def is_compatible(self, other):
        if not isinstance(other, NPSet): return False
        if self._values.dtype != other._values.dtype: return False
        if self._indices.dtype != other._indices.dtype: return False
        if self.capacity != other.capacity: return False
        return True

    def __eq__(self, other):
        assert self.is_compatible(other)
        return len(self) == len(other) and self.contains_all(other.values())

    def issubset(self, other):
        assert self.is_compatible(other)
        return other.contains_all(self.values())

    def issuperset(self, other):
        assert self.is_compatible(other)
        return self.contains_all(other.values())

    def isdisjoint(self, other):
        assert self.is_compatible(other)
        return not self.contains_any(other.values())

    def add(self, value):
        if self._indices[value] != -1: return

        index = self._size
        self._indices[value] = index
        self._values[self._size] = value
        self._size += 1

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

    def pop(self):
        if self._size == 0:
            raise KeyError("Set is empty")

        # Pop the last value
        value = self._values[self._size - 1]

        self._indices[value] = -1
        self._values[self._size - 1] = -1
        self._size -= 1

        return value

    def _deduplicate(self, values, deduplicate=True):
        # TODO implement update functions optimized for NPSet
        if isinstance(values, NPSet):
            assert self.is_compatible(values)
            return values.values()

        values = np.asanyarray(values, dtype=self._values.dtype).ravel()

        if deduplicate:
            values = np.unique(values)

        return values

    def update(self, values, deduplicate=True):
        values = self._deduplicate(values, deduplicate)

        indices = self._indices[values]

        values = values[indices == -1]

        new_size = self._size + values.size
        indices = np.arange(self._size, new_size)
        self._indices[values] = indices
        self._values[self._size:new_size] = values
        self._size = new_size
        return self

    def difference_update(self, values, deduplicate=True):
        values = self._deduplicate(values, deduplicate)

        indices = self._indices[values]

        # Remove values that are not in the set
        valid = indices != -1
        values = values[valid]
        indices = indices[valid]

        cutoff = self._size - len(values)

        replacements = self._values[cutoff:self._size]

        # Replacements are values after cutoff not in values to be removed,
        # as defined below:
        #
        # values_after_cutoff = set(values[indices >= cutoff])
        # replacements = [
        #     value for value in replacements
        #     if value not in values_after_cutoff]
        #
        # The following code does the same thing, but vectorized. We filter
        # replacements by temporarily using self._indices to store whether a
        # value is to be removed. The values after the cutoff contain exactly
        # enough replacements to override the removed values before the cutoff.

        values_after_cutoff = values[indices >= cutoff]

        # Backup original indices to restore later
        indices_backup = self._indices[values_after_cutoff]

        # Mark indices to be removed with -2
        self._indices[values_after_cutoff] = -2

        # Filter replacements to exclude values that will be removed
        replacements = replacements[self._indices[replacements] != -2]

        # Restore original indices
        self._indices[values_after_cutoff] = indices_backup

        # Override indices before the cutoff with replacements
        indices_before_cutoff = indices[indices < cutoff]
        self._values[indices_before_cutoff] = replacements

        self._indices[replacements] = indices_before_cutoff
        self._indices[values] = -1

        self._size = cutoff

        return self

    def intersection_update(self, values, deduplicate=True):
        values = self._deduplicate(values, deduplicate)

        # Remove values that are not in the set
        indices = self._indices[values]
        valid = indices != -1
        values = values[valid]
        indices = indices[valid]

        # Delete old indices
        self._indices[self.values()] = -1

        # Move common values and indices to the front
        self._indices[values] = np.arange(len(values))
        self._values[:len(values)] = values
        self._size = len(values)

        return self

    def symmetric_difference_update(self, values, deduplicate=True):
        values = self._deduplicate(values, deduplicate)

        common_values = values[self._indices[values] != -1]

        # a ^ b = (a | b) - (a & b)
        self.update(values, deduplicate=False)
        self.difference_update(common_values, deduplicate=False)

        return self

    __ior__ = update
    __isub__ = difference_update
    __iand__ = intersection_update
    __ixor__ = symmetric_difference_update

    def copy(self):
        new_set = NPSet(
            capacity=self.capacity,
            dtype=self._values.dtype,
            indices_dtype=self._indices.dtype)

        new_set._indices = self._indices.copy()
        new_set._values = self._values.copy()
        new_set._size = self._size

        return new_set

    def union(self, other):
        result = self.copy()
        result |= other
        return result

    def difference(self, other):
        result = self.copy()
        result -= other
        return result

    def intersection(self, other):
        result = self.copy()
        result &= other
        return result

    def symmetric_difference(self, other):
        result = self.copy()
        result ^= other
        return result

    __or__ = union
    __sub__ = difference
    __and__ = intersection
    __xor__ = symmetric_difference

    def __str__(self):
        return "{" + ", ".join(str(x) for x in self._values[:self._size]) + "}"

    __repr__ = __str__
