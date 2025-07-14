from npset import npset
import numpy as np

def test_random_single():
    # Test operations that act on a single value
    np.random.seed(0)

    m = 10
    s = npset(m)
    s_expected = set()

    for _ in range(1000):
        which = np.random.randint(4)

        if which == 0:
            value = np.random.randint(m)

            s.add(value)
            s_expected.add(value)

        elif which == 1:
            if s_expected:
                value = np.random.choice(s.values())

                s.remove(value)
                s_expected.remove(value)

        elif which == 2:
            if s:
                value = s.pop()
                s_expected.remove(value)

        elif which == 3:
            value = np.random.randint(m)

            s.discard(value)
            s_expected.discard(value)

        assert sorted(s) == sorted(s_expected)

def test_random_inplace():
    # Test inplace operations
    m = 10
    s = npset(m)
    s_expected = set()

    for _ in range(1000):
        count = np.random.randint(m)
        values = np.random.randint(m, size=count)

        if np.random.randint(2):
            values = npset(m, values)

        which = np.random.randint(4)

        if which == 0:
            s.update(values)
            s_expected.update(values)
        elif which == 1:
            s.difference_update(values)
            s_expected.difference_update(values)
        elif which == 2:
            s.intersection_update(values)
            s_expected.intersection_update(values)
        elif which == 3:
            s.symmetric_difference_update(values)
            s_expected.symmetric_difference_update(values)

        assert sorted(s) == sorted(s_expected)

def test_large():
    m = 100
    n = 10**5

    a = np.random.randint(m, size=n)
    b = np.random.randint(m, size=n)

    s = npset(m, a)
    t = npset(m, b)

    assert np.allclose(np.sort((s & t).values()), np.intersect1d(a, b))
    assert np.allclose(np.sort((s | t).values()), np.union1d(a, b))
    assert np.allclose(np.sort((s - t).values()), np.setdiff1d(a, b))
    assert np.allclose(np.sort((s ^ t).values()), np.setxor1d(a, b))

def test_misc():
    m = 10
    a = npset(m, [0, 1, 2, 3])
    b = npset(m, [2, 3, 4, 5])

    assert str(b) == "{2, 3, 4, 5}"
    assert a.is_compatible(b)
    assert len(a) == 4
    assert a.capacity == m
    assert list(b) == [2, 3, 4, 5]
    assert a != b
    assert not (a == b)
    assert 2 in a
    assert 2 not in (a ^ b)
    assert (a - b).issubset(a)
    assert (b - a).issubset(b)
    assert (a | b).issuperset(a)
    assert (a | b).issuperset(b)
    assert (a ^ b).isdisjoint(a & b)

    a.clear()

    assert list(a) == []

if __name__ == "__main__":
    test_misc()
    test_random_single()
    test_random_inplace()
    test_large()
