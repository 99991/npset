from npset import NPSet
import numpy as np

def test():
    a = NPSet(10, [1, 2, 3])
    b = NPSet(10, [2, 3, 4])

    assert a != b
    assert not (a == b)

    assert a & b == NPSet(10, [2, 3])
    assert a | b == NPSet(10, [4, 3, 2, 1])
    assert a ^ b == NPSet(10, [4, 1])

    assert 2 in a
    assert 2 not in (a ^ b)
    assert (a - b).issubset(a)

    np.random.seed(0)

    n = 10

    s = NPSet(n)

    values = [0, 2, 1, 6, 3, 5, 9, 7, 4]

    s.update(values)

    values = [9, 0, 5, 7, 2, 6]

    s.difference_update(values)

    assert (s._indices != -1).sum() == s._size

    for i, value in enumerate(s):
        assert s._indices[value] == i

    assert set(s.values()) == {1, 3, 4}

def test_random():
    np.random.seed(0)

    n = 100

    s = NPSet(n)
    s_expected = set()

    for _ in range(1000):
        which = np.random.randint(4)

        if which == 0:
            value = np.random.randint(n)

            if value not in s_expected:
                s.add(value)
                s_expected.add(value)

        elif which == 1:
            if len(s_expected) > 0:
                value = np.random.choice(s.values())

                s.remove(value)
                s_expected.remove(value)

        elif which == 2:
            count = np.random.randint(10)
            values = np.random.randint(n, size=count)
            values = list(set(values) - s_expected)

            s.update(values)
            s_expected.update(values)

        elif which == 3:
            count = min(np.random.randint(10), len(s_expected))
            values = np.random.choice(list(s_expected), size=count, replace=False)
            values = values.astype(np.int64)

            s.difference_update(values)
            s_expected.difference_update(values)

        assert len(s) == len(s_expected)
        assert set(s) == s_expected

def test_large():
    m = 100
    n = 10**5

    a = np.random.randint(m, size=n)
    b = np.random.randint(m, size=n)

    s = NPSet(m, a) & NPSet(m, b)

    assert np.allclose(np.sort(s.values()), np.intersect1d(a, b))


if __name__ == "__main__":
    test()
    test_random()
    test_large()
