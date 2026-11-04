import pytest

from calc import add, divide


def test_add():
    assert add(2, 3) == 5
    # RED RUN: change the line above to `assert add(2, 3) == 6`, push, watch it fail,
    # then change it back and push again for a GREEN run.


def test_divide():
    assert divide(9, 3) == 3
    with pytest.raises(ValueError):
        divide(1, 0)
