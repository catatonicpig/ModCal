from contextlib import contextmanager
import pytest


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (3, does_not_raise()),
        (2, does_not_raise()),
        (1, does_not_raise()),
        (0, pytest.raises(ZeroDivisionError)),
    ],
)
def test_division(example_input, expectation):
    """Test how much I know division."""
    with expectation:
        assert (6 / example_input) is not None