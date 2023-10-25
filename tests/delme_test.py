import pytest

from mdai_utils.delme import main


@pytest.fixture
def setup():
    print("setup")
    yield
    print("teardown")


def test_delme(setup):
    main()
    assert True
