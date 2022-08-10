import pytest

from tests.common.utils import SimpleRandomModelPack


@pytest.fixture
def model_with_random_data():
    return SimpleRandomModelPack()
