import pytest

pytest.importorskip("homeassistant")

# Home Assistant mounts the harness's testing_config, which has its own
# custom_components package, first on sys.path. Import the repository's package
# first so that the loader finds custom_components/betterybalance.
import custom_components  # noqa: E402, F401


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(recorder_mock, enable_custom_integrations):
    """Give every test a recorder (a manifest dependency) and the custom integrations."""
    yield
