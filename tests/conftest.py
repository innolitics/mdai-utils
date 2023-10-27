import json
from pathlib import Path

import mdai
import pytest

from mdai_utils.common import get_mdai_access_token


def pytest_addoption(parser):
    """Add a command line option to pytest."""
    parser.addoption(
        "--upload-only",
        action="store_true",
        default=False,
        help="run only tests marked as upload_only",
    )


# Register the custom mark to avoid pytest warnings
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "upload_only: Mark test to run only when --upload-only is provided"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Hook that runs before a test is set up."""
    is_upload_only_test = "upload_only" in item.keywords
    upload_only_mode = item.config.getoption("--upload-only")

    if upload_only_mode:
        if not is_upload_only_test:
            pytest.skip("Skipping non-upload tests in upload-only mode")
    else:
        if is_upload_only_test:
            pytest.skip("Skipping upload-only tests in standard mode")


_current_dir = Path(__file__).parent


@pytest.fixture(scope="session")
def mdai_setup(
    parameters_file=_current_dir / "test_local_parameters.json", input_token=None
):
    # Check if the parameters file exists
    print(f"Parameters file: {parameters_file}")
    if not parameters_file.exists():
        existing_parameters_file = _current_dir / "test_parameters.json"
        raise FileNotFoundError(
            f"Parameters file {parameters_file} not found. Please create one, using {existing_parameters_file} as a template."
        )
    token = input_token or get_mdai_access_token()
    with open(parameters_file) as f:
        parameters = json.load(f)

    fixtures_dir = _current_dir / "fixtures"
    mdai_domain = parameters.get("mdai_domain") or "md.ai"
    mdai_client = mdai.Client(domain=mdai_domain, access_token=token)
    return {
        "mdai_client": mdai_client,
        "parameters": parameters,
        "fixtures_dir": fixtures_dir,
    }
