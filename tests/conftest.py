import json
import os
from pathlib import Path

import mdai
import pytest

_current_dir = Path(__file__).parent if "__file__" in locals() else Path(".")


@pytest.fixture(scope="session")
def mdai_setup(parameters_file=_current_dir / "test_parameters.json", input_token=None):
    token = input_token or os.getenv("MDAI_TOKEN", "")
    if token == "":
        raise ValueError(
            "Please set the MDAI_TOKEN environment variable with your MDAI credentials."
        )
    with open(parameters_file) as f:
        parameters = json.load(f)

    mdai_domain = parameters.get("mdai_domain") or "md.ai"
    mdai_client = mdai.Client(domain=mdai_domain, access_token=token)
    return {"mdai_client": mdai_client, "parameters": parameters}
