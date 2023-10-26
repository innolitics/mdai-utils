import pytest

from mdai_utils.upload_dataset import upload_dataset


def test_pytest_fixture(mdai_setup):
    mdai_parameters = mdai_setup["parameters"]
    mdai_project_id = mdai_parameters.get("mdai_project_id")
    assert mdai_project_id is not None


@pytest.mark.upload_only(
    reason="Only need to upload once. run pytest tests with --upload-only to run it."
)
def test_upload_dataset(mdai_setup):
    mdai_parameters = mdai_setup["parameters"]
    mdai_dataset_id = mdai_parameters.get("mdai_dataset_id")
    fixture_dir = mdai_setup["fixture_dir"]
    dicom_dir = fixture_dir / "humanct_0002_1000_1004"
    assert fixture_dir.exists()
    completed_process = upload_dataset(mdai_dataset_id, dicom_dir)
    process_message = completed_process.stdout.strip()
    print(process_message)
    # Check the status of subprocess
    assert completed_process.returncode == 0


def test_upload_annotation(mdai_setup):
    mdai_setup["parameters"]
