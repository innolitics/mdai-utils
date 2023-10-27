import pytest

from mdai_utils.upload_annotations import upload_image_annotation_slice
from mdai_utils.upload_dataset import upload_dataset


def test_pytest_fixture(mdai_setup):
    mdai_parameters = mdai_setup["parameters"]
    mdai_project_id = mdai_parameters.get("mdai_project_id")
    assert mdai_project_id is not None


@pytest.mark.upload_only(
    reason="Only need to upload once. run pytest tests with --upload-only to run it."
)
def test_upload_dataset(mdai_setup):
    parameters = mdai_setup["parameters"]
    mdai_dataset_id = parameters.get("mdai_dataset_id")
    fixtures_dir = mdai_setup["fixtures_dir"]
    dicom_dir = fixtures_dir / "humanct_0002_1000_1004"
    assert dicom_dir.exists()
    completed_process = upload_dataset(mdai_dataset_id, dicom_dir)
    process_message = completed_process.stdout.strip()
    print(process_message)
    # Check the status of subprocess
    assert completed_process.returncode == 0


def test_upload_annotation(mdai_setup):
    parameters = mdai_setup["parameters"]
    fixtures_dir = mdai_setup["fixtures_dir"]
    mdai_client = mdai_setup["mdai_client"]
    # sop_instance_uid can be acquired from mdai, or from the metadata generated
    # by the function dicom_utils.read_dicoms_into_volume.
    sop_instance_uid = "1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414630768"
    mdai_label_ids = parameters.get("mdai_label_ids")
    labels_to_upload = parameters.get("labels")
    label_id = mdai_label_ids.get(labels_to_upload[0])

    failed_annotations = upload_image_annotation_slice(
        segmentation_image_path=fixtures_dir / "humanct_0002_1000_seg.nii.gz",
        sop_instance_uid=sop_instance_uid,
        mdai_client=mdai_client,
        mdai_project_id=parameters.get("mdai_project_id"),
        mdai_dataset_id=parameters.get("mdai_dataset_id"),
        mdai_label_id=label_id,
    )
    assert len(failed_annotations) == 0
