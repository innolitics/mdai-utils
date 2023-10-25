# from mdai_utils.upload import upload_data_annotation_slice


def test_upload_mask_annotation(mdai_setup):
    mdai_parameters = mdai_setup["parameters"]
    mdai_project_id = mdai_parameters.get("mdai_project_id")
    assert mdai_project_id is not None
