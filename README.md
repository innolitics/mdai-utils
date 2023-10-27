# mdai-utils

Utility functions for MD.ai. Download and upload 2D and 3D segmentation images.

## Download data

- Download all data, dicom and annotations. Dicoms are only needed once,
or when data is added to the mdai dataset)

```bash
python -m mdai_utils.download_annotations \
--parameters myparameters.json \
-o ./data \
--download_dicoms
```

Once dicoms are downloaded, just download annotations, a json file will be generated in `./data`:

```bash
python -m mdai_utils.download_annotations \
 --parameters myparameters.json \
 -o ./data
```

## Upload 2D segmentations

```bash
python -m mdai_utils.upload_annotation_slice \
 --parameters ./tests/test_local_parameters.json \
 --sop_instance_uid "1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414630768" \
 --label_name mylabel \
 -i ./tests/fixtures/humanct_0002_1000_seg.nii.gz
```

## Upload 3D segmentations

MD.ai works with dicoms, and use the SOPInstanceUID as the key to match slices.
Your algorithm might work with 3D volumes, (.nrrd, .nii.gz, etc). You can convert
an input dicom_folder to a 3D volume, and also store the mapping between the new
volume indices and the original dicom file, with its SOPInstanceUID.

```bash
python -m mdai_utils.dicom_to_volume -i ./tests/fixtures/humanct_0002_1000_1004 -o /tmp/humanct_0002_1000_1004.nrrd
```

Parallel to the output image location, a `{image_filename}_SOPInstanceUIDs.json`
will be saved with the slice
mappings.

If we have a 3D volume segmentation we want to upload, use the mappings:

```bash
python -m mdai_utils.upload_annotation_volume \
 --parameters ./tests/test_local_parameters.json \
 --sop_instance_uids_file ./tests/fixtures/humanct_0002_1000_1004_SOPInstanceUIDs.json \
 --label_name mylabel \
 -i ./tests/fixtures/humanct_0002_1000_1004_seg.nii.gz
```

## Development

For information about building, running, and contributing to this code base,
please read the [CONTRIBUTING](CONTRIBUTING.md) page.

## Data

Sample data for testing acquired from:
[PROJECT: HumanCT  >  SUBJECT: VHFCT1mm-Pelvis  >  000-000-002](https://central.xnat.org/app/action/DisplayItemAction/search_element/xnat%3ActSessionData/search_field/xnat%3ActSessionData.ID/search_value/CENTRAL04_E04384/popup/false/project/HumanCT)
and manually segmented using [Slicer3D](https://www.slicer.org/).

## Tests

Create a copy of the test_parameters where you will provide the mdai ids needed
to upload and download the test data.

`cp tests/test_parameters.json tests/test_local_parameters.json`

We provide a tiny dataset that you can upload to a test dataset in your md.ai
instance. Only needed to run once: `pytest tests -rP --upload-only`.
