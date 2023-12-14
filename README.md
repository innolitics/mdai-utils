# mdai-utils

Utility functions for MD.ai. Download and upload 2D and 3D segmentation images.

## Download data

- Download all data: dicoms and annotations.

```bash
python -m mdai_utils.download_annotations \
--parameters myparameters.json \
-o ./data \
--download_dicoms
```

Dicoms will be downloaded with the default md.ai structure:

```md
- data/mdai_uab_project_L1NprBvP_images_dataset_D_odXMLm_2023-12-13-152228/
    - 1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421/
      - 1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453/
            1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414630768.dcm
            1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414645741.dcm
            1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414662833.dcm
            1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414677861.dcm
            1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414694890.dcm
```

If the option `--create_volumes` is added (to cli or the parameters file), a 3D
image will be generated in parallel to the `.dcm` files:

```md
            image.nii.gz
            volume_metadata.json
```

The annotations/segmentations from md.ai are stored in json file.

```md
./data/mdai_uab_project_L1NprBvP_annotations_dataset_D_odXMLm_labelgroup_G_2Jy2yZ_2023-12-13-152213.json
```
and the masks/images are stored in a folder:

```md
./data/mdai_uab_project_L1NprBvP_annotations_dataset_D_odXMLm_labelgroup_G_2Jy2yZ_2023-12-13-152213_segmentations_2023-12-14-114011/
```

with structure:

```md
mylabel__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414630768.nii.gz
mylabel__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414645741.nii.gz
mylabel__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414662833.nii.gz
mylabel__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414677861.nii.gz
mylabel__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286421__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610384286453__1.2.826.0.1.3680043.2.1125.1.75064541463040.2005072610414694890.nii.gz
pair_data.json
volumes/ # Only generated with --create_volumes option
```

---

Once dicoms are downloaded locally, and they are not changed in md.ai, do not
pass the --download_dicoms option to avoid re-downloading them.

## Upload 2D segmentations

To upload a single image or slice, we need its sop_instance_uid.

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
will be saved with the slice mappings.

This json will be used in `upload_annotation_volume`:

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


## Contact

![innolitis_logo_primaryw400](https://github.com/innolitics/mdai-utils/assets/3021667/6f9e269f-f96e-4b27-90c5-8fb48da70901)

You can contact us directly through our [website][contact_link].

If you find a bug or have suggestions for improvement, please open a
[GitHub issue][issue_link] or make a [pull request][pr_link].

[contact_link]: https://innolitics.com/about/contact/
[issue_link]: https://github.com/innolitics/mdai_utils/issues/new/choose
[pr_link]: https://github.com/innolitics/mdai_utils/pulls
