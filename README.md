# mdai-utils

Utility functions for MD.ai

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
