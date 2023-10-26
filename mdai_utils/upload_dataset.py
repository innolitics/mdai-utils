import subprocess


def upload_dataset(mdai_dataset_id, dir_path, order_exams_by="default"):
    """
    Upload a dicom dataset to MD.ai via CLI.

    A wrapper for the MD.ai CLI command:
    mdai dataset load --dataset-id <mdai_dataset_id> --order-exams-by <order_exams_by> <dir_path>

    Note There is no Python API for uploading a dicom (a dataset), so we use the cli.
    Ensure you have the mdai CLI installed: https://docs.md.ai/cli/installation/
    Args:
        mdai_dataset_id (str): The dataset id provided by MD.ai to upload to.
        dir_path (str): The path to the directory containing the dicom images.
        order_exams_by (str, optional): The order of the exams. Defaults to "default".
        Options: default, patient_id, study_date_time, study_desc, random
    Returns:
        subprocess.CompletedProcess: The result of the subprocess call.

    """
    # Check that mdai CLI is installed
    try:
        subprocess.run(["mdai", "version"], check=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "The mdai CLI is not installed. Please install it from https://docs.md.ai/cli/installation/"
        )
    command = [
        "mdai",
        "dataset",
        "load",
        "--dataset-id",
        f"{mdai_dataset_id}",
        "--order-exams-by",
        f"{order_exams_by}",
        f"{dir_path}",
    ]
    completed_process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        check=True,
    )
    if completed_process.returncode or completed_process.stderr:
        raise Exception(f"Error uploading dataset: {completed_process}.")
    return completed_process


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload a dicom dataset to MD.ai via CLI."
    )
    parser.add_argument(
        "-i",
        "--mdai_dataset_id",
        type=str,
        help="The dataset id provided by MD.ai to upload to.",
        required=True,
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        help="The path to the directory containing the dicom images.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--order-exams-by",
        type=str,
        help="The order of the exams. Defaults to 'default'. Options: default, patient_id, study_date_time, study_desc, random",
        default="default",
    )
    args = parser.parse_args()

    mdai_dataset_id = args.mdai_dataset_id
    dataset_path = args.dir_path

    completed_process = upload_dataset(mdai_dataset_id, dataset_path)
    print(f"To follow progress, run:\nmdai dataset progress -i {mdai_dataset_id}")
    if completed_process.returncode or completed_process.stderr:
        print(completed_process)
