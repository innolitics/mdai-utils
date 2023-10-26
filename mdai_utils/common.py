import os


def get_mdai_access_token(env_variable: str = "MDAI_TOKEN") -> str:
    token = os.getenv(env_variable, "")
    if token == "":
        raise ValueError(
            f"Please set the {env_variable} environment variable with your MDAI credentials."
        )
    return token
