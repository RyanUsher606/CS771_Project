import os
import yaml

def load_config(config_path):
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Loaded configuration as a dictionary.
    """
    # Ensure the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Dynamically resolve relative paths to absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key, path in config.get("paths", {}).items():
        config["paths"][key] = os.path.join(project_root, path)

    return config
