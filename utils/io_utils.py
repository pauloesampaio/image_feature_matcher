import yaml
import os
import requests
from PIL import Image
from io import BytesIO


def check_if_exists(path, create=True):
    """Checks if a path exists and, if wanted, creates it

    Args:
        path (str): Path to be checked
        create (bool, optional): If path doesn't exists, creates it or not. Defaults to True.

    Returns:
        Bool: Exists or not
    """
    if os.path.exists(path):
        return True
    elif create:
        os.mkdir(path)
        return True
    else:
        return False


def yaml_loader(yaml_path):
    """Loads yaml from a path

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        dict: Dict with yaml content
    """
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def download_image(image_url):
    """Downloads image from an url and returns PIL image

    Args:
        image_url (str): url of the desires image

    Returns:
        PIL Image: downloaded image
    """
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = Image.open(im_bytes)
    return image
