import pandas as pd
import numpy as np
from utils.io_utils import yaml_loader, check_if_exists
from utils.model_utils import build_model, vectorize_dataframe, build_ann_index

config = yaml_loader("./config/config.yml")

if not check_if_exists(config["paths"]["vectors_path"], create=False):
    print(f'Vectors not found in {config["paths"]["vectors_path"]}')
    print(f'Vectorizing images from {config["paths"]["images_dataframe_path"]}')
    image_dataframe = pd.read_csv(config["paths"]["images_dataframe_path"])
    model = build_model(config)
    vectors = vectorize_dataframe(image_dataframe, model, config)
    np.save(config["paths"]["vectors_path"], vectors)
    print(f'Vectors saved to {config["paths"]["vectors_path"]}')
else:
    vectors = np.load(config["paths"]["vectors_path"])

ann_index = build_ann_index(vectors, config)
ann_index.saveIndex(config["paths"]["ann_index_path"])
print(f'Index saved to {config["paths"]["ann_index_path"]}')
