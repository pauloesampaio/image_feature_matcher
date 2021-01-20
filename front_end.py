import pandas as pd
import numpy as np
import streamlit as st
from utils.io_utils import yaml_loader, download_image
from utils.model_utils import (
    load_ann_index,
    query_ann_index,
    vectorize_image,
    build_model,
)
from PIL import Image


@st.cache
def ann_index_loader(config):
    ann_index = load_ann_index(config)
    return ann_index


@st.cache
def images_dataframe_loader(config):
    images_dataframe = pd.read_csv(config["paths"]["images_dataframe_path"])
    return images_dataframe


@st.cache
def vectors_loader(config):
    vectors = np.load(config["paths"]["vectors_path"])
    return vectors


@st.cache
def vectorizer_loader(config):
    model = build_model(config)
    return model


config = yaml_loader("./config/config.yml")
ann_index = ann_index_loader(config)
vectorizer_model = vectorizer_loader(config)
images_dataframe = images_dataframe_loader(config)
vectors = vectors_loader(config)

st.write(
    """
# Feature vector matcher
## Enter the image url
"""
)
url = st.text_input("Image url")

if url:
    image = download_image(url)
    query_vector = vectorize_image(vectorizer_model, image, config)
    results = query_ann_index(query_vector, ann_index, config)
    images = []
    captions = []
    for match_index, match_distance in zip(results["matches"], results["distances"]):
        images.append(
            Image.open(
                images_dataframe.iloc[match_index][config["dataframe"]["path_field"]]
            )
        )
        captions.append(f"Dist: {match_distance:.2f}")
    col1, col2 = st.beta_columns(2)
    col1.write("# Original:")
    col1.image(image=image, use_column_width=True)
    col2.write("# Matches")
    col2.image(image=images, width=128, caption=captions)
