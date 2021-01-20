import nmslib
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(config):
    """From a trained keras model, get vector result from an intermediate layer,
    according to configuration file.

    Args:
        config (dict): Configuration dictionary with model path and which layer to
        get vector input from.

    Returns:
        keras.model: Vectorizer model
    """
    model = load_model(config["paths"]["vectorizer_model_path"])
    model = Model(
        inputs=model.input,
        outputs=[model.get_layer(config["model"]["layer_to_get_input"]).input],
    )
    return model


def vectorize_dataframe(dataframe, model, config):
    """Given a dataframe with image pathes and a vectorizer model,
    vectorize all images from the dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe with image paths
        model (keras.model): Vectorizer model
        config (dict): Configuration dictionary

    Returns:
        np.array: Array with vectorized images
    """
    image_dataset = ImageDataGenerator()
    image_generator = image_dataset.flow_from_dataframe(
        dataframe=dataframe,
        x_col=config["dataframe"]["path_field"],
        y_col=None,
        target_size=config["model"]["input_size"],
        class_mode=None,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
    )
    vectors = model.predict(image_generator, verbose=1)
    return vectors


def build_ann_index(vectors, config):
    """Given a set of vectors, uses nmslib to build an approximate nearest
    neighbors index.

    Args:
        vectors (np.array): Vectors array
        config (dict): Configuration dictionary

    Returns:
        nmslib.index: approximate nearest neighbors index
    """
    ann_index = nmslib.init(**config["ann_index"]["init_time"])
    ann_index.addDataPointBatch(vectors)
    ann_index.createIndex(config["ann_index"]["index_time"], print_progress=True)
    return ann_index


def load_ann_index(config):
    """Loads a pre-fitted approximate nearest neighbors index.

    Args:
        config (dict): Configuration dictionary

    Returns:
        nmslib.index: pre-fitted approximate nearest neighbors
    """
    ann_index = nmslib.init(**config["ann_index"]["init_time"])
    ann_index.loadIndex(config["paths"]["ann_index_path"])
    ann_index.setQueryTimeParams(config["ann_index"]["query_time"])
    return ann_index


def query_ann_index(query_vector, ann_index, config):
    """Given a vector, queries the ann index to get best matches

    Args:
        query_vector (np.array): Vectorized image
        ann_index (nmslib.index): Approximate nearest neighbors index
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary with matches and distances
    """
    ids, distances = ann_index.knnQuery(query_vector, k=config["ann_index"]["k"])
    return {"matches": ids, "distances": distances}


def vectorize_image(model, image, config):
    """Given an image and a vectorizer model, gives the vectorized version of the image

    Args:
        model (keras.model): Vectorizer model
        image (PIL.Image): Image to vectorize
        config (dict): Configuration dictionary

    Returns:
        np.array: Vectorized image
    """
    image = np.array(image.resize(config["model"]["input_size"]))
    model_input = image.reshape(
        [
            1,
        ]
        + config["model"]["input_size"]
        + [
            3,
        ]
    )
    vector = model.predict(model_input)
    return vector
