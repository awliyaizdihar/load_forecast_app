import os
import streamlit as st


def check_tft_model_exists(model_path="models/best_tft_no_weather.ckpt"):
    """
    Check whether the TFT checkpoint file exists.
    """
    return os.path.exists(model_path)


@st.cache_resource
def load_tft_model(model_path="models/best_tft_no_weather.ckpt"):
    """
    Load the trained TFT No-Weather model.

    This function requires pytorch-forecasting and the original model checkpoint.
    """
    try:
        from pytorch_forecasting import TemporalFusionTransformer

        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        model.eval()

        return model

    except ImportError as error:
        raise ImportError(
            "pytorch-forecasting is not installed. "
            "Install it only after the dashboard baseline version works."
        ) from error


def predict_tft_no_weather(model, input_data):
    """
    Placeholder wrapper for TFT prediction.

    Important:
    TFT prediction requires a TimeSeriesDataSet-compatible dataframe.
    This function cannot be finalized until the training dataset configuration
    and checkpoint structure are confirmed.
    """
    prediction = model.predict(input_data)
    return prediction