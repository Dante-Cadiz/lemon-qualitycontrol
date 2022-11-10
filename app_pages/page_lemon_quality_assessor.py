import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_lemon_quality_assessor_body():
    st.info(
        f"* The client is interested in accurately and instantly predicting from a given image "
        f"whether a lemon is of good or poor quality."
        )