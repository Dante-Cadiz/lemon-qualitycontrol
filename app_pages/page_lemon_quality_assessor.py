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
        f"* The client is interested in accurately and instantly predicting "
        f"from a given image whether a lemon is of good or poor quality."
        )

    st.write(
        f"* You can download the original two lemon datasets from the "
        f"following Kaggle pages: \n"
        f"* https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset \n"
        f"* https://www.kaggle.com/datasets/maciejadamiak/lemons-quality"
        f"-control-dataset \n"
        f"* Before applying the model to these images, you will need to clean "
        f"themusing this external helper application.\n"
        f"* Clean your images here - https://lemon-image-cleaner"
        f".herokuapp.com/ \n"
        f"* Once you have saved the cleaned images, upload them below. \n"
        f"* Alternatively, you can download the cleaned images from the GitHub"
        f" repo below: \n"
        f"https://github.com/Dante-Cadiz/lemon-qualitycontrol/tree/main/inputs"
        f"/lemon-quality-dataset/lemon_dataset"
        )

    st.write("---")

    images_buffer = st.file_uploader(f'Upload lemon images. You may select '
                                     f'more than one.',
                                     type=['png', 'jpg'],
                                     accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image)).convert('RGB')
            st.info(f"Lemon Image: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil,
                     caption=f"Image Size: {img_array.shape[1]}px width "
                             f"x {img_array.shape[0]}px height")

            version = 'v5'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img,
                                                            version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Name": image.name,
                                          'Result': pred_class},
                                         ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report),
                        unsafe_allow_html=True)
