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
    
    st.write(
        f"* Link for the lemon images will go here "
        f"Figure out how to handle the data cleaning step"
        )

    st.write("---")

    images_buffer = st.file_uploader('Upload lemon images. You may select more than one.',
                                        type='png',accept_multiple_files=True)
   
    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Lemon Image: **{image.name}**")
            img_array = np.array(img_pil) # add reshape here to :3
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v4'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Name":image.name, 'Result': pred_class },
                                        ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

# This is the error that this function is currently throwing
# raise ValueError( ValueError: Input 0 of layer sequential_1 is incompatible with the layer: expected axis -1 of input shape to have value 3 but received input with shape (None, 100, 100, 4)
# The 4 is a RGBA value but the model expects RGB?