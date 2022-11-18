import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect that  "
        f" \n\n"
        f"* An Image Montage "
        f"*The data visualisation tasks of generating average images, variability "
        f"and difference between average images revealed a number of trends, "
        f"some of which are more useful than others."
        f"The average images for bad quality lemons showed markings present in the "
        f"centre of the fruit that may be caused by the presence of blemishes or "
        f"disfigurations in the lemon images. These markings were not present in the "
        f"average good quality lemon images. However, contrary to the initial hypothesis, "
        f"there was no colour difference between average images for bad and good quality lemons, "
        f"with only a difference in image lighting that was present in the two datasets"
        f"any clear pattern to differentiate one to another."

    )