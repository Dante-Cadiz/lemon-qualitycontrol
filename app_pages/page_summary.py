import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Quality control is an essential part of preparation of fruits for "
        f"commercial retail that takes place in factories.\n"
        f"* This process is multi-faceted and typically involves chemically "
        f"testing samples of the produce, measurements of fruit diameter, "
        f" and visual examination of the fruits for illness and blemishes.\n"
        f"* These processes take place to satisfy local/national government "
        f"regulations regarding the quality and condition of produce, "
        f"typically using a 'class' based evaluation system.\n\n"
        f"**Project Dataset**\n"
        f"* The available dataset is a combination of two externally sourced "
        f"datasets of close-up images of single lemons that are labelled based"
        f" on their determined quality.\n"
        f"The production dataset for this project contains a subset of 3521 of"
        f" these images in total."
    )

    st.write(
        f"* For additional information (particularly regarding the dataset "
        f"and data preparation), please visit the [Project README file]"
        f"(https://github.com/Dante-Cadiz/lemon-qualitycontrol/blob/main"
        f"/README.md).")

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in analysing the visual difference "
        f"between good and poor quality lemons, specifically the visual "
        f"markers for defects that define a poor quality lemon.\n"
        f"* 2 - The client is interested in accurately and instantly "
        f" predicting from a given image whether a lemon is of good or poor "
        f"quality.")
