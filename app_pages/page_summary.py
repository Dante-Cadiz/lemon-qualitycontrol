import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Quality control is an essential part of preparation of fruits for commercial retail "
        f"that takes place in factories.\n"
        f"* This process is multi-faceted and typically involves chemically testing samples of "
        f"the produce, measurements of fruit shape, and visual examination of the fruits for illness "
        f"and blemishes.\n"
        f"* These processes take place to satisfy local/national government regulations regarding the "
        f"quality and condition of produce, typically using a 'class' based evaluation system.\n\n"
        f"**Project Dataset**\n"
        f"* The available dataset is a combination of two externally sourced datasets of close-up "
        f"images of single lemons that are labelled based on their determined quality.\n"
        f"The production dataset for this project contains a subset of 3521 of these images in total"
    )

    st.write(
        f"* For additional information (particularly regarding the dataset and data preparation), please visit the "
        f"[Project README file](https://github.com/Dante-Cadiz/lemon-qualitycontrol/blob/main/README.md).")
        
    
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in analysing the visual difference between good and poor quality "
        f"lemons, specifically the visual markers for defects that define a poor quality lemon.\n"
        f"* 2 - The client is interested in accurately and instantly predicting from a given image "
        f"whether a lemon is of good or poor quality"
        )