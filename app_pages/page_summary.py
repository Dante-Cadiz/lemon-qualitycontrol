import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
    )

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Dante-Cadiz/lemon-qualitycontrol/blob/main/README.md).")
        
    
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in analysing the visual difference between good and poor quality "
        f"lemons, specifically the visual markers for defects that define a poor quality lemon.\n"
        f"* 2 - The client is interested in accurately and instantly predicting from a given image "
        f"whether a lemon is of good or poor quality"
        )