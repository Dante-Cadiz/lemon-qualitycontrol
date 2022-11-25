import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect that there would be a clear and significant visual "
        f"difference noticeable between the average and variability images "
        f"for each label, both in colour and texture, with bad quality lemons "
        f"presenting areas of discoloration and contouring compared to smooth "
        f"and plain yellow good quality lemon average images. \n"
        f"* The generation of image montages for each label proved useful in "
        f"confirming our first hypothesis. The image montage of the good "
        f"quality label produces lemons that are mostly uniform in colour, "
        f" shape, and skin condition.\n"
        f"* An image montage of the bad quality label shows lemons with a "
        f"variety of clearly visible defects.\n"
        f"* The data visualisation tasks of generating average images and "
        f"image variability plots, along with these images passed through "
        f"Sobel filtering to highlight dominant features was unsuccessful "
        f" in presenting patterns which could be used to generate insight.\n"
        f"* This is likely due to the wide range of defects that fall under "
        f"the umbrella term of a 'bad quality' lemon. As such these mean "
        f"images of the label do not highlight any single visual marker "
        f"distinctly. \n"
        f"* In further experimentation, this project may also be suited to a "
        f"multi-class classification but in its current iteration is limited "
        f"by the breadth of the dataset. \n"
        f"* We also suspected that providing a new lemon image and applying "
        f"the project's generated binary classification model to it would "
        f"allow the client to predict the likely quality "
        f"of a lemon to a high degree of accuracy.\n"
        f"* This hypothesis has been confirmed via the development of a model "
        f"that performs highly in both overall F1 score (0.96) and recall on "
        f"the bad quality label (0.98), the two key metrics for evaluating the"
        f"project's business functionality and success.\n"
        f"* Further information about the model can be found on "
        f"the ML performance metrics page.")
