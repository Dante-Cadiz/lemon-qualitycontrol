import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect that there would be a clear and significant visual difference " 
        f"noticeable between the average and variability images for each label, both in "
        f"colour and texture, with bad quality lemons presenting areas of discoloration "
        f"and contouring compared to smooth and plain yellow good quality lemon average images. "
        f" \n\n"
        f"* The generation of image montages for each label proved useful in confirming "
        f"our first hypothesis. The image montage of the good quality label produces lemons "
        f"that are mostly uniform in colour, shape, and skin condition."
        f"*An image montage of the bad quality label shows lemons with a variety of "
        f"clearly visible defects."
        f"* The data visualisation tasks of generating average images and image "
        f"variability plots, along with these images passed through Sobel filtering "
        f"to highlight dominant features was unsuccessful in presenting patterns "
        f"which could be used to generate insight."
        f"* This is likely due to the wide range of defects that fall under the umbrella "
        f"term of a 'bad quality' lemon. As such these mean images of the label do not highlight "
        f"any single visual marker distinctly. "
        f"* In further experimentation, this project may also be suited to a multi-class "
        f"classification but in its current iteration is limited by the breadth of the dataset."
        f" \n\n"
        f"* We also suspect that Providing a new lemon image and applying the project's generated" 
        f"binary classification model to it would allow the client to predict the likely quality "
        f"of a lemon to a high degree of accuracy."
        f"* This hypothesis has been confirmed via the development of a model that performs "
        f"highly in both overall F1 score (0.96) and recall on the bad quality label (0.98), "
        f"the two key metrics for evaluating the project's business functionality and success."
        f"* Further information about the model can be found on the ML performance metrics page."
    )