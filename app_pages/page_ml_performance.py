import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v5'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    confusion_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption="Confusion Matrix", width=500)

    st.write(f"* The overall accuracy of 0.9646, as well as a recall score "
             f"on the negative outcome of a bad quality lemon of 0.98 indicate that "
             f"this machine learning model is successful in detrmining the quality of a "
             f"lemon, and conforms with the success metrics as defined in the ML " 
             f"business requirements of an F1 score of 0.95 or higher and a recall on "
             f"bad quality lemons of 0.98 or higher"
    )