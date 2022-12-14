import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_lemons_visualiser_body():
    st.write("### Lemons Visualizer")
    st.info(
        f"* The client is interested in analysing the visual difference "
        f"between good and poor quality lemons, specifically the visual "
        f"markers for defects that define a poor quality lemon.")

    version = 'v5'
    if st.checkbox("Difference between average and variability image"):

        avg_bad_quality = plt.imread(
                            f"outputs/{version}/avg_var_bad_quality.png")
        avg_good_quality = plt.imread(
                            f"outputs/{version}/avg_var_good_quality.png")

        st.warning(
          f"* We notice the average and variability images didn't show "
          f"patterns where we could intuitively differentiate one from "
          f"another.\n"
          f"* Applying Sobel edge detection filters to the images was also "
          f"unsuccessful in illustrating a clear visual divide between "
          f"these average and variability images.")

        st.image(avg_bad_quality,
                 caption='Bad Quality Lemon - Average and Variability')
        st.image(avg_good_quality,
                 caption='Good Quality Lemon - Average and Variability')
        st.write("---")

    if st.checkbox(
            "Differences between average bad quality and good quality lemons"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"* We notice this study didn't show patterns "
            f" where we could intuitively differentiate one to another.\n"
            f"* While there is an area of difference shown on the left of "
            f"the lemon, this was hypothesised to be due to a difference "
            f"in photographic lighting between the two image datasets used, "
            f"one of which contained a large majority of defective lemons. "
            f"This difference will not affect the machine learning model's "
            f"performance, as the ImageDataGenerator task contains flipping "
            f"and rotation of images.")
        st.image(diff_between_avgs,
                 caption='Difference between average images')

    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on 'Create Montage' button")
        my_data_dir = 'inputs/lemon-quality-dataset/lemon_dataset'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select label", options=labels,
                                        index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(dir_path+'/' + label_to_display)
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            print(
                  f"Decrease nrows or ncols to create your montage. \n"
                  f"There are {len(images_list)} in your subset. "
                  f"You requested a montage with {nrows * ncols} spaces")
            return

        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows*ncols):
            img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
              f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
        plt.tight_layout()

        st.pyplot(fig=fig)

    else:
        print("The label you selected doesn't exist.")
        print(f"The existing options are: {labels}")
