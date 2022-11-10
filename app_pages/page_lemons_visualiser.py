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
        f"*The client is interested in analysing the visual difference between good and poor quality "
        f"lemons, specifically the visual markers for defects that define a poor quality lemon.")