import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_lemons_visualiser import page_lemons_visualiser_body
from app_pages.page_lemon_quality_assessor import page_lemon_quality_assessor_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name= "Lemon Quality Control") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Lemons Visualiser", page_lemons_visualiser_body)
app.add_page("Lemon Quality Assessor", page_lemon_quality_assessor_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()