import streamlit as st
import pandas as pd
import importlib
import os

pages = {
    'Home': 'pages.home',
    'Decision Tree': 'pages.decision_tree',
    'Naive Bayes': 'pages.naive_bayes',
}

page_names = tuple(pages.keys())

page_name = st.sidebar.selectbox("Select Visualisation", page_names, key='page_path')
page_path = pages[page_name]

page = importlib.import_module(page_path)
page.run(st)

