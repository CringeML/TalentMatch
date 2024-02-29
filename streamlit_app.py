import streamlit as st
import json
import numpy as np
import pandas as pd
from parser_my import *

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


st.set_page_config(page_title="Best resumes for this job", page_icon="👋", layout='wide')
file = st.sidebar.file_uploader("Upload your vacancy-resumes json here...", type=['json'])
print ('здорова')

if file:
    vacancy_resumes = json.load(file)

    st.write(f"## Вакансия: {vacancy_resumes['vacancy']['name']}")
    st.write(f"### Ключевые слова: {vacancy_resumes['vacancy']['keywords']}")
    st.write(f"Описание: {vacancy_resumes['vacancy']['description']}")

    st.write("## Резюме и их релевантность:")
    df_resumes = eval_json(vacancy_resumes)
    df_resumes_short = metrics_computation(df_resumes)[['first_name',
        'last_name', 'birth_date', 'country', 'city',
        'relevancy', 'uuid']]
    st.write(df_resumes_short)

    st.download_button(
        label="Download data as CSV",
        data=convert_df(df_resumes),
        file_name='resumes_relevancy.csv',
    )