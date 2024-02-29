import pandas as pd
import json
import numpy as np
from algorithm import algorithm


def parse_json_to_dict(data):
    parsed_data = json.loads(data)

    vacancy = parsed_data['vacancy']
    resumes = parsed_data['resumes']

    return {'vacancy': vacancy, 'resumes': resumes}


def eval_json(vacancy_resumes):
    vacancy = vacancy_resumes['vacancy']
    resumes = vacancy_resumes['resumes']
    df_resumes = pd.json_normalize(vacancy_resumes['resumes'])
    df_resumes['relevance'] = algorithm (vacancy, resumes)

    return df_resumes


def metrics_computation(resumes: pd.DataFrame):
    return resumes.sort_values(by='relevancy', ascending=False)