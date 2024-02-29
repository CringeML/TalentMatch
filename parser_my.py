import pandas as pd
import json
import numpy as np
from algorithm import algorithm

def concat_vacancy(vacancy: dict) -> str:
    print (vacancy)
    name = vacancy.get('name', '')
    keywords = str(vacancy.get('keywords', ''))
    description = vacancy.get('description', '')
    comment = vacancy.get('comment', '')

    string = ' '.join(
        [
            name if name is not None else '',
            keywords if keywords is not None else '',
            description if description is not None else '',
            comment if comment is not None else '',
        ]
    )

    return string


def concat_resume(resume: dict):
    print (str(resume))
    resume_concated = []

    exp_item = resume.get('experienceItem', '')
    key_skills = resume.get('key_skills', '')

    if exp_item is not None:
        for desc in exp_item:
            description = desc.get('description', '')

            if description is not None:
                resume_concated.append(description)

    resume_concated = ' '.join(resume_concated) if len(resume_concated) else ''

    return key_skills, resume_concated


def parse_json_to_dict(data):
    parsed_data = json.loads(data)

    vacancy = parsed_data['vacancy']
    resumes = parsed_data['resumes']

    return {'vacancy': vacancy, 'resumes': resumes}


def eval_json(vacancy_resumes):
    vacancy = vacancy_resumes['vacancy']
    resumes = vacancy_resumes['resumes']
    df_resumes = pd.json_normalize(vacancy_resumes['resumes'])
    vacancy_str = concat_vacancy(vacancy)
    resumes_str = [concat_resume(i) for i in resumes]

    df_resumes['relevance'] = algorithm (vacancy_str, resumes_str)

    return df_resumes


def metrics_computation(resumes: pd.DataFrame):
    return resumes.sort_values(by='relevancy', ascending=False)