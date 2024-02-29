import pandas as pd
import numpy as np
import nltk
from string import punctuation
from deep_translator import GoogleTranslator
from pymystem3 import Mystem
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
punctuation = list(punctuation)
m = Mystem()
tf_idf_vect = load(r'C:\Users\vsevo\PycharmProjects\pythonProject6\TalentMatch\tfidf.joblib')


def remove_html_tags(text):
    text = text.replace('&nbsp', " ")
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# Перевод текста на русский язык
def translate_text(text):
    translator = GoogleTranslator(source='auto', target='ru')
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:

        return text


# Разбиение на чанки для перевода
def translate_chunked(text, chunk_size=4999):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = [translate_text(chunk) for chunk in chunks]

    return ''.join(translated_chunks)


# Нормализация текста
def normalize_text(s):
    # Лемматизация
    lemms = m.lemmatize(s)
    stopwords = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('russian')
    lemms = [token for token in lemms if token not in stopwords \
             and token != " " \
             and token.strip() not in punctuation]

    # удаляем стоп-слова из нашего текста
    words_without_stop = [i for i in lemms if i not in stopwords]
    # Вывод значения в строке
    total = ''
    for el in words_without_stop:
        total += el
        total += ' '

    return total


def get_vacancy_key_words(s: str) -> str:
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {
                "role": "user",
                "content": f'Выпиши ключевые компетенции и технологический стек из следующей вакансии: {s}'
            }
        ],
        "temperature": 1,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.oqEaHFqkfNn1hnUzjJ-9JCYHpH_RXFS9oNCCdA0WKwLkkR0huvBbTKE5UjdK5HRiv4zdMiPg0SPSBXFCUEl63A-d5o9kgsdiZz_qWS8sVWqTLTVZBnYmgut_eYbZoMMK1fj1Ugv8ELRhmhb8vk4Gxe6FnhIYpcBd-d6bhwS9KJIcE2xQjDgc7nG6GqY4rPIrn3x_MyqgZCOQq-2U6P77Zy01fcAewuWR0yLV8FEm-sGxtZQXUupaC3Cyy3EESMVvloxK_hg66u_USAzE3SCQUbLj0KsN9qqjUMauWJ4QkLXOQZfTPTTdYRATDETaJg1_LfjUhBGjMgBiCDEiJTgCGw.OdUiYqad1NUtnCos5DgLyw.eqYwHnMRhH3uGYYGQgbVjdyf0bA8HujWXd-m4lFXeInTcyduzxYDjAJIPvIerc1SZHP0lJNuxJFeeGglhOCcuGM47PV8bzb6rLxmjBvGHkodCEBoArCVRNVVQggphwGtg_qO3qDQbYOM6nXyJOBP9dmnCCGJpv1k3KeF1dJ0KAAfICSme-a1qngbG0f0U47kU8RVwKrz1--WQseuJ0kvqyOoTlLjGifH5VDMNiPq26NBDWJ-yZ5cPatDibwXdXPTsYgkbBwL0ZJ9R1i1XjwVy9hiykl3ZRO1aTov1kejgEr3xnZ-jMNf-knFHfyjxs3d9sDARRHHZZOeUlh6Lv89ZGCxIQnqUKc-31DooAOKaKXGirDJj5CIsOb_oi2OuebBkXE7ekgcyJ1YckH4ir1Zor5B_O---te7fCvVniK5dxALjO69t_1ZFOW5KmE8QmNiPY9ibGWPBfwHASEnWBcFjWoxryjmxbzpBRR2PF192DoA73s-wdgPVpnlM80nFmxmR3DsNtM6AMtZisnMF7GR3LVDxfWZknUVGHbWYMBN-ZMHYMY4lhqIe7r8whwyzCvHSi32YJnsFIfBj7k4gn4gqaJbQwELtpta1jHykHyVaPpuAp95swxzUh9HsYMO9hfXOg7fbKYTjl-h3O7IvEjxhuspYNa3nWrDTCtMaX4ZYJXPMoaqIL27bF9nvXxUcVMYXtzSgn8xtfMJB-2qN4bYW63Dnz6ASWJPWiteMg3WTJo.9DE8cGdC5FPzKXj1-23uTvw9aLixdKEp6VeD4qS6tGM'}

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    # status_code == 200 - код успешного запроса
    if response.status_code == 200:
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        # Очистить от спец символов и букв в случае плохого ответа модели; -> int
        return content
    else:
        # Если запрос не успешен, вывести сообщение об ошибке
        print("Ошибка при выполнении запроса:", response.status_code)
        return s

def text_cos_sim():

        return

def tf_idf_cos_sim(vac: str,
                       resume: str,
                       vectorizer=tf_idf_vect):

        vac = normalize_text(remove_html_tags(translate_chunked(vac)))
        vac_vect = vectorizer.transform([vac])
        resume = normalize_text(remove_html_tags(translate_chunked(' '.join(resume))))
        resume_vect = vectorizer.transform([resume])
        cosin_sim = cosine_similarity(vac_vect[0], resume_vect[0])

        return cosin_sim

def cv_cos_sim(vac: str,
                   resume: str):

        vac = normalize_text(remove_html_tags(vac))
        cv_vectorizer = CountVectorizer(binary=True).fit([vac])
        vac_vect = cv_vectorizer.transform([vac])
        resume = normalize_text(remove_html_tags(translate_chunked(' '.join(resume))))
        resume_vect = cv_vectorizer.transform([resume])
        cosin_sim = cosine_similarity(vac_vect[0], resume_vect[0])

        return cosin_sim

def agregated_cos_sim(array_text, array_tf_idf, array_cv):

        array_text = normalize([array_text], norm="l1")[0]
        array_tf_idf = normalize([array_tf_idf], norm="l1")[0]
        array_cv = normalize([array_cv], norm="l1")[0]

        array_itog = [(i1 + i2 + i3) / 3 for i1, i2, i3 in zip(array_text, array_tf_idf, array_cv)]
        return array_itog

def algorithm (vacancy_str, resumes_str):
        array_text = []
        array_tf_idf = []
        array_cv = []

        vacancy_str_for_cv = get_vacancy_key_words (vacancy_str)

        for resume in resumes_str:
            array_text.append(text_cos_sim(vacancy_str, resume))
            array_tf_idf.append(tf_idf_cos_sim(vacancy_str, resume))
            array_cv.append(cv_cos_sim(vacancy_str_for_cv, resume))

        res = agregated_cos_sim (array_text, array_tf_idf, array_cv)
        return res

