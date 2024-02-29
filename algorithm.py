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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import string
from transformers import AutoTokenizer, AutoModel, \
    BartForConditionalGeneration, BartTokenizer
import torch

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
punctuation = list(punctuation)
m = Mystem()
tf_idf_vect = load(r'C:\Users\vsevo\PycharmProjects\pythonProject6\TalentMatch\tfidf.joblib')
printable = set(string.printable)

bart_tokenizer = BartTokenizer.from_pretrained(
    "Ameer05/bart-large-cnn-samsum-rescom-finetuned-resume-summarizer-10-epoch")
bart_model = BartForConditionalGeneration.from_pretrained(
    "Ameer05/bart-large-cnn-samsum-rescom-finetuned-resume-summarizer-10-epoch")

hrbert_tokenizer = AutoTokenizer.from_pretrained("RabotaRu/HRBert-mini", model_max_length=512)
hrbert_model = AutoModel.from_pretrained("RabotaRu/HRBert-mini")


def concat_vacancy(vacancy: dict) -> str:

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

    resume_concated = []

    exp_item = resume.get('experienceItem', '')
    key_skills = str(resume.get('key_skills', '')) + str(resume.get('about', ''))

    if exp_item is not None:
        for desc in exp_item:
            description = desc.get('description', '')

            if description is not None:
                resume_concated.append(description)

    resume_concated = ' '.join(resume_concated) if len(resume_concated) else ''

    return (key_skills, resume_concated)


def remove_html_tags(text):
    text = text.replace('&nbsp', " ")
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_non_ascii_crazyML(text):
    return ''.join(i for i in text if ord(i) < 128)


def translate_text_crazyML(text: str, lang: str) -> str:
    translator = GoogleTranslator(source='auto', target=lang)
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:

        return text


def translate_chunked_crazyML(text: str, lang: str, chunk_size: int = 4999) -> str:
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = [translate_text_crazyML(chunk, lang) for chunk in chunks]

    return ''.join(translated_chunks)


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
    chunks = [text[i1:i1 + chunk_size] for i1 in range(0, len(text), chunk_size)]
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


def text_cos_sim(vac: str,
                 resume: str,
                 model=hrbert_model,
                 tokenizer=hrbert_tokenizer) -> float:
    inputs_vac = tokenizer(vac, return_tensors="pt", truncation=True)
    outputs_vac = model(**inputs_vac)[1][0]

    inputs_resume = tokenizer(resume, return_tensors="pt", truncation=True)
    outputs_resume = model(**inputs_resume)[1][0]

    cosine_res_vac = torch.dot(outputs_resume, outputs_vac) \
                     / (torch.norm(outputs_resume) * torch.norm(outputs_vac))
    cosine_vac_res = torch.dot(outputs_vac, outputs_resume) \
                     / (torch.norm(outputs_vac) * torch.norm(outputs_resume))

    cosin_mean_val = torch.mean(
        torch.tensor([cosine_res_vac, cosine_vac_res])
    ).item()

    return cosin_mean_val


def tf_idf_cos_sim(vac: str,
                   resume: str,
                   vectorizer=tf_idf_vect):
    vac = normalize_text(remove_html_tags(translate_chunked(vac)))
    vac_vect = vectorizer.transform([vac])
    resume = normalize_text(remove_html_tags((' '.join(resume))))
    resume_vect = vectorizer.transform([resume])
    cosin_sim = cosine_similarity(vac_vect[0], resume_vect[0])

    print (cosin_sim)

    return cosin_sim[0][0]


def cv_cos_sim(vac: str,
               resume: str):
    vac = normalize_text(remove_html_tags(vac))
    cv_vectorizer = CountVectorizer(binary=True).fit([vac])
    vac_vect = cv_vectorizer.transform([vac])
    resume = normalize_text(remove_html_tags(translate_chunked(' '.join(resume))))
    resume_vect = cv_vectorizer.transform([resume])
    cosin_sim = cosine_similarity(vac_vect[0], resume_vect[0])

    print(cosin_sim)

    return cosin_sim[0][0]


def agregated_cos_sim(array_text, array_tf_idf, array_cv):
    array_text = normalize([array_text], norm="l1")[0]
    array_tf_idf = normalize([array_tf_idf], norm="l1")[0]
    array_cv = normalize([array_cv], norm="l1")[0]

    array_itog = [(i1 + i2 + i3) / 3 for i1, i2, i3 in zip(array_text, array_tf_idf, array_cv)]
    return array_itog


def algorithm(vacancy_str, resumes_str):
    array_text = []
    array_tf_idf = []
    array_cv = []

    vacancy_str = concat_vacancy(vacancy_str)
    vacancy_str_for_cv = get_vacancy_key_words(vacancy_str)

    for resume in resumes_str:

        array_text.append(encode_text_data(vacancy_str, concat_resume(resume)))
        array_tf_idf.append(tf_idf_cos_sim(vacancy_str, concat_resume(resume)))
        array_cv.append(cv_cos_sim(vacancy_str_for_cv, concat_resume(resume)))

    res = agregated_cos_sim(array_text, array_tf_idf, array_cv)
    return res


def summarize_text(text: str, model=bart_model, tokenizer=bart_tokenizer) -> str:
    input_ids = tokenizer(text, return_tensors="pt", truncation=True)
    generated_tokens = model.generate(**input_ids)

    result = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return result[0]


def encode_text_data(vac: str,
                     res: tuple,
                     model_sum=bart_model,
                     token_sum=bart_tokenizer,
                     model_emb=hrbert_model,
                     token_emb=hrbert_tokenizer
                     ) -> float:
    res_stack = res[0]
    res_exp = res[1]

    resume_exp_translated = translate_chunked_crazyML(res_exp, "en")
    resume_exp_translated = ''.join(
        filter(lambda x: x in printable, resume_exp_translated)
    )

    resume_exp_summary = summarize_text(
        resume_exp_translated,
        model_sum,
        token_sum
    )

    resume_summary_translated = translate_chunked_crazyML(res_exp, "ru")

    cos_sim = text_cos_sim(
        vac,
        resume_summary_translated + "\n" + res_stack,
        model_emb,
        token_emb
    )

    return cos_sim
