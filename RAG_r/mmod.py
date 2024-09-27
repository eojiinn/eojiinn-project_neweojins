# import pandas as pd
# from openai import OpenAI
# import pickle
# import numpy as np
# import random

# def mod(df):


#     # 랜덤 시드 고정
#     def set_seed(seed):
#         np.random.seed(seed)
#         random.seed(seed)

#     seed = 42
#     set_seed(seed)

#     # text, emotion만 뽑은 dataframe
#     df2 = df[["text", "emotion"]]

#     texts_df = df2[["text"]]


#     text_embedding_dic = {}

#     for idx, row in texts_df.iterrows():

#         input_text = row["text"]
#         response = client.embeddings.create(
#             input = input_text,
#             model = "text-embedding-3-small"
#         )
#         result = response.data[0].embedding
#         text_embedding_dic[input_text] = result


#     embedded_df = pd.DataFrame({
#         'text': text_embedding_dic.keys(),          # 원본 메시지
#         'embedding': text_embedding_dic.values()    # embedded 메시지
#     })

#     X = embedded_df["embedding"].tolist()


#     with open('C:/Users/USER/Desktop/vscode/momodel.pickle', 'rb') as f:
#         lgbm = pickle.load(df2)

#     # 예측 확률 계산
#     probabilities = lgbm.predict_proba(X)

#     # ['부정'클래스 확률, '긍정'클래스 확률]
#     # "긍정"클래스 확률만 추출
#     positive_probs = probabilities[:, 1]

#     result_proba = positive_probs.mean()

#     return result_proba


import pandas as pd
import numpy as np
import random
import pickle
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoTokenizer, AutoModel

# 랜덤 시드 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# OpenAI API 설정
# client = OpenAI(api_key=API_KEY)

# 텍스트 임베딩 캐싱 딕셔너리
text_embedding_cache = {}

# # 텍스트 임베딩 계산 함수
# def get_text_embedding(input_text):
#     if input_text in text_embedding_cache:
#         return text_embedding_cache[input_text]

#     response = client.embeddings.create(
#         input=input_text,
#         model="text-embedding-ada-002"
#     )
#     result = response.data[0].embedding
#     text_embedding_cache[input_text] = result
#     return result


# 모델 및 토크나이저 로드
model_name = 'jhgan/ko-sroberta-nli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to('cpu')

# 텍스트 임베딩 계산 함수
def get_text_embedding(input_text):
    if input_text in text_embedding_cache:
        return text_embedding_cache[input_text]

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to('cpu')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    text_embedding_cache[input_text] = embeddings
    return embeddings

# 데이터프레임 처리 함수
def mod(df):
    texts_df = df['text'].tolist()

    # # 병렬 처리를 통한 임베딩 계산
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     embeddings = list(executor.map(get_text_embedding, texts_df["text"]))

    # embedded_df = pd.DataFrame({
    #     'text': texts_df["text"],
    #     'embedding': embeddings
    # })

    # X = np.array(embedded_df["embedding"].tolist())

    # 텍스트 임베딩 생성
    embeddings = [get_text_embedding(text) for text in texts_df]

    # 임베딩을 데이터프레임에 추가
    df['embedding'] = embeddings

    X = df["embedding"].tolist()

    # 모델 로딩
    with open('./cuda_model_69252_acc8561.pickle', 'rb') as f:
    # with open('C:/Users/USER/Desktop/vscode/RAG_r/cuda_model_69252_acc8561.pickle', 'rb') as f:
        lgbm = pickle.load(f)

    # 예측 확률 계산
    probabilities = lgbm.predict_proba(X)

    # "긍정" 클래스 확률만 추출
    positive_probs = probabilities[:, 1]

    result_proba = positive_probs.mean()
    return result_proba