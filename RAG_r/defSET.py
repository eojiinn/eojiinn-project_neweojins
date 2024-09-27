import shutil
import os
import re
from datetime import datetime
from openai import OpenAI
import pandas as pd
import numpy as np

### 사용자가 업로드한 파일을 읽어와 지정된 디렉토리에 저장하는 함수
def save_uploaded_file(directory, file):

    ## 디렉토리가 없으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## 파일 경로 생성
    file_path = os.path.join(directory, file.name)

    ## 파일을 바이너리/쓰기 모드로 열고, 업로드된 파일의 버퍼 데이터를 파일에 씀 → 바이너리란? 텍스트 파일이 아닌, 이미지 동영상 등을 다룰 때 사용
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer()) # 저장
    return file_path




def cleanup_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):

            ## 현재 파일 또는 디렉토리의 전체 경로 생성
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): # 파일 또는 링크인지 확인
                    os.unlink(file_path) # 삭제
                elif os.path.isdir(file_path): # 디렉토리 인지 확인
                    shutil.rmtree(file_path) # 삭제
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')





def find_latest_file(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')] # txt로 끝나는 파일을 리스트로 생성
    if not txt_files:
        raise FileNotFoundError(f"디렉토리에서 .txt 파일을 찾을 수 없습니다: {folder_path}")
    txt_files = [os.path.join(folder_path, f) for f in txt_files] # 해당 파일의 경로와 파일 이름을 결합 → 각 파일의 전체 경로로 변경됨
    latest_file = max(txt_files, key=os.path.getctime) # getctime(파일생성시간)을 확인해서 가장 최근 파일
    return latest_file # 가장 최근 파일만 return





def extract_date_range(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines() # 줄 별로 읽기

    date_pattern = re.compile(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일')
    dates = []

    for line in lines:
        date_match = date_pattern.match(line)
        if date_match:
            date_str = f"{date_match.group(1)}. {date_match.group(2)}. {date_match.group(3)}."
            date_obj = datetime.strptime(date_str, '%Y. %m. %d.')
            dates.append(date_obj)

    if not dates:
        raise ValueError("파일에서 유효한 날짜를 찾을 수 없습니다.")

    # 첫 번째 날짜와 마지막 날짜를 반환
    return min(dates), max(dates)

import re

def convert_time_format(date_str):

    match = re.search(r'(\d{4}\. \d{1,2}\. \d{1,2}\.) (오전|오후) (\d{1,2}):(\d{2})', date_str)
    if match:
        year_month_day = match.group(1)
        period = match.group(2)
        hour = int(match.group(3))
        minute = match.group(4)

        if period == '오후' and hour != 12:
            hour += 12
        elif period == '오전' and hour == 12:
            hour = 0

        return f"{year_month_day} {hour:02}:{minute}"
    return date_str





### 대화 내용 추출하는 함수
def extract_all_chat_logs(file_path, start_date, end_date):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chat_logs = []
    current_date = None

    ## 다양한 패턴의 문자열을 하나의 형태로 일치시켜주기 위한 re.complile
    date_pattern = re.compile(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일')
    message_pattern1 = re.compile(r'(\d{4}\. \d{1,2}\. \d{1,2}\. .+?), (.+?) : (.+)') # 아이폰
    message_pattern2 = re.compile(r'(\d{4}년 \d{1,2}월 \d{1,2}일 .+?), (.+?) : (.+)') # 갤럭시
    ## '?' : 앞의 표현식이 0번 또는 1번 // '+' : 1번 이상 // '*' : 0번 이상
    ## '+?' : 앞의 표현식이 1번 이상 나타나되, 가능한 적게 매칭 // '*?' : 0번 이상 // '??' : 0번 또는 1번 이상
    #    \d{4}    : 정확히 4자리 숫자 (연도)
    #    \.       : 마침표 문자 (날짜 구분자)
    #    \d{1,2}  : 1자리 또는 2자리 숫자 (월 또는 일)
    #    .+?      : 비탐욕적 매칭으로, 가능한 적게 매칭 (여기서는 메시지 내용 중 콤마와 일치하는 부분까지 최소한의 텍스트를 매칭)
    #    (.+?)    : 비탐욕적 매칭으로, 콤마 앞의 발신자 이름
    #    (.+)     : 탐욕적 매칭으로, 콜론 뒤의 메시지 내용



    # 문자열 상태의 날짜를 datetime 형태로 변환
    start_date = datetime.strptime(start_date, '%Y. %m. %d.')
    end_date = datetime.strptime(end_date, '%Y. %m. %d.')


    for line in lines:
        date_match = date_pattern.match(line)
        if date_match:
            ## 매칭된 그룹을 사용하여 날짜를 datetime 객체로 변환
            year, month, day = map(int, date_match.groups())
            current_date = datetime(year, month, day)

        ## 현재 날짜가 설정되어 있고, 시작일과 종료일 사이에 있는 경우
        if current_date and start_date <= current_date <= end_date:
            message_match1 = message_pattern1.match(line)
            message_match2 = message_pattern2.match(line)
            if message_match1:
                date_time = convert_time_format(message_match1.group(1))
                name = message_match1.group(2)
                message = message_match1.group(3)
            elif message_match2:
                date_time = convert_time_format(message_match2.group(1))
                name = message_match2.group(2)
                message = message_match2.group(3)
            else:
                continue

            chat_logs.append({'date': date_time, 'name': name, 'text': message, 'emotion': ""})

    df = pd.DataFrame(chat_logs)

    ## text 컬럼에서 불용어 처리
    if not df.empty:
        unwanted_phrases = ['http', '이모티콘', '사진', '동영상']
        df = df[~df['text'].str.contains('|'.join(unwanted_phrases))]

    return df




def calculate_daily_message_count(df):
    df['date_only'] = pd.to_datetime(df['date']).dt.date  # 날짜만 추출
    daily_message_count = df.groupby('date_only').size().reset_index(name='message_count')
    return daily_message_count


def calculate_response_times(df, user_name):
    daily_response_times = []
    grouped = df.groupby('date_only')

    for date, group in grouped:
        user_msgs = group[group['name'] == user_name].sort_values(by='date_time')
        other_msgs = group[group['name'] != user_name].sort_values(by='date_time')

        response_times = []

        for i, other_msg in other_msgs.iterrows():
            previous_user_msg = user_msgs[user_msgs['date_time'] < other_msg['date_time']]
            if not previous_user_msg.empty:
                last_user_msg_time = previous_user_msg.iloc[-1]['date_time']
                response_time = (other_msg['date_time'] - last_user_msg_time).total_seconds() / 60.0
                response_times.append(response_time)

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
        else:
            avg_response_time = None

        daily_response_times.append(avg_response_time)

    # None 값을 제외하고 평균을 계산
    valid_response_times = [time for time in daily_response_times if time is not None]
    return valid_response_times




def system_template(examples):

    # 여기 template에 GPT에게 추천할 말을 적으면 됨. 두 가지 다 뽑아봐야 함.

    # 1. 토픽 이런거 없이 그냥 조언해달라고 할 때
    #   : 사용자 대화를 보고 자연스럽게 대화를 이어나갈 수 있는 주제를 골라서, 자연스럽게 넘어가게 조언해줘.

    # 2. 토픽 주면서 조언해달라고 할 때
    #   : 이건 토픽이야. 자연스럽게 대화를 이어나갈 수 있는 주제를 골라서, 자연스럽게 넘어가게 조언해줘.

    # 이건 멘토님이 주셨던
    template= \
f"""
    당신은 대화형 AI 모델입니다. 사용자와의 대화에서 호감도를 높이기 위해 다음 사항을 고려해 주세요:

    1. 사용자가 현재 이야기하고 있는 주제를 잘 이해하고 자연스럽게 이어나가세요.
    2. 사용자가 긍정적인 반응을 보인 주제는 계속 이어가며, 부정적인 반응을 보인 주제는 긍정적인 주제로 전환하세요.
    3. 주제를 변경할 때는 관련된 주제나 자연스럽게 연결될 수 있는 주제로 전환하세요.

    다음은 사용자가 입력한 대화 내용입니다:

    {examples}

    아래와 같은 방식으로 대화를 이어가세요:

    - Assistant: 강아지를 키우는 것은 정말 즐거운 일인 것 같아요. 어떤 종의 강아지를 키우고 계신가요? 특별한 추억이 있다면 나눠주세요.
    - Assistant: 반려동물과 함께하는 시간은 정말 소중하죠. 혹시 강아지와 함께하는 활동 중 가장 좋아하는 것은 무엇인가요?
    - Assistant: 네, 정말 걱정이 많으실 것 같아요. 이런 상황일수록 잠시 다른 주제로 기분을 환기시키는 것도 좋을 것 같아요. 최근에 즐겁게 본 영화나 흥미로운 취미 활동이 있으신가요?
    - Assistant: 그렇죠, 정치 상황이 많이 힘들죠. 그런데 요즘 즐겨하시는 취미나 기분을 좋게 만드는 활동이 있으신가요? 이런 주제로 얘기하면 조금 기분이 나아질 수도 있을 것 같아요.

    대화 내용은 보여주지 말고, Assistant 를 참고해서 "하면 어떨까요?" 형식으로 추천만 해줘.

"""
    return template



API_KEY = 'api_key'

client = OpenAI(api_key=API_KEY)
text_embedding_model = "text-embedding-ada-002"
LLM_model ="gpt-4o"

def ChatGPT_conversation(conversation=[]):
    response = client.chat.completions.create(
        model=LLM_model,
        messages=conversation
    )

    total_tokens = response.usage.total_tokens
    #logger.info(f"Total tokens used: {total_tokens}")

    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation