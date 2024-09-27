import io
import os
import re
import time
import shutil
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle

from konlpy.tag import Okt
from collections import Counter
from loguru import logger
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
from streamlit_date_picker import date_range_picker, PickerType
from streamlit_js_eval import streamlit_js_eval
from openai import OpenAI

API_KEY = 'api_key'

client = OpenAI(api_key=API_KEY)
# text_embedding_model = "text-embedding-ada-002"
LLM_model ="gpt-4o"

#디폴트
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="New Eojeans",
    page_icon=":unicorn_face:")

#버튼
st.markdown("""
<style>
div.stButton > button:first-child {
  font-family: "Open Sans", sans-serif;
  font-size: 16px;
  letter-spacing: 2px;
  text-decoration: none;
  text-transform: uppercase;
  color: #000;
  cursor: pointer;
  border: 3px solid;
  padding: 0.25em 0.5em;
  box-shadow: 1px 1px 0px 0px, 2px 2px 0px 0px, 3px 3px 0px 0px, 4px 4px 0px 0px, 5px 5px 0px 0px;
  position: relative;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}
.button-54:active {
  box-shadow: 0px 0px 0px 0px;
  top: 5px;
  left: 5px;
}
@media (min-width: 768px) {
  .button-54 {
    padding: 0.25em 0.75em;
  }
}
</style>""", unsafe_allow_html=True)

#글씨체_1
font_urll = "https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap"
st.markdown(f'<link href="{font_urll}" rel="stylesheet">', unsafe_allow_html=True)
custom_csss = """
    <style>
    body {
        font-family: "Do Hyeon", sans-serif;
    }
    </style>
"""
st.markdown(custom_csss, unsafe_allow_html=True)

#글씨체_2
font_url = "https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Do+Hyeon&display=swap"
st.markdown(f'<link href="{font_url}" rel="stylesheet">', unsafe_allow_html=True)
custom_css = """
    <style>
    body {
          font-family: "Do Hyeon", sans-serif;
    }
    h1 {
        font-family: "Do Hyeon", sans-serif;
    }
    h2 {
        font-family: "Do Hyeon", sans-serif;
    }
    h3 {
        font-family: "Do Hyeon", sans-serif;
    }
    p {
        font-family: "Do Hyeon", sans-serif;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

from defSET import save_uploaded_file
### 사용자가 업로드한 파일을 읽어와 지정된 디렉토리에 저장하는 함수
# def save_uploaded_file(directory, file):

#     ## 디렉토리가 없으면 생성
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     ## 파일 경로 생성
#     file_path = os.path.join(directory, file.name)

#     ## 파일을 바이너리/쓰기 모드로 열고, 업로드된 파일의 버퍼 데이터를 파일에 씀 → 바이너리란? 텍스트 파일이 아닌, 이미지 동영상 등을 다룰 때 사용
#     with open(file_path, 'wb') as f:
#         f.write(file.getbuffer()) # 저장
#     return file_path


from defSET import cleanup_directory
### 디렉토리 내에 존재하는 파일을 삭제하는 함수
# def cleanup_directory(directory):
#     if os.path.exists(directory):
#         for filename in os.listdir(directory):

#             ## 현재 파일 또는 디렉토리의 전체 경로 생성
#             file_path = os.path.join(directory, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path): # 파일 또는 링크인지 확인
#                     os.unlink(file_path) # 삭제
#                 elif os.path.isdir(file_path): # 디렉토리 인지 확인
#                     shutil.rmtree(file_path) # 삭제
#             except Exception as e:
#                 print(f'Failed to delete {file_path}. Reason: {e}')


from defSET import find_latest_file
### 가장 최근에 생성된 파일을 불러오는 함수
# def find_latest_file(folder_path):
#     txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')] # txt로 끝나는 파일을 리스트로 생성
#     if not txt_files:
#         raise FileNotFoundError(f"디렉토리에서 .txt 파일을 찾을 수 없습니다: {folder_path}")
#     txt_files = [os.path.join(folder_path, f) for f in txt_files] # 해당 파일의 경로와 파일 이름을 결합 → 각 파일의 전체 경로로 변경됨
#     latest_file = max(txt_files, key=os.path.getctime) # getctime(파일생성시간)을 확인해서 가장 최근 파일
#     return latest_file # 가장 최근 파일만 return


from defSET import extract_date_range
### 입력 데이터에서 첫 날짜와 마지막 날짜 구하는 함수
# def extract_date_range(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines() # 줄 별로 읽기

#     date_pattern = re.compile(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일')
#     dates = []

#     for line in lines:
#         date_match = date_pattern.match(line)
#         if date_match:
#             date_str = f"{date_match.group(1)}. {date_match.group(2)}. {date_match.group(3)}."
#             date_obj = datetime.strptime(date_str, '%Y. %m. %d.')
#             dates.append(date_obj)

#     if not dates:
#         raise ValueError("파일에서 유효한 날짜를 찾을 수 없습니다.")

#     # 첫 번째 날짜와 마지막 날짜를 반환
#     return min(dates), max(dates)


from defSET import convert_time_format
### 오전/오후 시간을 24시간 형식으로 변환해주는 함수
# def convert_time_format(date_str):

#     match = re.search(r'(\d{4}\. \d{1,2}\. \d{1,2}\.) (오전|오후) (\d{1,2}):(\d{2})', date_str)
#     if match:
#         year_month_day = match.group(1)
#         period = match.group(2)
#         hour = int(match.group(3))
#         minute = match.group(4)

#         if period == '오후' and hour != 12:
#             hour += 12
#         elif period == '오전' and hour == 12:
#             hour = 0

#         return f"{year_month_day} {hour:02}:{minute}"
#     return date_str


from defSET import extract_all_chat_logs
### 대화 내용 추출하는 함수
# def extract_all_chat_logs(file_path, start_date, end_date):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()

#     chat_logs = []
#     current_date = None

#     ## 다양한 패턴의 문자열을 하나의 형태로 일치시켜주기 위한 re.complile
#     date_pattern = re.compile(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일')
#     message_pattern1 = re.compile(r'(\d{4}\. \d{1,2}\. \d{1,2}\. .+?), (.+?) : (.+)') # 아이폰
#     message_pattern2 = re.compile(r'(\d{4}년 \d{1,2}월 \d{1,2}일 .+?), (.+?) : (.+)') # 갤럭시
#     ## '?' : 앞의 표현식이 0번 또는 1번 // '+' : 1번 이상 // '*' : 0번 이상
#     ## '+?' : 앞의 표현식이 1번 이상 나타나되, 가능한 적게 매칭 // '*?' : 0번 이상 // '??' : 0번 또는 1번 이상
#     #    \d{4}    : 정확히 4자리 숫자 (연도)
#     #    \.       : 마침표 문자 (날짜 구분자)
#     #    \d{1,2}  : 1자리 또는 2자리 숫자 (월 또는 일)
#     #    .+?      : 비탐욕적 매칭으로, 가능한 적게 매칭 (여기서는 메시지 내용 중 콤마와 일치하는 부분까지 최소한의 텍스트를 매칭)
#     #    (.+?)    : 비탐욕적 매칭으로, 콤마 앞의 발신자 이름
#     #    (.+)     : 탐욕적 매칭으로, 콜론 뒤의 메시지 내용



#     # 문자열 상태의 날짜를 datetime 형태로 변환
#     start_date = datetime.strptime(start_date, '%Y. %m. %d.')
#     end_date = datetime.strptime(end_date, '%Y. %m. %d.')


#     for line in lines:
#         date_match = date_pattern.match(line)
#         if date_match:
#             ## 매칭된 그룹을 사용하여 날짜를 datetime 객체로 변환
#             year, month, day = map(int, date_match.groups())
#             current_date = datetime(year, month, day)

#         ## 현재 날짜가 설정되어 있고, 시작일과 종료일 사이에 있는 경우
#         if current_date and start_date <= current_date <= end_date:
#             message_match1 = message_pattern1.match(line)
#             message_match2 = message_pattern2.match(line)
#             if message_match1:
#                 date_time = convert_time_format(message_match1.group(1))
#                 name = message_match1.group(2)
#                 message = message_match1.group(3)
#             elif message_match2:
#                 date_time = convert_time_format(message_match2.group(1))
#                 name = message_match2.group(2)
#                 message = message_match2.group(3)
#             else:
#                 continue

#             chat_logs.append({'date': date_time, 'name': name, 'text': message, 'emotion': ""})

#     df = pd.DataFrame(chat_logs)

#     ## text 컬럼에서 불용어 처리
#     if not df.empty:
#         unwanted_phrases = ['http', '이모티콘', '사진', '동영상']
#         df = df[~df['text'].str.contains('|'.join(unwanted_phrases))]

#     return df


from defSET import calculate_daily_message_count
### 날짜별 주고 받은 마디 수 계산하는 함수
# def calculate_daily_message_count(df):
#     df['date_only'] = pd.to_datetime(df['date']).dt.date  # 날짜만 추출
#     daily_message_count = df.groupby('date_only').size().reset_index(name='message_count')
#     return daily_message_count


from defSET import calculate_response_times
### 사용자 모두 답장 평균 속도 구하는 함수
# def calculate_response_times(df, user_name):
#     daily_response_times = []
#     grouped = df.groupby('date_only')

#     for date, group in grouped:
#         user_msgs = group[group['name'] == user_name].sort_values(by='date_time')
#         other_msgs = group[group['name'] != user_name].sort_values(by='date_time')

#         response_times = []

#         for i, other_msg in other_msgs.iterrows():
#             previous_user_msg = user_msgs[user_msgs['date_time'] < other_msg['date_time']]
#             if not previous_user_msg.empty:
#                 last_user_msg_time = previous_user_msg.iloc[-1]['date_time']
#                 response_time = (other_msg['date_time'] - last_user_msg_time).total_seconds() / 60.0
#                 response_times.append(response_time)

#         if response_times:
#             avg_response_time = sum(response_times) / len(response_times)
#         else:
#             avg_response_time = None

#         daily_response_times.append(avg_response_time)

#     # None 값을 제외하고 평균을 계산
#     valid_response_times = [time for time in daily_response_times if time is not None]
#     return valid_response_times


from mmod import mod




# 각 사람의 평균 답장 속도를 계산
response_times_dict = {}

### 메인 함수
def main():
    #배경화면
    page_bg_img = '''
    <style>
    [data-testid = 'stAppViewContainer'] > .main{
    background-image: url("https://i.imgur.com/LpBIJyr.jpg");
    background-attachment: fixed;
    background-size: cover
    }

    [data-testid = 'stHeader'] {
    background-color: rgba(0, 0, 0, 0);
    }

    [class = "st-emotion-cache-uhkwx6 ea3mdgi6"] {
    background-color: rgba(0, 0, 0, 0);
    }

    [data-testid = 'stToolbar'] {
    right : 2rem;
    }

    [data-testid='stSidebar'] > div:first-child {
    background-color :#ffffff
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)




    st.sidebar.image("https://i.imgur.com/BFWaHsz.png", use_column_width=True)
    with st.sidebar:
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []

        uploaded_files = st.file_uploader("Upload your file", type=['txt'])
        if uploaded_files:
            txt_file = io.StringIO(uploaded_files.getvalue().decode('utf-8')).read()
            st.code(txt_file)

            file_path = save_uploaded_file('tmp', uploaded_files)
            st.session_state.uploaded_files.append(file_path)
            st.success(f'파일이 임시 위치에 저장되었습니다: {file_path}')

    if uploaded_files:
        if st.button("HOME"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        st.subheader(':black_small_square:추출하고 싶은 기간을 선택하세요')

        #배경화면
        page_bg_img = '''

            <style>
            [data-testid = 'stAppViewContainer'] > .main{
            background-image: url("https://i.imgur.com/4Mj1yR2.jpg");
            background-attachment: fixed;
            background-size: cover
            }

            [data-testid = 'stHeader'] {
            background-color: rgba(0, 0, 0, 0);
            }

            [class = "st-emotion-cache-uhkwx6 ea3mdgi6"] {
            background-color: rgba(0, 0, 0, 0);
            }

            [data-testid = 'stToolbar'] {
            right : 2rem;
            }

            [data-testid='stSidebar'] > div:first-child {
            background-color :#ffffff
            }

            </style>
            '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

        try:
            default_start, default_end = extract_date_range(file_path)
        except ValueError as e:
            st.error(f"날짜를 추출하는 동안 오류가 발생했습니다: {e}")
            return

        date_range_string = date_range_picker(picker_type=PickerType.date,
                                              start=default_start, end=default_end,
                                              key='date_range_picker')

        if date_range_string:
            start, end = date_range_string

            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')

            start_formatted = f"{start_date.year}. {start_date.month}. {start_date.day}."
            end_formatted = f"{end_date.year}. {end_date.month}. {end_date.day}."
            st.subheader(f"[{start_formatted}] 부터 [{end_formatted}] 까지 선택되었습니다.")

        st.text("")
        st.text("")

        if st.button("분석하기"):
            try:
                latest_file_path = find_latest_file('tmp')
                df = extract_all_chat_logs(latest_file_path, start_formatted, end_formatted)

                if not df.empty:
                    daily_message_count = calculate_daily_message_count(df)
#####################################################################################
                    forllm = df.tail(10)
                    forllm['name'] = forllm['name'] + ':'
                    forllm = forllm.iloc[:10, 1:3]
                    forllm_string = forllm.to_string()
#####################################################################################  답장 속도

                    # 날짜와 시간을 datetime 형식으로 변환
                    df['date_time'] = pd.to_datetime(df['date'])
                    df['date_only'] = df['date_time'].dt.date
#####################################################################################  답장 속도

                    # 두 사람의 이름을 추출
                    names = df['name'].unique()
                    for name in names:
                        response_times = calculate_response_times(df, name)
                        if response_times:  # 빈 리스트가 아닌 경우에만 계산
                            avg_response_time = round(sum(response_times) / len(response_times), 1)
                        else:
                            avg_response_time = None
                        response_times_dict[name] = avg_response_time

                    st.text("")
                    st.text("")
                    st.text("")


                    st.title(':black_small_square: 평균 답장 시간')
                    # Create list of GTR sentences
                    results = []
                    for name, time_value in response_times_dict.items():
                        GTR = '"{0}" 의 평균 답장 속도는 "{1}분"'.format(name, time_value)
                        results.append(GTR)

                    # Join all GTR sentences with <br> for HTML line breaks
                    combined_text = '<br>'.join(results)

                    def stream_data(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.05)  # Adjust the speed of text display

                    font_size = 40

                    placeholder = st.empty()
                    accumulated_text = ""
                    for updated_text in stream_data(combined_text):
                        accumulated_text += updated_text
                        placeholder.markdown(f'<p style="font-size: {font_size}px;">{accumulated_text}</p>', unsafe_allow_html=True)

                    st.text("")
                    st.text("")

#####################################################################################  채팅 시간대
                    st.title(':black_small_square:시간대별 대화수')
                    ## date 컬럼에서 시간대 뽑기
                    df['hour'] = pd.to_datetime(df['date'], format='%Y. %m. %d. %H:%M').dt.hour

                    result = {}

                    # 시간대 별로 key 값 나누고 각각 해당하는 밸류값에
                    for hour in range(24):
                        time_range = f"{hour}"####
                        count = df[df['hour'] == hour].shape[0]
                        result[time_range] = count

                    # Print the result
                    # st.dataframe(result)
                    # st.text(result)

                    hours = []
                    amounts = []

                    for hour, amount in result.items():
                        hours.append(hour)
                        amounts.append(amount)

                    hours = list(map(int, hours))

                    option = {
                        "xAxis": {
                            "type": "category",
                            "data": hours,
                            "axisLine": {
                                "lineStyle": {
                                    "color": "#ffffff"  # Color of the x-axis line
                                }
                            },
                            "axisLabel": {
                                "color": "#333333"  # Color of the x-axis labels
                            }
                        },
                        "yAxis": {
                            "type": "value",
                            "axisLine": {
                                "lineStyle": {
                                    "color": "#ffffff"  # Color of the y-axis line
                                }
                            },
                            "axisLabel": {
                                "color": "#333333"  # Color of the y-axis labels
                            }
                        },
                        "series": [
                            {
                                "data": amounts,
                                "type": "line",
                                "itemStyle": {
                                    "color": "#FF3CBB",  # Tomato red color for the line
                                    "borderWidth": 2,
                                    "borderColor": "#FF3CBB"  # Same color for border
                                },
                                "lineStyle": {
                                    "width": 4,  # Thickness of the line
                                    "type": "solid"  # Dashed line style?
                                },
                                "symbol": "none",  # Shape of the markers?
                                "symbolSize": 8,  # Size of the markers
                                "markPoint": {
                                    "data": [
                                        {"type": "max", "name": "Max Value", "symbol": "pin", "symbolSize": 50, "itemStyle": {"color": "#FF3CBB"}},
                                        {"type": "min", "name": "Min Value", "symbol": "pin", "symbolSize": 50, "itemStyle": {"color": "#FF3CBB"}}
                                    ]
                                },
                                "markLine": {
                                    "data": [
                                        {"type": "average", "name": "Average Line", "lineStyle": {"color": "#FF3CBB"}}
                                    ]
                                },
                                "areaStyle": {
                                    "color": {
                                        "type": "linear",
                                        "x": 0,
                                        "y": 0,
                                        "x2": 1,
                                        "y2": 1,
                                        "global": False,
                                        "globalAlpha": 0.2,
                                        "colorStops": [
                                            {"offset": 0, "color": "rgba(245, 40, 145, 0.8)"},
                                            {"offset": 1, "color": "rgba(255, 205, 0, 0.8)"}
                                        ]
                                    }
                                }
                            }
                        ],
                        "tooltip": {
                            "trigger": "axis",
                            "axisPointer": {
                                "type": "cross",
                                "label": {
                                    "backgroundColor": "#6a7985"
                                }
                            },
                            "backgroundColor": "#ffffff",  # Background color of the tooltip
                            "borderColor": "#cccccc",  # Border color of the tooltip
                            "borderWidth": 1,
                            "textStyle": {
                                "color": "#333333"  # Text color inside the tooltip
                            }
                        }
                    }
                    st_echarts(options=option, height="400px")
#####################################################################################  많이 사용된 키워드
                    st.title(':black_small_square: 가장 많이 나온 단어')
                    # Okt 형태소 분석기 초기화
                    okt = Okt()

                    ## 불용어
                    stopword_file_path = "C:/Users/USER/Desktop/vscode/RAG_r/stopwords-ko.csv"
                    stopwords_df = pd.read_csv(stopword_file_path, encoding='cp949')
                    stopwords = set(stopwords_df['stopwords'])

                    # 텍스트 컬럼 추출, 결측값 제거
                    texts = df['text'].dropna()

                    # 텍스트를 토큰화하고 명사만 추출
                    nouns = []
                    for text in texts:
                        # 각 텍스트에서 명사만 추출하고 불용어 제거
                        nouns.extend([noun for noun in okt.nouns(text) if noun not in stopwords])

                    # 명사의 빈도수 계산
                    count = Counter(nouns)

                    # 빈도수를 데이터프레임으로 변환
                    keywords_df = pd.DataFrame(count.items(), columns=['Keyword', 'Frequency'])
                    # 빈도수를 기준으로 내림차순 정렬
                    keywords_df = keywords_df.sort_values(by='Frequency', ascending=False)
                    nested_list = [tuple(x) for x in keywords_df.to_records(index=False)]
                    top10 = keywords_df.head(20)

                    # 상위 20개의 키워드 출력
                    # st.dataframe(keywords_df.head(20))
                    # st.text(keywords_df.head(20))

                    nested_list = [tuple(x) for x in top10.to_records(index=False)]
                    converted_list = [(item[0], str(item[1])) for item in nested_list]
                    data = [
                        {"name": name, "value": value}
                    for name, value in converted_list
                    ]
                    wordcloud_option = {
                        "series": [{
                            "type": "wordCloud",
                            "data": data,
                            "textStyle": {
                                "fontFamily": "Arial",  # Optional: Specify font family
                                "fontWeight": "bold",   # Optional: Font weight (normal, bold, etc.)
                                "color": "#000000",        # Optional: Color of the text (can use any color format)
                                "shadowBlur": 1,        # Optional: Shadow blur (optional for adding shadow)
                                "shadowColor": "#333",  # Optional: Shadow color (if shadowBlur is set)
                            },
                            "sizeRange": [30, 60]    # Adjust range of font sizes here (min, max)
                        }]
                    }
                    st_echarts(wordcloud_option)
#####################################################################################





                    st.title(':black_small_square: 대화의 온도')

                    with open(latest_file_path, 'r', encoding='utf-8') as file:
                        text_data = file.readlines()




                    # # 데이터를 데이터프레임으로 변환
                    df3 = pd.DataFrame({'text': text_data})



                    # 결과 계산
                    result22 = mod(df3)
#####################################################################################
                    # df2 = df[["text", "emotion"]]

                    # texts_df = df2[["text"]]

                    # client = OpenAI(api_key = API_KEY)

                    # text_embedding_dic = {}

                    # for idx, row in texts_df.iterrows():

                    #     input_text = row["text"]
                    #     response = client.embeddings.create(
                    #         input = input_text,
                    #         model = "text-embedding-3-small"
                    #     )
                    #     result = response.data[0].embedding
                    #     text_embedding_dic[input_text] = result


                    # embedded_df = pd.DataFrame({
                    #     'text': text_embedding_dic.keys(),          # 원본 메시지
                    #     'embedding': text_embedding_dic.values()    # embedded 메시지
                    # })

                    # X = embedded_df["embedding"].tolist()



                    # with open('C:/Users/USER/Desktop/vscode/momodel.pickle', 'rb') as f:
                    #     lgbm = pickle.load(f)

                    # # 예측 확률 계산
                    # probabilities = lgbm.predict_proba(X)

                    # # ['부정'클래스 확률, '긍정'클래스 확률]
                    # # "긍정"클래스 확률만 추출
                    # positive_probs = probabilities[:, 1]

                    # result_proba = positive_probs.mean()


###############################################################################
                    liquidfill_option = {

                            "title": {
                            "text": "",
                            "textStyle": {
                                "color": "#000000",  # Change the title text color
                                "fontSize": 24  # Change the title font size
                            },
                            "left": "center",  # Position the title in the center
                            "top": "top"  # Position the title at the top
                        },

                        "series": [
                            {
                                'shape': 'path://M140 20C73 20 20 74 20 140c0 135 136 170 228 303 88-132 229-173 229-303 0-66-54-120-120-120-48 0-90 28-109 69-19-41-60-69-108-69z',
                                "type": "liquidFill",
                                "data": [round(result22,2)],
                                'radius' : '80%',
                                "itemStyle": {
                                    "color": "#ff00ea",  # Change the color of the liquid
                                    "borderColor": "#ff00ea",  # Change the border color
                                    "borderWidth": 10  # Change the border width
                                },
                                "outline": {
                                    "show": True,
                                    "borderDistance": 0,  # Distance between the border and the liquid
                                    "itemStyle": {
                                        "borderColor": "#ff00ea",  # Border color
                                        "borderWidth": 10  # Border width
                                    }
                                }
                            }
                        ]
                    }
                    st_echarts(liquidfill_option)
###############################################################################################
                    from defSET import system_template, ChatGPT_conversation

                    st.title(':black_small_square: 분석?')
                    conversation_example = forllm_string
                    system_message = system_template(conversation_example)
                    conversation = [
                                {"role": "system", "content": system_message},
                                ]
                    # 대화 생성
                    conversation = ChatGPT_conversation(conversation)
                    streaming = conversation[-1]["content"].strip()

                    def stream_data():
                        text = ''
                        for word in streaming.split(" "):
                            text += word + " "
                            yield text
                            time.sleep(0.05)

                    font_size = 30


                    placeholder = st.empty()
                    for updated_text in stream_data():
                        placeholder.markdown(f'<p style="font-size: {font_size}px;">{updated_text}</p>', unsafe_allow_html=True)


                else:
                    st.write("해당 기간과 이름에 맞는 대화 내용이 없습니다.")
            except FileNotFoundError as e:
                st.error(f"Error: {e}")


if __name__ == '__main__':
    cleanup_directory('tmp')
    main()


