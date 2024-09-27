import time
import numpy as np
import pandas as pd
import streamlit as st



#평균 답장 시간
st.title(':black_small_square: 평균 답장 시간')
result_1 = '"당신"의 평균 답장 속도는 "30분" 그리고 "상대방"의 평균 답장 속도는 "25분" 입니다'

def stream_data():
    text = ''
    for word in result_1.split(" "):
        text += word + " "
        yield text
        time.sleep(0.1)

font_size = 40

if st.button("분석 시작"):
    placeholder = st.empty()
    for updated_text in stream_data():
        placeholder.markdown(f'<p style="font-size: {font_size}px;">{updated_text}</p>', unsafe_allow_html=True)

