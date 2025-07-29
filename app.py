import streamlit as st
import streamlit.components.v1 as htmlviewer
import base64

# ✅ Streamlit 설정
st.set_page_config(layout='wide', page_title='5계 분류하기 모델')
st.title('This is yang Webapp!!')


# ✅ 이미지 파일을 base64 문자열로 변환하는 함수
def img_to_base64(img_path):
    with open(img_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f'data:image/png;base64,{encoded}'


# ✅ 이미지 base64 인코딩
flowchart_base64 = img_to_base64('flowchart.png')   # indexmy1.html 에서 사용
biocard_base64 = img_to_base64('biocard.png')       # indexmy2.html 에서 사용
chart_base64 = img_to_base64('chart.png')       # indexmy3.html 에서 사용



# ✅ HTML 파일 읽기
with open('indexmy1.html', 'r', encoding='utf-8') as f:
    html1 = f.read()

with open('indexmy2.html', 'r', encoding='utf-8') as f:
    html2 = f.read()

with open('indexmy3.html', 'r', encoding='utf-8') as f:
    html3 = f.read()




# ✅ 이미지 경로를 base64로 치환
html1 = html1.replace('src="flowchart.png"', f'src="{flowchart_base64}"')
html2 = html2.replace('src="biocard.png"', f'src="{biocard_base64}"')
html3 = html3.replace('src="chart.png"', f'src="{chart_base64}"')


# ✅ Streamlit 레이아웃 구성
col1, col2 = st.columns((4, 1))

with col1:
    with st.expander('Content #1...'):
        url = 'https://www.youtube.com/watch?v=XyEOEBsa8I4'
        st.info('Content..')
        st.video(url)

    with st.expander('Content #2...'):
        htmlviewer.html(html1, height=3000)

    with st.expander('Content #3...'):
        htmlviewer.html(html2, height=3000)

    with st.expander('Content #4...'):
        htmlviewer.html(html3, height=2000)

with col2:
    with st.expander('Tips..'):
        st.info('Tips..')

# ✅ 하단 저작권 표시
st.markdown('<hr>', unsafe_allow_html=True)
st.write(
    '<font color="BLUE">(c)copyright. all rights reserved by hcyang</font>',
    unsafe_allow_html=True
)
