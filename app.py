
# 라이브러리 불러오기 

import pandas as pd
import numpy as np
import datetime
import joblib
from keras.models import load_model
from haversine import haversine
from urllib.parse import quote
import streamlit as st
from streamlit_folium import st_folium
import folium
import branca
from geopy.geocoders import Nominatim
import ssl
from urllib.request import urlopen
import plotly.express as px



# -------------------- ▼ 필요 함수 생성 코딩 Start ▼ --------------------


# geocoding : 거리주소 -> 위도/경도 변환 함수
# Nominatim 파라미터 : user_agent = 'South Korea', timeout=None
# 리턴 변수(위도,경도) : lati, long
# 참고: https://m.blog.naver.com/rackhunson/222403071709
def geocoding(address):
    geolocoder = Nominatim(user_agent = 'South Korea', timeout=None)
    location = geolocoder.geocode(address)
    
    lati = location.latitude
    long = location.longitude
    return lati, long


# preprocessing : '발열', '고혈압', '저혈압' 조건에 따른 질병 전처리 함수(미션3 참고)
# 리턴 변수(중증질환,증상) : X, Y
def preprocessing(desease):

    desease['발열'] = [ 1 if x >=37 else 0 for x in desease['체온']]
    desease['고혈압'] = [1 if x >= 140 else 0 for x in desease['수축기 혈압']]
    desease['저혈압'] = [1 if x <= 90 else 0 for x in desease['수축기 혈압']]
    
    Y = desease['중증질환']
    X = desease[['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란','간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다', 
           '무감각', '마비', '현기증', '졸도', '말이 어눌해졌다', '시력이 흐려짐', '발열', '고혈압', '저혈압']]
    
    return X, Y


# predict_disease : AI 모델 중증질환 예측 함수 (미션1 참고)
# 사전 저장된 모델 파일 필요(119_model_XGC.pkl)
# preprocessing 함수 호출 필요 
# 리턴 변수(4대 중증 예측) : sym_list[pred_y_XGC[0]]
def predict_disease(patient_data):
    
    sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']
    
    new_data = pd.DataFrame(patient_data)
    new_x, new_y = preprocessing(new_data)
    
    model_m = joblib.load('./119_model_XGC.pkl')
    
    pred_new_m = model_m.predict(new_x)
    
    return sym_list[pred_new_m[0]]


# find_hospital : 실시간 병원 정보 API 데이터 가져오기 (미션1 참고)
# 리턴 변수(거리, 거리구분) : distance_df
def find_hospital(special_m, lati, long):

    context=ssl.create_default_context()
    context.set_ciphers("DEFAULT")
      
    #  [국립중앙의료원 - 전국응급의료기관 조회 서비스] 활용을 위한 개인 일반 인증키(Encoding) 저장
    key = 'J5wwF%2BxjaafMT2LMp7dQNu%2FVRpSHxBp9myHZD46e8u%2Bwc2kLhluM%2FjbJg2HO%2BW9S56ylujIaqSizb3V7VAHamg%3D%3D'

    # city = 대구광역시, 인코딩 필요
    city = quote("대구광역시")
    
    # 미션1에서 저장한 병원정보 파일 불러오기 
    solution_df = pd.read_csv('./daegu_hospital_list.csv')

    # 응급실 실시간 가용병상 조회
    url_realtime = 'https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire' + '?serviceKey=' + key + '&STAGE1=' + city + '&pageNo=1&numOfRows=100'
    result = urlopen(url_realtime, context=context)
    emrRealtime = pd.read_xml(result, xpath='.//item')
    solution_df = pd.merge(solution_df, emrRealtime[['hpid', 'hvec', 'hvoc']], on = 'hpid', how='left')

    # 응급실 실시간 중증질환 수용 가능 여부
    url_acpt = 'https://apis.data.go.kr/B552657/ErmctInfoInqireService/getSrsillDissAceptncPosblInfoInqire' + '?serviceKey=' + key + '&STAGE1=' + city + '&pageNo=1&numOfRows=100'
    result = urlopen(url_acpt, context=context)
    emrAcpt = pd.read_xml(result, xpath='.//item')
    emrAcpt = emrAcpt.rename(columns={'dutyName':'hpid'})
    solution_df = pd.merge(solution_df,
                           emrAcpt[['hpid', 'MKioskTy1', 'MKioskTy2', 'MKioskTy3', 'MKioskTy4', 'MKioskTy5', 'MKioskTy7',
                                'MKioskTy8', 'MKioskTy9', 'MKioskTy10', 'MKioskTy11']])

    # 컬럼명 변경
    column_change = {'hpid': '병원코드',
                     'dutyName': '병원명',
                     'dutyAddr': '주소',
                     'dutyTel3': '응급연락처',
                     'wgs84Lat': '위도',
                     'wgs84Lon': '경도',
                     'hperyn': '응급실수',
                     'hpopyn': '수술실수',
                     'hvec': '가용응급실수',
                     'hvoc': '가용수술실수',
                     'MKioskTy1': '뇌출혈',
                     'MKioskTy2': '뇌경색',
                     'MKioskTy3': '심근경색',
                     'MKioskTy4': '복부손상',
                     'MKioskTy5': '사지접합',
                     'MKioskTy7': '응급투석',
                     'MKioskTy8': '조산산모',
                     'MKioskTy10': '신생아',
                     'MKioskTy11': '중증화상'
                     }
    solution_df = solution_df.rename(columns=column_change)
    solution_df = solution_df.replace({"정보미제공": "N"})

    # 응급실 가용율, 포화도 추가
    
    solution_df.loc[solution_df['가용응급실수'] < 0, '가용응급실수'] = 0
    solution_df.loc[solution_df['가용수술실수'] < 0, '가용수술실수'] = 0

    solution_df['응급실가용율'] = round(solution_df['가용응급실수'] / solution_df['응급실수'], 2)
    solution_df.loc[solution_df['응급실가용율'] > 1,'응급실가용율']=1
    solution_df['응급실포화도'] = pd.cut(solution_df['응급실가용율'], bins=[-1, 0.1, 0.3, 0.6, 1], labels=['불가', '혼잡', '보통', '원활'])

    ### 중증 질환 수용 가능한 병원 추출
    ### 미션1 상황에 따른 병원 데이터 추출하기 참고

    if special_m == "중증 아님":
        condition1 = (solution_df['응급실포화도'] != '불가')
        distance_df = solution_df[condition1].copy()
    else:
        condition1 = (solution_df[special_m] == 'Y') & (solution_df['가용수술실수'] >= 1)
        condition2 = (solution_df['응급실포화도'] != '불가')

        distance_df = solution_df[condition1 & condition2].copy()

    ### 환자 위치로부터의 거리 계산
    distance = []
    patient = (lati, long)
    
    for idx, row in distance_df.iterrows():
        distance.append(round(haversine((row['위도'], row['경도']), patient, unit='km'), 2))

    distance_df['거리'] = distance
    distance_df['거리구분'] = pd.cut(distance_df['거리'], bins=[-1, 2, 5, 10, 100],
                                 labels=['2km이내', '5km이내', '10km이내', '10km이상'])
            
    return distance_df



data = pd.read_csv('119_emergency_dispatch.csv', encoding="cp949")

## 오늘 날짜
now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
now_date2 = datetime.datetime.strptime(now_date.strftime("%Y-%m-%d"), "%Y-%m-%d")

## 2023년 최소 날짜, 최대 날짜
first_date = pd.to_datetime("2023-01-01")
last_date = pd.to_datetime("2023-12-31")

## 출동 이력의 최소 날짜, 최대 날짜
min_date = datetime.datetime.strptime(data['출동일시'].min(), "%Y-%m-%d")
max_date = datetime.datetime.strptime(data['출동일시'].max(), "%Y-%m-%d")


# -------------------- 필요 함수 생성 코딩 END --------------------

# -------------------- ▼ 1-0그룹 Streamlit 웹 화면 구성 Tab 생성 START ▼ --------------------

# 레이아웃 구성하기 
st.set_page_config(layout="wide")

st.image("./119.png", width=300)
# tabs 만들기 
tab1, tab2 = st.tabs(["출동 일지", "대시보드"])

# tab1 내용물 구성하기 
with tab1:

    # 제목 넣기
    st.markdown("## 119 응급 출동 일지")
    
    # 시간 정보 가져오기 
    now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)

    
    # 환자정보 널기
    st.markdown("#### 환자 정보")

    ## -------------------- ▼ 1-1그룹 날짜/시간 입력 cols 구성(출동일/날짜정보(input_date)/출동시간/시간정보(input_time)) ▼ --------------------
     
    col110, col111, col112, col113 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col110:
        st.info("출동일")
    with col111:
        input_date = st.date_input("날짜정보", label_visibility="collapsed")
    with col112:
        st.info("출동시간")
    with col113:
        input_time = st.time_input("시간정보",label_visibility="collapsed")

    ## -------------------------------------------------------------------------------------


    ## -------------------- ▼ 1-2그룹 이름/성별 입력 cols 구성(이름/이름 텍스트 입력(name)/나이/나이 숫자 입력(age)/성별/성별 라디오(patient_s)) ▼ --------------------

    col120, col121, col122, col123, col124, col125 = st.columns([0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
    with col120:
        st.info("이름")
    with col121:
        name = st.text_input("이름 입력",label_visibility="collapsed")
    with col122:
        st.info("나이")
    with col123:
        age = st.number_input("나이 입력", 1, 200, label_visibility="collapsed")
    with col124:
        st.info("성별")
    with col125:
        patient_s = st.radio("성별 선택", ['남', '여'], horizontal=True, label_visibility="collapsed")

   ##-------------------------------------------------------------------------------------

    
    ## -------------------- ▼ 1-3그룹 체온/환자위치(주소) 입력 cols 구성(체온/체온 숫자 입력(fever)/환자 위치/환자위치 텍스트 입력(location)) ▼ --------------------

    col130, col131, col132, col133 = st.columns([0.1, 0.3, 0.1, 0.3]) # col 나누기
    with col130:
        st.info("체온")
    with col131:
        fever = st.number_input("체온 입력", 10.0, 50.0, 36.5, label_visibility="collapsed")
    with col132:
        st.info("환자 위치")
    with col133:
        location = st.text_input("환자 위치 주소로 입력",label_visibility="collapsed")
    
    ##-------------------------------------------------------------------------------------

    ## ------------------ ▼ 1-4그룹 혈압 입력 cols 구성(수축기혈압/수축기 입력 슬라이더(high_blood)/이완기혈압/이완기 입력 슬라이더(low_blood)) ▼ --------------------
    ## st.slider 사용

    col140, col141, col142, col143 = st.columns([0.1, 0.3, 0.1, 0.3]) # col 나누기
    with col140:
        st.info("수축기혈압")
    with col141:
        high_blood = st.slider('수축기혈압', 140, 90, label_visibility="collapsed") # 140이상 고혈압, 90이하 저혈압
    with col142:
        st.info("이완기혈압")
    with col143:
        low_blood = st.slider('이완기혈압', 90, 60, label_visibility="collapsed") # 90이상 고혈압, 60이하 저혈압

        
    st.markdown("#### 증상 체크하기")

    col150, col151, col152, col153, col154, col155= st.columns([0.2,0.1,0.1,0.1,0.1,0.1]) # col 나누기
    with col150:
        st.info("증상 체크")
    with col151:
        cough_check = st.checkbox("기침")
        convulsion_check = st.checkbox("간헐적 경련")
        dizziness_check = st.checkbox("현기증")
    with col152:
        paralysis_check = st.checkbox("마비")
        insensitive_check = st.checkbox("무감각")
        speechlessness_check = st.checkbox("말이 어눌해짐")
    with col153:        
        pain_check = st.checkbox("통증")
        touch_pain_check = st.checkbox("만지면 아픔")
        pass_out_check = st.checkbox("졸도")
    with col154:
        diarrhea_check = st.checkbox("설사")
        bleeding_check = st.checkbox("출혈")
    with col155:        
        blurred_check = st.checkbox("시력 저하")
        breath_check = st.checkbox("호흡 곤란")


    ## -------------------------------------------------------------------------------------
    
    ## -------------------- ▼ 1-6그룹 중증 질환 여부, 중증 질환 판단(special_yn) col 구성 ▼ --------------------
    ## selectbox  사용(변수: special_yn)
    
    col160, col161 = st.columns([0.3, 0.7]) # col 나누기
    with col160:
        st.info("AI 모델을 이용한 중증질환 예측 여부")
    with col161:
        special_yn = st.selectbox('중증질환', ('--', '예측하기', '기타 중증질환', '중증질환 없음', ), label_visibility="collapsed")

    
    ## -------------------- ▼ 1-7그룹 중증 질환 선택 또는 예측 결과 표시 cols 구성 ▼ --------------------
    
    col171, col172 = st.columns([0.3, 0.7]) # col 나누기

    with col171:
        st.info("AI 예측 결과")
    with col172:
        if special_yn == "예측하기":


            patient_data = {
                "체온": [fever],
                "수축기 혈압": [high_blood],
                "이완기 혈압": [low_blood],
                "호흡 곤란": [int(breath_check)],
                "간헐성 경련": [int(convulsion_check)],
                "설사": [int(diarrhea_check)],
                "기침": [int(cough_check)],
                "출혈": [int(bleeding_check)],
                "통증": [int(pain_check)],
                "만지면 아프다": [int(touch_pain_check)],
                "무감각": [int(insensitive_check)],
                "마비": [int(paralysis_check)],
                "현기증": [int(dizziness_check)],
                "졸도": [int(pass_out_check)],
                "말이 어눌해졌다": [int(speechlessness_check)],
                "시력이 흐려짐": [int(blurred_check)],
                "중증질환": [""]
            }
            
            # AI 모델 중증질환 예측 함수 호출
            special_m = predict_disease(patient_data)
            
            st.markdown(f"### 예측된 중증 질환은 {special_m}입니다")


        elif special_yn == "기타 중증질환":
            special_m = st.radio("기타 중증질환",
                                    ['뇌출혈', '신생아', '중증화상', "뇌경색", "심근경색", "복부손상", "사지접합",  "응급투석", "조산산모"],
                                    horizontal=True)
        elif special_yn == "--":

            st.write("중증 질환 예측은 뇌출혈, 뇌경색, 심근경색, 복부손상 4가지만 분류됩니다.")
            st.write("이외의 중증 질환으로 판단될 경우, '기타 중증질환'을 통해 직접 선택하세요")
        else:
            special_m = "중증질환 없음"
            st.write("중증이 아니라고 판단될 시 휴식 후, 그래도 증상이 지속된다면 가까운 병원에 내원하세요.")



    # ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼  [도전미션] ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ 
    
    ## -------------------- ▼ 1-8그룹 가용병원 표시 폼 지정 ▼ --------------------
    
    with st.form(key='tab1_first'):
        
        ### 병원 조회 버튼 생성
        if st.form_submit_button(label="병원조회"):

            #### 거리주소 -> 위도/경도 변환 함수 호출
            lati, long = geocoding(location)

            #### 인근 병원 찾기 함수 호출
            hospital_list =  find_hospital(special_m, lati, long)
            
            #### 필요 병원 정보 추출 
            display_column = ['병원명', "주소", "응급연락처", "응급실수", "수술실수", "가용응급실수", "가용수술실수", '응급실포화도', '거리', '거리구분']
            display_df = hospital_list[display_column].sort_values(['거리구분', '응급실포화도', '거리'], ascending=[True, False, True])
            display_df.reset_index(drop=True, inplace=True)

            #### 추출 병원 지도에 표시
            with st.expander("인근 병원 리스트", expanded=True):
                st.dataframe(display_df)
                m = folium.Map(location=[lati,long], zoom_start=11)
                icon = folium.Icon(color="red")
                folium.Marker(location=[lati, long], popup="환자위치", tooltip="환자위치: "+location, icon=icon).add_to(m)

                
                ###### folium을 활용하여 지도 그리기 (3일차 교재 branca 참조)
                st.markdown("지도 2")
                for idx, row in hospital_list[:5].iterrows():

                    html = """<!DOCTYPE html>
                    <html>
                        <table style="height: 126px; width: 330px;"> <tbody> <tr>
                            <td style="background-color: #2A799C;">
                            <div style="color: #ffffff;text-align:center;">병원명</div></td>
                            <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['병원명'])+"""</tr>
                            <tr><td style="background-color: #2A799C;">
                            <div style="color: #ffffff;text-align:center;">위도</div></td>
                            <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['위도'])+"""</tr>
                            <tr><td style="background-color: #2A799C;">
                            <div style="color: #ffffff;text-align:center;">경도</div></td>
                            <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['경도'])+"""</tr>
                        </tbody> </table> </html> """
                                    
                    iframe = branca.element.IFrame(html=html, width=350, height=150)
                    popup_text = folium.Popup(iframe,parse_html=True)
                    icon = folium.Icon(color="blue")
                
                    folium.Marker(location=[row['위도'], row['경도']],
                        popup=popup_text, tooltip=row['병원명'], icon=icon).add_to(m)

                st_folium(m, width=1000)



    #  -------------------- ▼ 1-9그룹 완료시간 저장 폼 지정 ▼  --------------------
    with st.form(key='tab1_second'):

        ## 완료시간 시간표시 cols 구성
        col180, col181 = st.columns([0.3, 0.7]) # col 나누기
        with col180:
            st.info("환자 이송 완료 시간")
        with col181:
            end_time = st.time_input('완료시간입력', label_visibility="collapsed")

        ## 완료시간 저장 버튼
        if st.form_submit_button(label='csv 파일에 기록하기'):
            dispatch_data = pd.read_csv('./119_emergency_dispatch.csv', encoding="cp949" )
            id_num = list(dispatch_data['ID'].str[1:].astype(int))
            max_num = np.max(id_num)
            max_id = 'P' + str(max_num)
            elapsed = (end_time.hour - input_time.hour)*60 + (end_time.minute - input_time.minute)

            check_condition1 = (dispatch_data.loc[dispatch_data['ID'] ==max_id, '출동일시'].values[0]  == str(input_date))
            check_condition2 = (dispatch_data.loc[dispatch_data['ID']==max_id, '이름'].values[0] == name)

            ## 마지막 저장 내용과 동일한 경우, 내용을 update 시킴
            
            if check_condition1 and check_condition2:
                dispatch_data.loc[dispatch_data['ID'] == max_id, '나이'] = age
                dispatch_data.loc[dispatch_data['ID'] == max_id, '성별'] = patient_s
                dispatch_data.loc[dispatch_data['ID'] == max_id, '체온'] = fever
                dispatch_data.loc[dispatch_data['ID'] == max_id, '수축기 혈압'] = high_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이완기 혈압'] = low_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '호흡 곤란'] = int(breath_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '간헐성 경련'] = int(convulsion_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '설사'] = int(diarrhea_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '기침'] = int(cough_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '출혈'] = int(bleeding_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '통증'] = int(pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '만지면 아프다'] = int(touch_pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '무감각'] = int(insensitive_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '마비'] = int(paralysis_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '현기증'] = int(dizziness_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '졸도'] = int(pass_out_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '말이 어눌해졌다'] = int(speechlessness_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '시력이 흐려짐'] = int(blurred_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '중증질환'] = special_m
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이송 시간'] = int(elapsed)

            else: # 새로운 출동 이력 추가하기
                new_id = 'P' + str(max_num+1)
                new_data = {
                    "ID" : [new_id],
                    "출동일시" : [str(input_date)],
                    "이름" : [name],
                    "성별" : [patient_s],
                    "나이" : [age],
                    "체온": [fever],
                    "수축기 혈압": [high_blood],
                    "이완기 혈압": [low_blood],
                    "호흡 곤란": [int(breath_check)],
                    "간헐성 경련": [int(convulsion_check)],
                    "설사": [int(diarrhea_check)],
                    "기침": [int(cough_check)],
                    "출혈": [int(bleeding_check)],
                    "통증": [int(pain_check)],
                    "만지면 아프다": [int(touch_pain_check)],
                    "무감각": [int(insensitive_check)],
                    "마비": [int(paralysis_check)],
                    "현기증": [int(dizziness_check)],
                    "졸도": [int(pass_out_check)],
                    "말이 어눌해졌다": [int(speechlessness_check)],
                    "시력이 흐려짐": [int(blurred_check)],
                    "중증질환": [special_m],
                    "이송 시간" : [int(elapsed)]
                }

                new_df= pd.DataFrame(new_data)
                dispatch_data = pd.concat([dispatch_data, new_df], axis=0, ignore_index=True)

            dispatch_data.to_csv('./119_emergency_dispatch.csv', encoding="cp949", index=False)

 

with tab2:


    st.markdown("## 대시보드")
    
    st.info('금일 출동 내역')

    today_date = now_date.strftime("%Y-%m-%d")
    today_count = len(data[data['출동일시']==today_date])

    if today_count > 0 :
        st.dataframe(data[data['출동일시'] == today_date])
    else:
        st.markdown("금일 출동내역이 없습니다.")

    
    st.info("선택한 기간과 주기에 따른 출동건수와 중증질환자 통계량을 확인할 수 있습니다.")

    col210, col211, col212 = st.columns([0.3,0.3,0.3])  
    with col210:
        slider_date = st.slider("일간", min_value=min_date, max_value=max_date, value=(min_date,now_date2))
    with col211:
        slider_week = st.slider("주간", min_value=min_date, max_value=max_date, step=datetime.timedelta(weeks=1), value=(min_date,now_date2))
    with col212:
        slider_month = st.slider("월간", min_value=min_date, max_value=max_date, step=datetime.timedelta(weeks=1), value=(min_date,now_date2), format="YYYY-MM")
 
 
    ## 선택된 일자의 data 추출
    # data['datetime'] = pd.to_datetime(data['출동일시'])
    # day_list_df = data[data['datetime'].dt.date == slider_date[0].date()]
    data['datetime'] = pd.to_datetime(data['출동일시'])
    day_list_df = data[(data['datetime'].dt.date >= slider_date[0].date()) & (data['datetime'].dt.date <= slider_date[1].date())]

    ## 선택된 주간의 data 추출
    
    data['주별'] = data['datetime'].dt.strftime("%W").astype(int)
    
    min_week = int(slider_week[0].strftime("%W"))
    max_week = int(slider_week[1].strftime("%W"))
    week_list_df = data[(data['주별'] >= min_week) & (data['주별'] <= max_week)]
        

    ## 선택된 월의 data 추출
    
    data['월별'] = data['datetime'].dt.month.astype(int)
    min_month = slider_month[0].month
    max_month = slider_month[1].month
    
    month_list_df = data[(data['월별'] >= min_month) & (data['월별'] <= max_month)]

#     ## -------------------------------------------------------------------

    select_bins = st.radio("주기", ('일별', '주별', '월별'), horizontal=True)
    st.error("출동 건수")

    if select_bins == '일별':
        group_day = day_list_df.groupby(by=['datetime'], as_index=False)['ID'].count()
        group_day = group_day.rename(columns={"ID": '출동건수', 'datetime': '일별'})
        group_day['일별'] = group_day['일별'].dt.strftime('%Y-%m-%d')
        group_day = group_day.sort_values('일별', ascending=True)
        st.bar_chart(data=group_day, x='일별', y='출동건수', use_container_width=True)

        st.warning("중증 질환별 통계")
        
        group_day_disease = day_list_df.groupby(by=['중증질환'], as_index=False)['ID'].count()
        group_day_disease = group_day_disease.rename(columns={"ID": '출동건수', '중증질환': '중증질환'})
        group_day_disease = group_day_disease.sort_values('출동건수', ascending=False)

        fig = px.pie(group_day_disease, values='출동건수', names='중증질환', 
                    title='일간 중증질환자 통계량')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.2, xanchor="right", x=1))
        st.plotly_chart(fig)


    elif select_bins=='주별':

        group_week = week_list_df.groupby(by='주별', as_index=False)['ID'].count()
        group_week = group_week.rename(columns={"ID" : '출동건수'})
        group_week = group_week.sort_values('주별', ascending=True)

        st.bar_chart(data=group_week, x='주별', y='출동건수', use_container_width=True)

        st.warning("중증 질환별 통계")

        group_week_disease = week_list_df.groupby(by=['중증질환'], as_index=False)['ID'].count()
        group_week_disease = group_week_disease.rename(columns={"ID": '출동건수', '중증질환': '중증질환'})
        group_week_disease = group_week_disease.sort_values('출동건수', ascending=False)

        fig = px.pie(group_week_disease, values='출동건수', names='중증질환', 
                    title='주간 중증질환자 통계량')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.2, xanchor="right", x=1))
        st.plotly_chart(fig)

    else:

        group_month = month_list_df.groupby(by='월별', as_index=False)['ID'].count()
        group_month = group_month.rename(columns={"ID": '출동건수'})
        group_month = group_month.sort_values('월별', ascending=True)

        st.bar_chart(data=group_month, x='월별', y='출동건수', use_container_width=True)

        st.warning("중증 질환별 통계")

        group_month_disease = month_list_df.groupby(by=['중증질환'], as_index=False)['ID'].count()
        group_month_disease = group_month_disease.rename(columns={"ID": '출동건수', '중증질환': '중증질환'})
        group_month_disease = group_month_disease.sort_values('출동건수', ascending=False)

        fig = px.pie(group_month_disease, values='출동건수', names='중증질환', 
                    title='월간 중증질환자 통계량')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.2, xanchor="right", x=1))
        st.plotly_chart(fig)

    
    st.info("이송시간 통계")

    col230, col231, col232 = st.columns([0.3, 0.3, 0.3])
    with col230:

        group_day_time = data.groupby(by=['출동일시'], as_index=False)['이송 시간'].mean()
        group_day_time = group_day_time.rename(columns={"이송 시간" : '평균 이송 시간'})
        st.line_chart(data=group_day_time, x='출동일시', y='평균 이송 시간', use_container_width=True)
    
    with col231:

        group_week_time = data.groupby(by=['나이'], as_index=False)['이송 시간'].mean()
        group_week_time = group_week_time.rename(columns={"이송 시간": '평균 이송 시간'})
        st.line_chart(data=group_week_time, x='나이', y='평균 이송 시간', use_container_width=True)


    with col232:

        group_month_time = data.groupby(by=['중증질환'], as_index=False)['이송 시간'].mean()
        group_month_time = group_month_time.rename(columns={"이송 시간": '평균 이송 시간'})
        st.line_chart(data=group_month_time, x='중증질환', y='평균 이송 시간', use_container_width=True)
