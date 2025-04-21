import streamlit as st
import pickle
import pandas as pd
teams = ['Royal Challengers Bangalore',
         'Kings XI Punjab',
         'Mumbai Indians',
         'Kolkata Knight Riders',
         'Rajasthan Royals',
         'Chennai Super Kings',
         'Sunrisers Hyderabad',
         'Delhi Capitals']
cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru']
pipe=pickle.load(open('C:\\Users\\aayus\\ML(PROJECT)\\IPL Winning\\pipe.pkl','rb'))
st.title('IPL WIN PREDICTOR')
col1,col2=st.columns(2)

with col1:
   batting_team= st.selectbox('Select The batting Team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select The Bowling Team', (teams))
selected_city=st.selectbox('Select The Host City',sorted(cities))
target=st.number_input('Target')
st.header('Enter the necessary inputs')
col3,col4,col5= st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    over=st.number_input('Over completed')
with col5:
    wicket=st.number_input('Wickets falls')
if st.button('Probability Of winning'):
    run_left=target-score
    ball_left=120-(over*6)
    wicket=10-wicket
    curr=score/over
    reqrun=(run_left*6)/ball_left
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                           'city':[selected_city],'run_left':[run_left],'ball_left':[ball_left],
                           'wicket':[wicket],'total_runs_x':[target],'curr':[curr],'reqrun':[reqrun]})
    st.table(input_df)
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team+"-"+str(round(win*100))+"%")
    st.header(bowling_team + "-" + str(round(loss * 100)) + "%")

