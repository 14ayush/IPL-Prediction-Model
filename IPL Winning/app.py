import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipe = pickle.load(open('C:\\Users\\aayus\\ML(PROJECT)\\IPL Winning\\win.pkl', 'rb'))

# Teams and cities list
teams = [
    'Royal Challengers Bangalore', 'Kings XI Punjab', 'Mumbai Indians',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Chennai Super Kings',
    'Sunrisers Hyderabad', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# UI Layout
st.title('🏏 IPL Win Predictor')

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #f1c40f;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #e67e22;
    }
    .stSelectbox>div>div>input {
        background-color: #ecf0f1;
    }
    .stDataFrame {
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stTitle {
        color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

# City selection
selected_city = st.selectbox('Select the Host City', sorted(cities))

# Target input
target = st.number_input('Target Score', min_value=1, step=1)

st.markdown("---")
st.subheader('Match Situation')

# Match inputs
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    over = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Winning Probability'):

    if over == 0 or (120 - (over * 6)) == 0:
        st.error("⚠️ Overs must be greater than 0 and match not fully completed.")
    else:
        # Feature engineering
        runs_left = target - score
        balls_left = 120 - (over * 6)
        wickets_left = 10 - wickets
        current_run_rate = score / over
        required_run_rate = (runs_left * 6) / balls_left

        # Dataframe for model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'run_left': [runs_left],
            'ball_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'curr': [current_run_rate],
            'reqrun': [required_run_rate]
        })

        # Show inputs
        st.dataframe(input_df)

        # Prediction
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display result
        st.markdown("---")
        st.subheader("🔮 Win Probability")
        st.success(f"🏏 {batting_team}: **{round(win_prob * 100, 2)}%**")
        st.info(f"🎯 {bowling_team}: **{round(loss_prob * 100, 2)}%**")
