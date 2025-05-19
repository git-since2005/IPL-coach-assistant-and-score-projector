import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

batsmans = pd.read_csv("batsman.csv").iloc[:, 1:]
bowlers = pd.read_csv("bowlers.csv").iloc[:, 1:]
bowler_desc = bowlers.describe()
batsman_desc = batsmans.describe()
with st.sidebar:
    st.header("Abhishek's Data Visualisation for IPL")
    selection = st.selectbox("Select a visualisation option", ['Coaching Assistant', 'Score Prediction'])
if selection=="Coaching Assistant":
    st.markdown("<style> .big-font {font-size:50px !important; margin-left:-10px}</style>", unsafe_allow_html=True)
    st.markdown('<p class="big-font">Select a name for personalized data of players:</p>', unsafe_allow_html=True)
    name = st.selectbox("Player", list(batsmans['batsman'])+list(bowlers['bowlers']), index = 1)
    find_bowler = bowlers[bowlers['bowlers']==name]
    balls, extras,wickets,catches, mom,matches = find_bowler['balls'],find_bowler['extras'], find_bowler['wickets'],find_bowler['catches'], find_bowler['mom'], find_bowler['matches']
    divs1, divs2 = st.columns(2)
    with divs1:
        bowling = st.checkbox("Bowling", value=False)
    with divs2:    
        batting = st.checkbox("Batting", value = False)
    if bowling:
        st.success(name)
        data = {}
        if sum(name == bowlers['bowlers']):
            for i in bowlers.columns[1:]:
                var = list(bowlers[bowlers['bowlers'] == name][i])
                if var:
                    data[i] = var[0]
                    st.write(i+":", var[0])
            st.subheader("Advices")
            advices1 = False
            if bowler_desc['wickets']['75%']>data['wickets']:
                st.markdown("- Must practice bowling")
                advices1 = True
            if bowler_desc['extras']['75%']>data['extras']:
                st.markdown("- Must not give extras")
                advices1 = True
            if bowler_desc['catches']['75%']>data['catches']:
                st.markdown("- Catch Practice")
                advices1 = True
            if advices1==False:
                st.success("No advices")
            
            st.subheader(":heavy_plus_sign: Points")
            if bowler_desc['catches']['75%']<data['catches']:st.markdown("- Takes catches for team")
            if bowler_desc['mom']['75%']<data['mom']:st.markdown("- Takes more wickets for team")
            if bowler_desc['wickets']['75%']<data['wickets']:st.markdown("- Takes more wickets from overall players")
            if bowler_desc['matches']['75%']<data['matches']:st.markdown("- Experienced player")


        else:
            st.markdown("<style>.text{font-family: Monospace;}</style>", unsafe_allow_html=True)
            st.markdown("<p class='text'>There is no data of this players in bowling</p>", unsafe_allow_html=True)
    elif batting:
        st.success(name)
        data = {}
        if sum(name == batsmans['batsman']):
            for i in batsmans.columns[1:]:
                var = list(batsmans[batsmans['batsman'] == name][i])
                if var:
                    data[i] = var[0]
                    st.write(i+":", var[0])
            st.subheader("Advices")
            advices = False
            if batsman_desc['sixes']['75%']>data['sixes'] and batsman_desc['fours']['75%']>data['fours']:
                st.markdown("- Hitting Practice")
                advices = True
            if batsman_desc['singles']['75%']<data['singles']:
                st.markdown("- Running between the wickets")
                advices = True
            if batsman_desc['sr']['75%']>data['sr']:
                st.markdown("- Must increase strike rate")
                advices = True
            if advices==False:
                st.success("No advices")
            
            st.subheader(":heavy_plus_sign: Points")
            if batsman_desc['sr']['75%'] < data['sr']:
                st.markdown("- Strike rate is higher by overall players")
            if batsman_desc['runs']['75%']<data['runs']:
                st.markdown("- Contriubutes in team runs")
            if batsman_desc['sixes']['75%']<data['sixes'] or batsman_desc['fours']['75%']<data['fours']:
                st.markdown("- Hits boundaries")
            if batsman_desc['mom']['75%']<data['mom']:
                st.markdown("- Takes more runs for team")
            if batsman_desc['matches']['75%']<data['matches']:st.markdown("- More experienced player")

        else:
            st.markdown("<style>.text{font-family: Monospace;}</style>", unsafe_allow_html=True)
            st.markdown("<p class='text'>There is no data of this players in batting</p>", unsafe_allow_html=True)
elif selection == 'Score Prediction':
    st.header("Score Predictor")
    st.subheader("Team must have completed 8 overs to predict the score:")
    over = st.checkbox("Played 8 Overs", value = True)
    st.warning("Caution! \n You have to select runs made in 8 overs one by one. If eight instances are not selected it will throw an error")
    st.success("Type in numbers 8 times for 8 overs.")
    options = [[j for j in range(1, 37)] for i in range(8)]
    runs_per_over = []
    for i in range(8):
        run = st.number_input(f"Enter {i+1}th over runs", min_value=0, max_value=36)
        runs_per_over.append(run)
    df = pd.DataFrame()
    submit = st.button("Submit")
    if submit and len(runs_per_over)==8:
        df['overs'] = [i for i in range(1, 9)]
        df['runs'] = runs_per_over
        x_train, y_train, x_test, y_test = train_test_split(df['overs'], df['runs'], random_state=42, test_size=0.3)
        X,y = df.overs.values.reshape(-1, 1), df.runs.values.reshape(-1, 1)
        lr = LinearRegression()
        rfclf = RandomForestClassifier()
        rfclf.fit(df.overs.values.reshape(-1, 1),df.runs.values.reshape(-1, 1))
        lr.fit(df.overs.values.reshape(-1, 1), df.runs.values.reshape(-1, 1))
        proj = sum(lr.predict(pd.Series([i for i in range(9, 21)]).values.reshape(-1, 1)))+sum(runs_per_over)
        proj1 = sum(rfclf.predict(pd.Series([i for i in range(9, 21)]).values.reshape(-1, 1)))+sum(runs_per_over)
        st.success(f"By linear regression: {int(proj)} runs.")
        st.success(f"By RandomForestClassifier: {int(proj1)} runs.")
        that = pd.DataFrame()
