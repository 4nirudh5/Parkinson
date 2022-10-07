import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title = None,
    options = ["Home","Predict","About"], 
    icons = ["house","speedometer","person"], 
    menu_icon = "menu-button", 
    orientation = "horizontal",
)

if selected == "Home":
    with st.container():
        st.title('Parkinson disease predictor!')
        st.subheader('By Anirudh :wave:')
    with st.container():
        st.title('Predict wheter you have parkionson disease :information_source:')
        st.write('')
        st.subheader('Disclaimer :heavy_exclamation_mark:')
        st.write('')
        st.subheader('This is a sample Dataset:')
        df = pd.read_csv("parkinsonpp.csv")
        st.write(df)


if selected == "Predict":
    st.title('Predict whether you have the disease')
    st.subheader('Enter the details')
    
    model = st.selectbox('choose a model',
      ('K-nearest Neighbour', 'Decision Tree', 'Naive Bayes Classification','Random Forest'))

    if model =='K-nearest Neighbour':
       m = pickle.load(open('knn_sm.sav', 'rb'))
    elif model =='Decision Tree':
       m = pickle.load(open('decisiontree_sm.sav', 'rb'))
    elif model == 'Naive Bayes Classification':
       m = pickle.load(open('naive_sm.sav', 'rb'))  
    else:
        m = pickle.load(open('randomforest_sm.sav', 'rb'))

    def pp(input_data):
        na = np.asarray(input_data)
        d = na.reshape(1,-1)
        
        prediction = m.predict(d)
        return prediction


    def main():

        MDVPFoHz = st.text_input('MDVPFoHz')
        MDVPFhiHz = st.text_input('MDVPFhiHz')
        MDVPFloHz = st.text_input('MDVPFloHz')
        MDVPJitter = st.text_input('DVPJitter')
        MDVPJitterAbs = st.text_input('MDVPJitterAbs')
        MDVPRAP = st.text_input('MDVPRAP') 
        MDVPPPQ = st.text_input('MDVPPPQ')
        JitterDDP = st.text_input('JitterDDP')
        MDVPShimmer = st.text_input('MDVPShimmer')
        MDVPShimmerdB = st.text_input('MDVPShimmerdB') 
        ShimmerAPQ3 = st.text_input('ShimmerAPQ3')
        ShimmerAPQ5 = st.text_input('ShimmerAPQ5') 
        MDVPAPQ = st.text_input('MDVPAPQ') 
        ShimmerDDA = st.text_input('ShimmerDDA') 
        NHR = st.text_input('NHR') 
        HNR = st.text_input('HNR') 
        RPDE = st.text_input('RPDE')
        DFA = st.text_input('DFA')
        spread1 = st.text_input('spread 1')
        spread2 = st.text_input('spread 2')
        D2 = st.text_input('D2')
        PPE = st.text_input('PPE')

        Parkinson_finder = ''
        if st.button('Get Results'):
           Parkinson_finder  = pp([MDVPFoHz,MDVPFhiHz,MDVPFloHz,MDVPJitter,MDVPJitterAbs,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmerdB,ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])  
           st.success(Parkinson_finder)

    if __name__ == '__main__':
        main()

if selected == "About":
    #st.title('About')
    st.title('Hi, Anirudh here :smiley:')
    st.subheader('Hope this app was useful to you :innocent:')
    st.write('I am a student as of now who is passionate about Data Science and this is a insurance prdictor which uses linear regression to predict the cost of the insurance but the model accuracy is low ie 0.7454474167181153 so it is not exact but it is enough to get a idea. see the sample dataset to know how to fill in details.')
    st.write('[More projects in GITHUB >](https://github.com/4nirudh5)')
