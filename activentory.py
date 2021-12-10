# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:07:08 2021

@author: livel
"""

import streamlit as st
import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='Enter demo name. Options are \n1. \'radio\' \n2. text')
parser.add_argument('--demo', type=str, dest='demo', help='Enter valid demo name from above options.')
args = parser.parse_args()

#Radio Button
def radio():
    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:
    left_column.button('Press me!')
    
    # Or even better, call Streamlit functions inside a "with" block:
    with right_column:
        chosen = st.radio(
            'Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
        st.write(f"You are in {chosen} house!")
        
def line():
    chart_data = pd.DataFrame(
         np.random.randn(20, 3),
         columns=['a', 'b', 'c'])
    
    st.line_chart(chart_data)
    
def show_map():
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])
    
    st.map(map_data)
        
def select():
    option = st.selectbox(
        'Which number do you like best?',
         [0,1,2])
    
    'You selected: ', option
    

def map_cols():
    import folium as fl
    from IPython import display
    from streamlit_folium import folium_static
    import time
    import pandas as pd
    import streamlit as st
    import numpy as np
    import pickle as pkl
    import os
    import matplotlib.pyplot as plt
    import graphviz as graphviz
    st.set_page_config(layout='wide')
    from PIL import Image
    
    lat = 40.44283008724992
    long = -79.98941715251705
    main_map = fl.Map(location=[lat, long], default_zoom_start=15,width='65%',height='65%')
    n_queries = 40

    
    df_map = pd.read_csv('LSL_property_merged.csv')
    st.sidebar.write('## Creating Lead Service Line (LSL) Inventory from scratch')
    st.sidebar.write('With our KETOS Shield (TM) and Activentory (TM) solution, utilities can build an inventory of lead service lines from the ground up.')
    st.sidebar.write('')
    st.sidebar.write('## How does it work?')
    st.sidebar.write('Activentory (TM) is a machine learning model that takes in commonly available attributes such as house area, price sold, year built among other commonly available property and geographical attributes.')
    st.sidebar.write('The model suggests the most promising locations to test water for lead.')
    st.sidebar.write('Shield (TM) is KETOS\' groundbreaking water quality monitoring solution that can test for lead levels at identified locations with high accuracy in a matter of minutes. Shield allows utilities to quickly assess above-threshold lead presence on-site.')
    st.sidebar.write('The presence or absence of lead at a given site is fed back to the predictive model. As the model gets fed data from multiple sites, its accuracy improves.')
    st.sidebar.write('At an accuracy of 70-80% the model can be used to assess the likelihood of lead service line presence of the entire service area.')
    
    c1,c2, c3 = st.columns([0.25,0.5,0.25])
    with c1:
        st.empty()
    with c2:
        image = Image.open('Activentory.png')
        st.image(image, use_column_width='auto', clamp=True)
    with c3:
        st.empty()
        
    c1,c2,c3 = st.columns([2.5,1,2.5])
    
    with c1:
        
        # for i in range(n_queries):
        #display.clear_output(wait=False)
        idx = np.random.randint(0,df_map.shape[0])
        row = df_map.iloc[idx]
        lat = row['Lat']
        long = row['Long']
    
        fl.Marker(location=[lat, long],popup='Location likely to contain LSL',tooltip = "Click for more information",icon=fl.Icon(color='blue')).add_to(main_map)
        #display.display(main_map)
    
        color = "green"
        info = "Location does not contain LSL"
        option = 'No'
        if option == "Yes":
            color = "red"
            info = "Location contains LSL"
        # Read data
        cache_path = 'cache.csv'
        df = pd.read_csv(cache_path)
        entry = {'Lat':lat, 'Long':long, 'color':color}
        df = df.append(entry, ignore_index=True)
        df.to_csv(cache_path, index=False)
    
        df.apply(lambda row: fl.Marker(location=[row.Lat, row.Long],popup=info,tooltip = "Click for more information",icon=fl.Icon(color=row.color)).add_to(main_map), axis=1)
    
        # fl.Marker(location=[lat, long],popup=info,tooltip = "Click for more information",icon=fl.Icon(color=color)).add_to(main_map)
        folium_static(main_map)
    
    with c2:
        st.write('')
        st.write('')
        st.write('')
        st.write('Address containing potential LSL.')
        st.write(row['Address'])
        st.write('Test this location for Lead with KETOS Shield (TM)')
        st.write('')
        option = st.selectbox("Does this location contain an LSL?",("Select","Yes","No","stop"),0)

    with c3:
        acc_cache_path = 'acc_cache.csv'
        if os.path.exists(acc_cache_path):
            acc_df = pd.read_csv(acc_cache_path)
            mu,sigma = 0.75,0.1
            acc = np.random.normal(mu,sigma)
            acc_df = acc_df.append({'accuracy':acc},ignore_index=True)
            acc_df.to_csv(acc_cache_path,index=False)
            fig,ax = plt.subplots(figsize=(4,3))
            ax.plot(100*acc_df['accuracy'],'bo-')
            ax.set_title('Model Accuracy versus training iterations')
            ax.set_xlabel('Training Iterations')
            ax.set_ylabel('Percent Accuracy')
            st.pyplot(fig,)
        else:
            acc_df = pd.DataFrame(columns=['accuracy'])
            acc_df.to_csv(acc_cache_path,index=False)




                                                    
if __name__ == "__main__":
    demo = args.demo
    if demo == "radio":
        radio()
    elif demo == "line":
        line()
    elif demo == "map":
        show_map()
    elif demo == "select":
        select()
    elif demo == "map_and_select":
        map_and_select()
    elif demo == "map_cols":
        map_cols()
