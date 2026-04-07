import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# page configuration
st.set_page_config('Linear Reg, model',layout ='wide')

# loading pre-trained model
with open ('linear_model.pkl','rb')as f:
    lm =pickle.load(f)

# loading feature importance from an excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# loading ther feature importance dataframe
final_fi =load_feature_importance('feature_importance.xlsx')

# setting up streamlit sidebar
image_sidebar =Image.open('pk2.png')
st.sidebar.image(image_sidebar, width='stretch')
st.sidebar.header('Vehicle Features')

# split layout into two columns
# left_col, right_col =st.sidebar.columns(2)
# gathering the input 
# feature selection on sidebar
def get_user_input():
    left_col, right_col =st.sidebar.columns(2)
    with left_col:
        wheel_base =st.sidebar.number_input('wheel_base (No.)',min_value=0.0,max_value=120.9,step=0.5,value=88.6)
        length =st.sidebar.number_input('length (No.)',min_value=0.0,max_value=208.1,step=0.5,value=166.8)
        width =st.sidebar.number_input('width (No.)',min_value=0.0,max_value=72.0,step=0.5,value=65.8)
        height =st.sidebar.number_input('height(No.)',min_value=0.0,max_value=59.8,step=0.5,value=53.6)
        curb_weight =st.sidebar.number_input('curb_weight (No.)',min_value=0.0, max_value=4066.0,step=0.5,value=2555.6)
        number_of_cylinders =st.sidebar.selectbox('No_of_cylinders',[2,3,4,5,6,8,12])
        engine_size =st.sidebar.number_input('engine_size (No.)',min_value=0.0,max_value=326.0,step=0.5,value=126.8)
        bore =st.sidebar.number_input('bore(No.)',min_value=0.0,max_value=3.94,step=0.5,value=3.32)
        horsepower =st.sidebar.number_input('horse_power (No.)',min_value=0.0,max_value=262.0,step=0.5,value=103.3)
        peak_rpm =st.sidebar.number_input('rmp(No.)',min_value=0.0,max_value=6600.0,step=0.5,value=5121.3)
    
    with right_col:
        make =st.sidebar.selectbox('make',['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda','isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury','mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault','saab', 'subaru', 'toyota', 'volkswagen', 'volvo'])
        fuel_type =st.sidebar.selectbox('Fuel_type',['diesel','gas'])
        aspiration =st.sidebar.selectbox('Aspiration',['std','turbo'])
        number_of_doors =st.sidebar.selectbox('No_of_Doors',['four','two'])
        body_style =st.sidebar.selectbox('Body_style',['convertible','hardtop', 'hatchback','sedan','wagon']) 
        drive_wheels =st.sidebar.selectbox('drive_wheels',['4wd', 'fwd', 'rwd'])
        engine_location =st.sidebar.selectbox('Engin_location',['front','rear'])
        engine_type =st.sidebar.selectbox('engine_type',['dohc', 'l', 'ohc' 'ohcf' ,'ohcv' ,'rotor'])    
        fuel_system =st.sidebar.selectbox('fuel_system',['1bbl','2bbl','4bbl' ,'idi','mfi','mpfi','spdi','spfi']) 
    
    user_data={
        'wheel_base':wheel_base, 
        'length':length,
        'width':width, 
        'height':height, 
        'curb_weight':curb_weight,
        'number_of_cylinders':number_of_cylinders, 'engine_size':engine_size,
        'bore':bore, 
        'horsepower':horsepower,
        'peak_rpm':peak_rpm,
        f'make_{make}':1, 
        f'fuel_type_{fuel_type}':1, 
        f'aspiration_{aspiration}':1,
        f'number_of_doors_{number_of_doors}':1, 
        f'body_style_{body_style}':1, 
        f'drive_wheels_{drive_wheels}':1, 
        f'engine_location_{engine_location}':1, 
        f'engine_type_{engine_type}':1, 
        f'fuel_system_{fuel_system}':1
    }
    return user_data

# user_data =get_user_input()

# adding main header image
impage_banner =Image.open('pk.png')
st.image(impage_banner,width='stretch')

# centerizing title
st.markdown("<h1 style =text-align:'center;'>vehicle prediction app</>", unsafe_allow_html=True)

# split layout into two columns
left_col, right_col =st.columns(2)

# "left columns: contain feature importance interactive bar chart"
with left_col:
    st.header("Feature Importance")

    # # sorting freature importance dataframe by 'feature importance score 
    final_fi_sorted =final_fi.sort_values(by='feature importance score:', ascending=True)
    
    # variable	feature importance score:
    # create intaractive bar chart with plotly
    fig =px.bar(
    final_fi_sorted,
    x='feature importance score:',
    y='variable',
    orientation ='h',
    title='Feature Importance',
    labels={'feature importance score:':'importance','variable':'feature'},
    text ='feature importance score:',
    color_discrete_sequence=['#48a3b4']
    )
    
    fig.update_layout(
    xaxis_title ='Feature importance score:',
    yaxis_title ='variable',
    template ='plotly_white',
    height =500
    )
    
    # setting the streamlit graph function
    st.plotly_chart(fig,use_column_width=True)

# right column: prediction interface
with right_col:
    st.header("Predict Vehicle Price")

    # getting user inputs from sidebar
    user_data =get_user_input()

    # tranform the input into the required format
    def prepare_input(data, feature_list):
        input_data ={feature: data.get(feature,0) for feature in feature_list}
        return np.array([list(input_data.values())])
    
    # feature list (same order as use during model train)
    features =['wheel_base', 'length', 'width', 'height', 'curb_weight','number_of_cylinders', 'engine_size', 'bore', 'horsepower', 'peak_rpm','make_alfa-romero', 'make_audi', 'make_bmw', 'make_chevrolet', 'make_dodge','make_honda', 'make_isuzu', 'make_jaguar', 'make_mazda', 'make_mercedes-benz', 'make_mercury', 'make_mitsubishi', 'make_nissan','make_peugot', 'make_plymouth', 'make_porsche', 'make_renault', 'make_saab', 'make_subaru', 'make_toyota', 'make_volkswagen', 'make_volvo', 'fuel_type_diesel', 'fuel_type_gas', 'aspiration_std', 'aspiration_turbo', 'number_of_doors_four', 'number_of_doors_two','body_style_convertible', 'body_style_hardtop', 'body_style_hatchback','body_style_sedan', 'body_style_wagon', 'drive_wheels_4wd','drive_wheels_fwd', 'drive_wheels_rwd', 'engine_location_front','engine_location_rear', 'engine_type_dohc', 'engine_type_l', 'engine_type_ohc', 'engine_type_ohcf', 'engine_type_ohcv', 'engine_type_rotor', 'fuel_system_1bbl', 'fuel_system_2bbl', 'fuel_system_4bbl', 'fuel_system_idi', 'fuel_system_mfi', 'fuel_system_mpfi', 'fuel_system_spdi', 'fuel_system_spfi']
    
    
    # predict button
    if st.button('Predict'):
         input_array =prepare_input(user_data,features)
         prediction =lm.predict(input_array)
         st.subheader('Predicted Price:')
         st.write(f"₦{prediction[0]:,.2f}")

