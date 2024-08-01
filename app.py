import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import os
import warnings
import streamlit.web.cli as stcli
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

#page configuration
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Amriti Gupta", page_icon=":bar_chart:", layout="wide")
st.title("Content Insight Generation Based on Customer Journeys")
st.markdown("<style> div.block-container(padding-top:0.5rem;)</style>", unsafe_allow_html=True)

paths = ['Collections -> Product View', 'Product View -> Cart Add', 'Cart Add -> Checkout', 'Checkout -> Purchase']
st.sidebar.header("Attribute Level")
path = 'Collections -> Product View'
path = st.sidebar.selectbox("Choose your Path :", paths)

# Function to get data based on region
data_region_1 = pd.read_csv("statistics_checkout->purchase/attribute_level/final_ranks.csv")
data_region_2 = pd.read_csv("statistics_listadd->checkout/attribute_level/final_ranks.csv")
data_region_3 = pd.read_csv("statistics_collectionsclicked/attribute_level/final_ranks.csv")
data_region_4 = pd.read_csv("statistics_prodview->listadd/attribute_level/final_ranks.csv")

imagedf1 = pd.read_csv("streamlitassets_collectionscliceked.csv")
imagedf1 = imagedf1[~imagedf1['assetID'].str.contains("disruptors", case=False, na=False)]
imagedf2 = pd.read_csv("skuassets_with_features.csv")

def get_data(region):
    if region == 'Collections -> Product View':
        values = ['tags', 'foregroundColors', 'visualAttentionSpread', 'visualContentDensity', 'overallTone']
        avgctr = 0.13169961706
        return data_region_3, values, avgctr, imagedf1
    elif region == 'Product View -> Cart Add':
        values = ['tags', 'foregroundColors', 'visualAttentionSpread', 'visualContentDensity']
        avgctr = 0.11253844126
        return data_region_4, values, avgctr, imagedf2
    elif region == 'Cart Add -> Checkout':
        values = ['tags', 'foregroundColors', 'visualAttentionSpread', 'visualContentDensity']
        avgctr = 0.40914734473
        return data_region_2, values, avgctr, imagedf2
    elif region == 'Checkout -> Purchase':
        values = ['tags', 'foregroundColors', 'visualAttentionSpread', 'visualContentDensity']
        avgctr = 0.38429280213
        return data_region_1, values, avgctr, imagedf2

def get_data2(region, value):
    if region == 'Collections -> Product View':
        stringhelp = 'statistics_collectionsclicked'
    elif region == 'Product View -> Cart Add':
        stringhelp = 'statistics_prodview->listadd'
    elif region == 'Cart Add -> Checkout':
        stringhelp = 'statistics_listadd->checkout'
    elif region == 'Checkout -> Purchase':
        stringhelp = 'statistics_checkout->purchase'

    str1 = f"{stringhelp}/positive_{value}.csv"
    str2 = f"{stringhelp}/negative_{value}.csv"

    df1 = pd.read_csv(str1, index_col=False)
    df2 = pd.read_csv(str2, index_col=False)

    if region == 'Collections -> Product View':
        if value == 'tags':
            def has_suffix(value):
                for suffix in suffixes_to_remove:
                    if value.endswith(suffix):
                        return True
                return False
            suffixes_to_remove = [
                "underpants", "underwear", "active_wear", "holding", "lingerie", "posing", 'crop_top', 'bodysuit', 'athlete', 'apparel', 'wearing', 'camisole', 'camera', 'shirt', 'face', 'wear', 'style'
            ]
            suffix_renames = {
                "dress": "shapewear",
                "sports_bra": "bra",
            }
            def renaming(value):
                for suffix, new_name in suffix_renames.items():
                    if value.endswith(suffix):
                        return value.replace(suffix, new_name)
                return value
            
            condition = df1['Column'].apply(has_suffix)
            df1 = df1[~condition]
            df1['Column'] = df1['Column'].apply(renaming)
            condition = df2['Column'].apply(has_suffix)
            df2 = df2[~condition]
            df2['Column'] = df2['Column'].apply(renaming)
        elif value == 'foregroundColors':
            def has_suffix(value):
                for suffix in suffixes_to_remove:
                    if value.endswith(suffix):
                        return True
                return False
            suffixes_to_remove = []
            condition = df1['Column'].apply(has_suffix)
            df1 = df1[~condition]
            condition = df2['Column'].apply(has_suffix)
            df2 = df2[~condition]
    else:
        if value == 'tags':
            def has_suffix(value):
                for suffix in suffixes_to_remove:
                    if value.endswith(suffix):
                        return True
                return False
            suffix_renames = {
                "dress": "shapewear",
                "sports_bra": "bra",
            }
            def renaming(value):
                for suffix, new_name in suffix_renames.items():
                    if value.endswith(suffix):
                        return value.replace(suffix, new_name)
                return value
            suffixes_to_remove = [
                "underwear", "posing", "style", "bodysuit", "active_wear", "camisole", "underpants", "holding", "face", "back", 'skin', 'wear', 'wearing', 'hip', 'lingerie'
            ]
            condition = df1['Column'].apply(has_suffix)
            df1 = df1[~condition]
            df1['Column'] = df1['Column'].apply(renaming)
            condition = df2['Column'].apply(has_suffix)
            df2 = df2[~condition]
            df2['Column'] = df2['Column'].apply(renaming)
        elif value == 'foregroundColors':
            def has_suffix(value):
                for suffix in suffixes_to_remove:
                    if value.endswith(suffix):
                        return True
                return False
            suffixes_to_remove = ["dark_pink", "dark_blue", "orange"]
            condition = df1['Column'].apply(has_suffix)
            df1 = df1[~condition]
            condition = df2['Column'].apply(has_suffix)
            df2 = df2[~condition]

    if 'Unnamed: 0' in df1.columns:
        df1 = df1.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in df2.columns:
        df2 = df2.drop(columns=['Unnamed: 0'])

    return df1, df2

values = ['tags', 'foregroundColors', 'visualAttentionSpread', 'visualContentDensity']
avgctr = 0
valueleveldata, values, avgctr, imagedf = get_data(path)
valueleveldata = valueleveldata.sort_values(by='Final Rank')
valueleveldata = valueleveldata.drop(columns=['Unnamed: 0', 'Average Rank'])
valueleveldata.rename(columns={'Final Rank': 'Rank of Importance'}, inplace=True)
valueleveldata['Feature'] = valueleveldata['Feature'].str.replace(r'^contentFeaturization_', '', regex=True)

st.sidebar.write("Average Probability:", avgctr)
st.sidebar.dataframe(valueleveldata)

st.sidebar.header("Value Level")
value_selected = st.sidebar.selectbox("Choose the value:", values)

img = Image.open('streamlit_conversiongraph.jpeg')
st.image(img)

data1, data2 = get_data2(path, value_selected)
data1.columns = data1.columns.str.strip()
data2.columns = data2.columns.str.strip()
if 'Unnamed: 0' in data1.columns:
    data1 = data1.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in data2.columns:
    data2 = data2.drop(columns=['Unnamed: 0'])

st.write("Positive Attributes:")
st.dataframe(data1)

# Create a bar chart using Altair
# data1_sorted = data1.sort_values(by='Importance', ascending=False)

# # Create a bar chart using Plotly
# fig = px.bar(data1_sorted, x='Column', y='Importance',
#              labels={'Column': 'Item', 'Importance': 'Importance'})

# fig.update_traces(marker_color='#f05baf')

data1_sorted = data1.sort_values(by='Importance', ascending=False)

# Create a bar chart using Plotly
fig = px.bar(data1_sorted, x='Column', y='Importance',
             labels={'Column': 'Item', 'Importance': 'Importance'})

# Add a line chart on the same figure
fig.add_trace(go.Scatter(x=data1_sorted['Column'], y=data1_sorted['Average CTR 1'], 
                         mode='lines+markers', 
                         name='Value',
                         yaxis='y2'))

# Update the layout to include a secondary y-axis


fig.update_layout(
    yaxis2=dict(
        overlaying='y',
        side='right',
        title='Value'
    ),
    xaxis=dict(
        title='Item'
    ),
    yaxis=dict(
        title='Importance'
    )
)

# Update the color of the bar chart
fig.update_traces(marker_color='#f05baf', selector=dict(type='bar'))


fig.update_layout(width=900)


# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=False)

st.write("Negative Attributes:")
st.dataframe(data2)

data2_sorted = data2.sort_values(by='Importance', ascending=False)

fig = px.bar(data2_sorted, x='Column', y='Importance',
             labels={'Column': 'Item', 'Importance': 'Importance'})

# Add a line chart on the same figure
fig.add_trace(go.Scatter(x=data2_sorted['Column'], y=data2_sorted['Average CTR 1'], 
                         mode='lines+markers', 
                         name='Value',
                         yaxis='y2'))

# Update the layout to include a secondary y-axis


fig.update_layout(
    yaxis2=dict(
        overlaying='y',
        side='right',
        title='Value'
    ),
    xaxis=dict(
        title='Item'
    ),
    yaxis=dict(
        title='Importance'
    )
)

# Update the color of the bar chart
fig.update_traces(marker_color='#f05baf', selector=dict(type='bar'))


fig.update_layout(width=900)


# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=False)


def process_image_df(imagedf):
    def replace_nan_and_eval(column):
        return column.apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))

    imagedf['contentFeaturization.tags'] = replace_nan_and_eval(imagedf['contentFeaturization.tags'])
    imagedf['contentFeaturization.foregroundColors'] = replace_nan_and_eval(imagedf['contentFeaturization.foregroundColors'])
    imagedf['contentFeaturization.backgroundColors'] = replace_nan_and_eval(imagedf['contentFeaturization.backgroundColors'])

    mdf = imagedf.copy()
    
    tag_replacement_dict = {
        'blond': 'blonde',
        'belt': 'strap',
        'pajama': 'pajamas',
        'swimsuit': 'bodysuit',
        'swimwear': 'bodysuit',
        'workout_clothes': 'active_wear',
        'exercise': 'active_wear',
        'workout': 'active_wear',
        'swimmer': 'bodysuit',
        'swimming': 'bodysuit',
        'swim': 'bodysuit',
        'water': 'bodysuit',
        'suit': 'bodysuit',
        'print': 'leopard_print',
        'drink': 'drinking',
        'toast': 'drinking',
        'glass': 'drinking',
        'glasses': 'drinking',
        'wine': 'drinking',
        'wine_glass': 'drinking',
        'alcohol': 'drinking',
        'champagne': 'drinking',
        'celebration': 'drinking',
        'cheers': 'drinking',
        'purse': 'handbag',
        'safe': 'security',
        'lock': 'security',
        'shoulder': 'shoulders',
        'arm': 'shoulders',
        'cocktail_dress': 'midi_dress',
        'box': 'boxer',
        'boxing': 'boxer',
        'boxers': 'boxer',
        'shoe': 'shoes',
        'leopard': 'leopard_print',
        'bandage': 'corset',
        'injury': 'back',
        'bedding': 'bed',
        'pillow': 'bed',
        'bike': 'cycling',
        'drawing': 'animation',
        'cartoon': 'animation',
        'dancer': 'athlete',
        'denim': 'jeans',
        'pants': 'underpants',
        'undergarment': 'underpants',
        'silk': 'satin',
        'tank': 'tank_top',
        'abdomen': 'waist'
    }

    tags_to_remove = {'wear', 'wearing', 'back'}
    # Function to replace specified tags in a list of tags
    def replace_tags(tags, tag_replacement_dict):
        return [tag_replacement_dict.get(tag, tag) for tag in tags]
    def remove_tags(tags, tags_to_remove):
        return [tag for tag in tags if tag not in tags_to_remove]

    mdf['contentFeaturization.tags'] = mdf['contentFeaturization.tags'].apply(lambda tags: replace_tags(tags, tag_replacement_dict))
    mdf['contentFeaturization.tags'] = mdf['contentFeaturization.tags'].apply(set)
    mdf['contentFeaturization.tags'] = mdf['contentFeaturization.tags'].apply(list)
    mdf['contentFeaturization.tags'] = mdf['contentFeaturization.tags'].apply(lambda tags: remove_tags(tags, tags_to_remove))

    imagedf = mdf.copy()

    imagedf = pd.get_dummies(imagedf, columns=['contentFeaturization.overallTone'], prefix='overallTone')
    imagedf = pd.get_dummies(imagedf, columns=['contentFeaturization.visualContentDensity'], prefix='visualContentDensity')
    imagedf = pd.get_dummies(imagedf, columns=['contentFeaturization.visualAttentionSpread'], prefix='visualAttentionSpread')

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(imagedf['contentFeaturization.tags'])
    tags_encoded_df = pd.DataFrame(tags_encoded, columns=[f'tags_{tag}' for tag in mlb.classes_])
    backgroundColors_encoded = mlb.fit_transform(imagedf['contentFeaturization.backgroundColors'])
    backgroundColors_encoded_df = pd.DataFrame(backgroundColors_encoded, columns=[f'backgroundColors_{tag}' for tag in mlb.classes_])
    foregroundColors_encoded = mlb.fit_transform(imagedf['contentFeaturization.foregroundColors'])
    foregroundColors_encoded_df = pd.DataFrame(foregroundColors_encoded, columns=[f'foregroundColors_{tag}' for tag in mlb.classes_])
    imagedf = pd.concat([imagedf, tags_encoded_df, foregroundColors_encoded_df, backgroundColors_encoded_df], axis=1)

    return imagedf

imagedf1 = process_image_df(imagedf1)
imagedf2 = process_image_df(imagedf2)

st.sidebar.header("Images")
attributes_list = ['tags', 'overallTone', 'visualContentDensity', 'visualAttentionSpread', 'foregroundColors']
attribute = st.sidebar.selectbox("Choose Value :", attributes_list)

def get_attributelist(region, attribute):
    if region == 'Collections -> Product View':
        imagedf = imagedf1.copy()
        cols = [col for col in imagedf.columns if col.startswith(attribute + '_')]
        cols = ['tags_bra' if col == 'tags_sports_bra' else col for col in cols]
        cols = ['tags_shapewear' if col == 'tags_dress' else col for col in cols]
    else:
        imagedf = imagedf2.copy()
        cols = [col for col in imagedf.columns if col.startswith(attribute + '_')]
        cols = ['tags_bra' if col == 'tags_sports_bra' else col for col in cols]
        cols = ['tags_shapewear' if col == 'tags_dress' else col for col in cols]
    return cols

attributes_list = get_attributelist(path, attribute)
choose_attribute = st.sidebar.selectbox("Choose Attribute :", attributes_list)

ccdf = pd.read_csv("cc_30july.csv")

if path == 'Collections -> Product View':
    if choose_attribute == 'tags_bra':
        choose_attribute = 'tags_sports_bra'
    if choose_attribute == 'tags_shapewear':
        choose_attribute = 'tags_dress'
    filtered_df1 = ccdf[ccdf[choose_attribute] == 1]
    for url in filtered_df1['assetID']:
        st.sidebar.image(url, width=200)
else:
    
    if choose_attribute == 'tags_bra':
        choose_attribute = 'tags_sports_bra'
    if choose_attribute == 'tags_shapewear':
        choose_attribute = 'tags_dress'
    filtered_df = imagedf2[imagedf2[choose_attribute] == 1]
    for url in filtered_df['assetID']:
        st.sidebar.image(url, width=200)


