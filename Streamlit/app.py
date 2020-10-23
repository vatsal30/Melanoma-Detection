import os
from os import listdir
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set(style="whitegrid")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


#plotly
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

from prediction import ensemble

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

temp = train_df.groupby(['benign_malignant','sex']).count()['image_name'].to_frame()


st.sidebar.title("Melenoma Detection")
nav = st.sidebar.radio("Go to",["Prediction","EDA","About Project","About US"])

if nav=="About US":
    st.title("About US")
    st.write("Team Members are :")
    st.markdown("""
    ## 1. Vatsal Vora
    > [LinkedIn] (https://www.linkedin.com/in/vatsal30/) [Github] (https://github.com/vatsal30/)

    ## 2. Mit Suthar
    > [LinkedIn] (https://www.linkedin.com/in/mit-suthar-7b5328161/) [Github] (https://github.com/mit-27)

    ## 3. Kashyap Shyani
    > [LinkedIn] (https://www.linkedin.com/in/kashyap-shyani-24219b179) [Github] (https://github.com/kashyapshyani/)

    # Project Link

    From the following link, you can find the project details:

    https://github.com/vatsal30/HackGujarat
    """,True)


if nav=="Prediction":
    st.title("Melanoma Prediction")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Choose an image for Prediction", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=384)
        # st.write(
        #             '''
        #             <img  class="Img">
        #                 src <br />
        #             </img>
        #             ''',
        #             unsafe_allow_html=True
        #         )
        st.write("")
        st.write("")


        if st.button("Predict"):
            prediction = ensemble(uploaded_file)

            if prediction<=0:
                st.write(
                    '''
                    <span  class="resultFalse">
                        Melanoma not Detected. <br />
                    </span>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.write(
                    '''
                    <span  class="resultTrue">
                        Melanoma Detected. <br />
                    </span>
                    ''',
                    unsafe_allow_html=True
                )



if nav=="EDA":
    st.sidebar.title("EDA")
    st.title("Exploratory Data Analysis (EDA)")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    st.write("Train size = ",train_df.shape)
    st.write("Test Size = ",test_df.shape)
    temp = train_df.groupby(['benign_malignant','sex']).count()['image_name'].to_frame()
    st.write('## **Train Set** ')
    temp=train_df.describe()
    st.write(temp)
    st.write('## **Test Set** ')
    temp = test_df.describe()
    st.write(temp)
    temp = train_df["image_name"].count()
    st.write('Total Images In Training DataSet: ',temp)
    temp = train_df.groupby(["benign_malignant"]).get_group("benign").count()["sex"]
    st.write('Number of Benign Sample in Training DataSet : ', temp)
    temp = train_df.groupby(["benign_malignant"]).get_group("malignant").count()["sex"]
    st.write(' Number of Malignant Sample in Training DataSet : ',temp)
    temp = test_df["image_name"].count()
    st.write('Total Images In Training DataSet: ',temp)
    temp = train_df['patient_id'].count()
    temp2 = train_df['patient_id'].nunique()
    st.write("The total Patient IDs are ",temp," from those unique ids are ",temp2)
    columns = list(train_df.columns)
    st.write("## **Column Names :** ")
    st.write(columns)

    if st.sidebar.checkbox("Distribution of The Target columns in training set"):
        train_df_new = train_df['target'].value_counts(normalize=True).reset_index()
        fig = px.bar(train_df_new,x='target',y='target')
        st.write("## **Distribution of The Target columns in training set**")
        fig.update_layout(
            # title="Distribution of The Target columns in training set",
            xaxis_title="Target",
            yaxis_title="Percentage",
            width=500,
            height=550,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Distribution of The Gender columns in training set"):
        train_df_new = train_df['sex'].value_counts(normalize=True).reset_index()
        fig = px.bar(x=train_df_new['sex'].index ,y=train_df_new['sex'].values)
        st.write("## **Distribution of The Sex columns in training set**")
        fig.update_layout(
            # title="Distribution of The Sex columns in training set",
            yaxis_title="Percentage",
            width=500,
            height=550,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Missing Values"):
        st.write("## **Missing Values**")
        dd = train_df[["sex","age_approx","anatom_site_general_challenge"]].isnull().sum()
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        fig = px.bar(dd,x='gh',y='values')
        # fig = px.bar(dd, x='gh' ,y='values')
        fig.update_layout(
            # title="Missing Values",
            xaxis_title="Count",
            yaxis_title="Columns",
            width=800,
            height=600,
        )
        # st.write(fig)
        st.plotly_chart(fig)
        st.write("### **Gender for Anatomy**")
        anatomy = train_df.copy()
        anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
        ztemp=anatomy.groupby(['sex','flag'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Missing', x=ztemp[ztemp['sex']=='male']['flag'], y=ztemp[ztemp['sex']=='male']['target']),
            go.Bar(name='Not Missing',  x=ztemp[ztemp['sex']=='female']['flag'], y=ztemp[ztemp['sex']=='female']['target'])
        ])
        # Change the bar mode
        fig.update_layout(barmode='group',
                          # title="Gender for Anatomy",
                          xaxis_title="Missing vs Not Missing",
                          yaxis_title="Count",
                          width=750,
                          height=600,)

        st.plotly_chart(fig)
        st.write("### **Distribution Plot**")
        anatomy = train_df.copy()
        anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
        colors_nude = ['#e0798c','#65365a','#da8886','#cfc4c4','#dfd7ca']
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
        sns.distplot(anatomy[anatomy['flag'] == 'missing']['age_approx'],
                     hist=False, rug=True, label='Missing',
                     color=colors_nude[2], kde_kws=dict(linewidth=4))
        sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age_approx'],
                     hist=False, rug=True, label='Not Missing',
                     color=colors_nude[3], kde_kws=dict(linewidth=4))
        st.pyplot()
    # nav2 = st.sidebar.radio("EDA Options",["Missing Values","Explore Target Column","Gender Wise Distribution","Gender vs Target Dsitribution"])



    # if st.sidebar.checkbox("Gender for Anatomy"):
    #     st.write("## **Gender for Anatomy**")
    #     anatomy = train_df.copy()
    #     anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
    #     ztemp=anatomy.groupby(['sex','flag'])['target'].count().to_frame().reset_index()

    #     fig = go.Figure(data=[
    #         go.Bar(name='Missing', x=ztemp[ztemp['sex']=='male']['flag'], y=ztemp[ztemp['sex']=='male']['target']),
    #         go.Bar(name='Not Missing',  x=ztemp[ztemp['sex']=='female']['flag'], y=ztemp[ztemp['sex']=='female']['target'])
    #     ])
    #     # Change the bar mode
    #     fig.update_layout(barmode='group',
    #                       # title="Gender for Anatomy",
    #                       xaxis_title="Missing vs Not Missing",
    #                       yaxis_title="Count",
    #                       width=750,
    #                       height=600,)

    #     st.plotly_chart(fig)

    if st.sidebar.checkbox("Gender vs Target Dsitribution"):
        st.write("## **Gender Vs Target Distribution**")
        z=train_df.groupby(['benign_malignant','sex'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Male', x=z[z['sex']=='male']['benign_malignant'], y=z[z['sex']=='male']['target']),
            go.Bar(name='Female', x=z[z['sex']=='female']['benign_malignant'], y=z[z['sex']=='female']['target'])
        ])
        # Change the bar mode

        fig.update_layout(barmode='group',
                          # title="Gender Vs Target Distribution",
                          xaxis_title="Benign:0 Vs Malignanat:1",
                          yaxis_title="Count",
                          width=750,
                          height=600,)

        st.plotly_chart(fig)

    if st.sidebar.checkbox(" Target columns Distribution in training set"):
        st.write("## **Target columns Distribution in training set**")
        dd = train_df['anatom_site_general_challenge'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.bar(dd, x='values' ,y='gh', orientation="h")
        fig.update_layout(
            title="Distribution of The Target columns in training set",
            yaxis_title="Percentage",
            width=900,
            height=500,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Location of Imaged Site w.r.t. Gender"):
        st.write("## **Location of Imaged Site w.r.t. Gender**")
        z1 = train_df.groupby(['sex','anatom_site_general_challenge'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Male', x=z1[z1['sex']=='male']['anatom_site_general_challenge'], y=z1[z1['sex']=='male']['target']),
            go.Bar(name='Female', x=z1[z1['sex']=='female']['anatom_site_general_challenge'], y=z1[z1['sex']=='female']['target'])
        ])
        # Change the bar mode
        fig.update_layout(barmode='group',
                          # title="Location of Imaged Site w.r.t. Gender",
                          xaxis_title="Location of Imaged site",
                          yaxis_title="Count of melanoma cases",
                          width=1000,
                          height=600)

        st.plotly_chart(fig)

    if st.sidebar.checkbox("Age Distribution"):
        st.write("## **Age Distribution**")
        dd = train_df['age_approx'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.histogram(dd, x='gh' ,y='values',nbins=30)
        st.write("### **Age Distribution of Patients**")
        fig.update_layout(
            # title="Age Distribution of Patients",
            xaxis_title="Age Distribution",
            yaxis_title="Count",
            width=1000,
            height=600,
        )

        st.plotly_chart(fig)

    if st.sidebar.checkbox("Diagnosis of Target Columns in Training DataSet"):
        st.write("## **Diagnosis of Target Columns in Training DataSet**")
        dd = train_df['diagnosis'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.bar(dd, x='values' ,y='gh', orientation="h")
        fig.update_layout(
            # title="Distribution of The Target columns in training set",
            xaxis_title="Percentage",
            yaxis_title="Diagnosis",
            width=900,
            height=500,
        )
        st.plotly_chart(fig)

    # if st.sidebar.checkbox("Distibution plot"):
    #     st.write("## **Distribution Plot**")
    #     anatomy = train_df.copy()
    #     anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
    #     colors_nude = ['#e0798c','#65365a','#da8886','#cfc4c4','#dfd7ca']
    #     # f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    #     sns.distplot(anatomy[anatomy['flag'] == 'missing']['age_approx'],
    #                  hist=False, rug=True, label='Missing',
    #                  color=colors_nude[2], kde_kws=dict(linewidth=4))
    #     sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age_approx'],
    #                  hist=False, rug=True, label='Not Missing',
    #                  color=colors_nude[3], kde_kws=dict(linewidth=4))
    #     st.pyplot()

    # if st.sidebar.checkbox("Explore Dataset Columns"):
    #     train_df = pd.read_csv("data/train.csv")
    #     test_df = pd.read_csv("data/test.csv")
    #     st.write("Train size = ",train_df.shape)
    #     st.write("Test Size = ",test_df.shape)
    #     temp = train_df.groupby(['benign_malignant','sex']).count()['image_name'].to_frame()
    #     st.title('Train Set ')
    #     temp=train_df.describe()
    #     st.write(temp)
    #     st.title('Test Set ')
    #     temp = test_df.describe()
    #     st.write(temp)
    #     temp = train_df["image_name"].count()
    #     st.write('Total Images In Training DataSet: ',temp)
    #     temp = train_df.groupby(["benign_malignant"]).get_group("benign").count()["sex"]
    #     st.write('Number of Benign Sample in Training DataSet : ', temp)
    #     temp = train_df.groupby(["benign_malignant"]).get_group("malignant").count()["sex"]
    #     st.write(' Number of Malignant Sample in Training DataSet : ',temp)
    #     temp = test_df["image_name"].count()
    #     st.write('Total Images In Training DataSet: ',temp)
    #     temp = train_df['patient_id'].count()
    #     temp2 = train_df['patient_id'].nunique()
    #     st.write("The total Patient IDs are ",temp," from those unique ids are ",temp2)
    #     columns = list(train_df.columns)
    #     st.title("Column Names : ")
    #     st.write(columns)

if nav=="About Project":
    st.title("All about Melenoma")
    st.markdown("""
    > Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.
https://www.kaggle.com/c/siim-isic-melanoma-classification

> Melanoma is a skin cancer that arises from a skin cell called a melanocyte, which makes a the pigment (melanin) that gives your skin its color. Melanoma can appear in different ways, most commonly as a new spot on the skin or as an already existing mole that changes in color, size, or shape. While considered the most dangerous type of skin cancer because of its ability to rapidly spread throughout the body, melanoma is generally very treatable if found early.
https://www.verywellhealth.com/what-is-melanoma-514215

<img src='https://impactmelanoma.org/wp-content/uploads/2018/11/Standard-Infographic_0.jpg' style="width:700px;height:400px;">


> The[ Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/page/about_siim) is the leading healthcare professional organization for those interested in the current and future use of informatics in medical imaging. The society's mission is to advance medical imaging informatics across the enterprise through education, research, and innovation in a multi-disciplinary community. The [International Skin Imaging Collaboration or
ISIC](https://siim.org/page/about_siim) Melanoma Project is an academia and industry partnership designed to facilitate the application of digital skin imaging to help reduce melanoma mortality

> The overarching goal of the ISIC Melanoma Project is to support efforts to reduce melanoma-related deaths and unnecessary biopsies by improving the accuracy and efficiency of melanoma early detection since when recognized and treated in its earliest stages, melanoma is readily curable


## **Symptoms**


> Melanomas can develop anywhere on your body. They most often develop in areas that have had exposure to the sun, such as your back, legs, arms and face.

> Melanomas can also occur in areas that don't receive much sun exposure, such as the soles of your feet, palms of your hands and fingernail beds. These hidden melanomas are more common in people with darker skin.

> The first melanoma signs and symptoms often are:

> A change in an existing mole
The development of a new pigmented or unusual-looking growth on your skin
Melanoma doesn't always begin as a mole. It can also occur on otherwise normal-appearing skin.


## **Causes**
<img src='https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/11/15/17/40/ds00190_-ds00439_-ds00924_-ds00925_im02400_c7_skincancerthu_jpg.jpg' style="width:500px;height:300px;">

> Melanoma occurs when something goes wrong in the melanin-producing cells (melanocytes) that give color to your skin.
Normally, skin cells develop in a controlled and orderly way — healthy new cells push older cells toward your skin's surface, where they die and eventually fall off. But when some cells develop DNA damage, new cells may begin to grow out of control and can eventually form a mass of cancerous cells.

> Just what damages DNA in skin cells and how this leads to melanoma isn't clear. It's likely that a combination of factors, including environmental and genetic factors, causes melanoma. Still, doctors believe exposure to ultraviolet (UV) radiation from the sun and from tanning lamps and beds is the leading cause of melanoma.

> UV light doesn't cause all melanomas, especially those that occur in places on your body that don't receive exposure to sunlight. This indicates that other factors may contribute to your risk of melanoma.


## **Objective**

> The objective of this competition is to identify melanoma in images of skin lesions. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.In particular, we need to use images within the same patient and determine which are likely to represent a melanoma. In other words, we need to create a model which should predict the probability whether the lesion in the image is malignantor benign.Value 0 denotes benign, and 1 indicates malignant.

## **Dataset**

> The dataset which we are going to use are from following sources:
1. Kaggle SIIM Melanoma Classification Challange
    https://www.kaggle.com/c/siim-isic-melanoma-classification
2. ISIC 2019 Melanoma Classification
    https://challenge2019.isic-archive.com/

> The dataset consists of images in :
* DIOCOM format
* JPEG format in JPEG directory
* TFRecord format in tfrecords directory

> Additionally, there is a metadata comprising of train, test and submission file in CSV format.

## **Understanding the Evaluation Metric**

> For this particluar problem, our submissions will be evaluated using **area under the ROC curve**. An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

![](https://imgur.com/yNeAG4M.png)

> An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

![](https://imgur.com/N3UOcBF.png)

source: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc """,True)

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
