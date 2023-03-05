from email import message
from tabnanny import verbose
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import plotly.express as px
import io 
import squarify

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/trainForLearnInformation.csv")
centerInformation = pd.read_csv("data/centerInformation.csv")
mealInformation = pd.read_csv("data/mealInformation.csv")
df = df.merge(centerInformation, on='center_id', how='left')    
df = df.merge(mealInformation, on='meal_id', how='left')


st. set_page_config(layout="wide")
with st.sidebar:
    choose = option_menu("App Gallery", [ "Descriptive Analytics", "Data Visulation", "Predictive Analytics","About", "Contact"],
                         icons=[ 'camera fill', 'kanban', 'book','house','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "skyblue", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

logo = Image.open(r'images/icon1.png')
profile = Image.open(r'images/icon1.png')

if choose == "Descriptive Analytics":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #d1fd00;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Descriptive Analytics of Fast Food</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo
        st.image(logo,  width=150)

    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Learn Python for Data Science</p>', unsafe_allow_html=True)

    st.subheader('Import Data into Python')
    st.markdown('To start a data science project in Python, you will need to first import your data into a Pandas data frame. Often times we have our raw data stored in a local folder in csv format. Therefore let\'s learn how to use Pandas\' read_csv method to read our sample data into Python.')

    #Display the first code snippet
    code = '''import pandas as pd #import the pandas library\ndf=pd.read_csv(r'G:\1College\7th Sem\F1 - Data Analytics\Project\App\data\trainForLearnInformation.csv') #read the csv file into pandas\ndf.head() #display the first 5 rows of the data'''
    st.code(code, language='python')

    #Allow users to check the results of the first code snippet by clicking the 'Check Results' button
    dfnow=pd.read_csv(r'data/trainForLearnInformation.csv')
    df_head=dfnow.head()
    if st.button('Check Results', key='1'):
        st.write(df_head)
    else:
        st.write('---')

    #Display the second code snippet
    code = '''#display the merged data'''
    st.code(code, language='python')

    #Allow users to check the results of the second code snippet by clicking the 'Check Results' button
    dfnow=pd.read_csv(r'data/trainForLearnInformation.csv')
    df_tail=df.tail()
    if st.button('Check Results', key='2'):
        st.write(df_tail)
    else:
        st.write('---')     

    #Display the third code snippet
    st.write('   ')
    st.markdown('After we import the data into Python, we can use the following code to check the information about the data frame, such as number of rows and columns, data types for each column, etc.')
    code = '''dfnow.info()''' 
    st.code(code, language='python')

    #Allow users to check the results of the third code snippet by clicking the 'Check Results' button
    import io 
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    if st.button('Check Results', key='3'):
        st.text(s)
    else:
        st.write('---')
    
    col8, col9 = st.columns([0.4,0.4])

    with col8:
        st.subheader("Mean of checkout price wrt category")
        base_sum = df.groupby(['category']).mean()
        center = base_sum.index
        sum_base = base_sum['checkout_price']
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(
            x=sum_base, y=center, color="pink", ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Books Read")
        st.pyplot(fig) 

    with col9:
        st.subheader("Mean of num_orders wrt cuisine")
        base_sum = df.groupby(['cuisine']).mean()
        center = base_sum.index
        sum_base = base_sum['num_orders']
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(
            x=center, y=sum_base, color="blue", ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Books Read")
        st.pyplot(fig) 
    
    col3,col4 =  st.columns( [0.4, 0.4])
    with col3:
        st.subheader("Minimum of checkout price wrt center")

        # Graph : 1 - Mean of checkout price wrt center
        base_sum = df.groupby(['center_type']).min()
        center = base_sum.index
        sum_base = base_sum['checkout_price']
        fig,ax = plt.subplots(figsize=(10, 10))
        sns.barplot(
            x=center, y=sum_base, color="goldenrod", ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Books Read")
        st.pyplot(fig)
    
    with col4:
        st.subheader("Maximum of no of orders wrt center")
        # Graph : 2 - Mean of no of orders wrt center
        base_sum = df.groupby(['center_type']).max()
        center = base_sum.index
        sum_base = base_sum['num_orders']
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(
            x=center, y=sum_base, color="blue", ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Books Read")
        st.pyplot(fig)



elif choose == "Data Visulation":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #d1fd00;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Data Visulation of Fast Food</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo
        st.image(logo,  width=150)

    col3,dum =  st.columns([0.9,0.1])
    with col3:
        st.dataframe(df.head())
    col4, col5 = st.columns([0.2,0.2])
    with col4:        
        st.subheader("Correlation Plot of the Dataset")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        st.write(fig)
    with col5:
        st.subheader("Scatter Plot of num_orders and checkout_price")
        fig,ax = plt.subplots(figsize=(10, 10))
        plt.scatter(df['num_orders'],df['checkout_price'])
        st.pyplot(plt)
    
    col6, col7 = st.columns([0.4,0.4])

    with col6:
        st.subheader("Maximum of price wrt Category")
        base_sum = df.groupby(['category']).min()
        center = base_sum.index
        sum_base = base_sum['checkout_price']
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(
            x=sum_base, y=center, color="blue", ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Year")
        ax.set_ylabel("Books Read")
        st.pyplot(fig) 


    with col7:
        st.subheader("Percentage of Cuisines Available")
        labels = df['cuisine'].unique()
        sizes = df['cuisine'].value_counts()
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)   

    col6, col7 = st.columns([0.4,0.4])

    with col6:
        st.subheader("TreeMap of Center types")        
        base_sum = df.groupby(['center_type']).mean()
        volume = base_sum['checkout_price']
        labels = base_sum.index
        # plt.rc('font', size=14)
        squarify.plot(sizes=volume, label=labels, alpha=0.7)
        # plt.axis(labels)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


elif choose == "Predictive Analytics":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Predictive Analytics </p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )

    st.subheader('CatBoostRegressor')
    st.markdown('CatBoost builds upon the theory of decision trees and gradient boosting. The main idea of boosting is to sequentially combine many weak models (a model performing slightly better than random chance) and thus through greedy search create a strong competitive predictive model.')

    st.markdown('Here we will predict the number of orders for a given week.')

    st.write("Prediction Going ON!!!!!!!!!!!!!! Please Wait")

    df_train = pd.read_csv("data/trainForLearnInformation.csv")
    df_test = pd.read_csv("data/testInformation.csv")
    centerInformation2 = pd.read_csv("data/centerInformation.csv")
    mealInformation2 = pd.read_csv("data/mealInformation.csv")

    # Validate to return dataframe
    learnFromTrainData = pd.concat([df_train, df_test], axis=0)
    learnFromTrainData = learnFromTrainData.merge(centerInformation2, on='center_id', how='left')
    learnFromTrainData = learnFromTrainData.merge(mealInformation2, on='meal_id', how='left')

##Deriving New Features

# Special Price
    learnFromTrainData['special price'] = learnFromTrainData['base_price'] - learnFromTrainData['checkout_price']

# Special Price Percent
    learnFromTrainData['special price percent'] = ((learnFromTrainData['base_price'] - learnFromTrainData['checkout_price']) / learnFromTrainData['base_price']) * 100

# Special Price T/F
    learnFromTrainData['special price t/f'] = [1 if x > 0 else 0 for x in (learnFromTrainData['base_price'] - learnFromTrainData['checkout_price'])]
    learnFromTrainData = learnFromTrainData.sort_values(['center_id', 'meal_id', 'week']).reset_index()

# Weekly Price Comparison
    learnFromTrainData['weekly_price_comparison'] = learnFromTrainData['checkout_price'] - learnFromTrainData['checkout_price'].shift(1)
    learnFromTrainData['weekly_price_comparison'][learnFromTrainData['week'] == 1] = 0
    learnFromTrainData = learnFromTrainData.sort_values(by='index').reset_index().drop(['level_0', 'index'], axis=1)

# Weekly Price Comparison T/F
    learnFromTrainData['weekly_price_comparison t/f'] = [1 if x > 0 else 0 for x in learnFromTrainData['weekly_price_comparison']]
    learnFromTrainData.head()
    learnFromTrainData.isnull().sum()
    trainStart = datetime.datetime.now()

    trainData = learnFromTrainData[learnFromTrainData['week'].isin(range(1, 146))]

# Copying to DataFrame
    dataFrameFromTrainData = learnFromTrainData.copy()
# Return top n (5 by default) rows of a data frame
    dataFrameFromTrainData.head()

# Encoding all the categorical features
    dataFrameFromTrainData['center_id'] = dataFrameFromTrainData['center_id'].astype('object')
    dataFrameFromTrainData['meal_id'] = dataFrameFromTrainData['meal_id'].astype('object')
#dataFrameFromTrainData['region_code'] = dataFrameFromTrainData['region_code'].astype('object')

    dataTypeOne = dataFrameFromTrainData[['center_id', 'meal_id', 'center_type', 'category', 'cuisine']]
    dataTypeTwo = dataFrameFromTrainData.drop(['center_id', 'meal_id', 'center_type', 'category', 'cuisine'], axis=1)

# Drop one dimension from the representation to avoid dependency among the variables and convert
    dummyVar = pd.get_dummies(dataTypeOne, drop_first=True)

# Merge DataFrames by indexes
    dataFrameFromTrainData = pd.concat([dataTypeTwo, dummyVar], axis=1)
    dataFrameFromTrainData.head()

# Returns the absolute value of data frame num_orders
    abs(trainData.corr()['num_orders']).sort_values(ascending=False)

    standardScaler = StandardScaler()
    dataTypeOne = dataFrameFromTrainData.drop(['checkout_price', 'base_price', 'special price', 'special price percent', 'weekly_price_comparison'], axis=1)
    dataTypeTwo = dataFrameFromTrainData[['checkout_price', 'base_price', 'special price', 'special price percent', 'weekly_price_comparison']]

# Data standardization
    standardizationData = pd.DataFrame(standardScaler.fit_transform(dataTypeTwo), columns=dataTypeTwo.columns)
    concatData = pd.concat([standardizationData, dataTypeOne], axis=1)

# Copy modified dataframe to concatDataFrame variable
    concatDataFrame = concatData.copy()

# Categorizing weeks to quarters
    concatDataFrame['Quarter'] = (concatData['week'] / 13).astype('int64')
    concatDataFrame['Quarter'] = concatDataFrame['Quarter'].map({0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4', 4: 'Q1', 5: 'Q2', 6: 'Q3', 7: 'Q4', 8: 'Q1',
                                                             9: 'Q2', 10: 'Q3', 11: 'Q4'})
# Returns object containing counts of unique values in quarter
    concatDataFrame['Quarter'].value_counts()

# Categorizing weeks to years
    concatDataFrame['Year'] = (concatData['week'] / 52).astype('int64')
    concatDataFrame['Year'] = concatDataFrame['Year'].map({0: 'Y1', 1: 'Y2', 2: 'Y3'})

    dataPartOne = concatDataFrame[['Quarter', 'Year']]
    dataPartTwo = concatDataFrame.drop(['Quarter', 'Year'], axis=1)

# Drop one dimension from the representation to avoid dependency among the variables and convert
    dummyVar = pd.get_dummies(dataPartOne, drop_first=True)
    dummyVar.head()

    concatDataFrame = pd.concat([dataPartTwo, dummyVar], axis=1)
    concatDataFrame.head()

# Applying log transformation on the target feature
    concatDataFrame['num_orders'] = np.log1p(concatDataFrame['num_orders'])
    testDataAnalysis = learnFromTrainData[learnFromTrainData['week'].isin(range(136, 146))]
    trainData = concatDataFrame[concatDataFrame['week'].isin(range(1,136))]
    testData = concatDataFrame[concatDataFrame['week'].isin(range(136,146))]

    # Removing unwanted variables from train dataset
    X_train = trainData.drop(['id', 'num_orders', 'week', 'special price', 'city_code', 'special price percent'], axis=1)

    #num_oreders array of data from traindata
    y_train =  trainData['num_orders']

    # Removing unwanted variables from test dataset
    X_test = testData.drop(['id', 'num_orders', 'week', 'special price', 'city_code', 'special price percent'], axis=1)

    #num_oreders array of data from testdata
    y_test = testData['num_orders']

    # Catboostregressor

    CGB = CatBoostRegressor(learning_rate = 0.3,loss_function = "RMSE",max_depth = 9,verbose= False)
    CGB.fit(X_train,y_train)
    CGBPred = CGB.predict(X_test)

    st.write("Prediction Done!!!!!!!!!!!!!! Thankyou For Your Patience!!!!!")

    
    st.subheader("###### Accuracy Metrics for CATBoostRegressor ######")

    mae = mean_absolute_error(y_test,CGBPred)
    st.markdown("Mean Absolute Error: "+str(mae))

    mse = mean_squared_error(y_test,CGBPred)
    st.markdown("Mean Squared Error: "+str(mse))

    rmse = np.sqrt(mse)
    st.markdown("Root Mean Squared Error: "+str(rmse))

    st.subheader('XGBoostRegressor')
    st.markdown('XGBoost is an implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning.')

    st.markdown('Here we will predict the number of orders for a given week.')

    st.write("Prediction Going ON!!!!!!!!!!!!!! Please Wait")

    

    # XGBoostRegressor

    XGB = XGBRegressor(learning_rate = 0.3,loss_function = "RMSE",max_depth = 9,verbose= False)
    XGB.fit(X_train,y_train)
    XGBPred = XGB.predict(X_test)

    st.write("Prediction Done!!!!!!!!!!!!!! Thankyou For Your Patience!!!!!")

    st.subheader("###### Accuracy Metrics for XGBoost ######")

    maeXGBPred = mean_absolute_error(y_test,XGBPred)
    st.markdown("Mean Absolute Error: "+str(maeXGBPred))

    mseXGBPred = mean_squared_error(y_test,XGBPred)
    st.markdown("Mean Squared Error: "+str(mseXGBPred))

    rmseXGBPred = np.sqrt(mseXGBPred)
    st.markdown("Root Mean Squared Error: "+str(rmseXGBPred))

    st.subheader("###### Result Comparasion ######")

    metname = ['Mean Absolute Error','Mean Square Error','Root Mean Square Error']
    listcatboost = [mae,mse,rmse]
    listxgboost = [maeXGBPred,mseXGBPred,rmseXGBPred]

    df_mertics = pd.DataFrame(list(zip(metname,listcatboost,listxgboost)),columns=['Metric Name','Catboost Accuracy','XGBoost Accuracy'])
    st.table(df_mertics)

    predictedDemandResult = pd.DataFrame(CGBPred)

    predictedDemandResult = np.expm1(predictedDemandResult).astype('int64')

    dataFile = pd.DataFrame(columns=['id','num_orders_predicted','week','city_code','center_id','meal_id','checkout_price','base_price'])

    
    dataFile['id'] = testData['id']
    dataFile['num_orders_predicted'] = predictedDemandResult.values
    dataFile['week'] = testData['week']
    dataFile['city_code'] = testDataAnalysis['city_code']
    dataFile['center_id'] = testDataAnalysis['center_id']
    dataFile['meal_id'] = testDataAnalysis['meal_id']
    dataFile['checkout_price'] = testDataAnalysis['checkout_price']
    dataFile['base_price'] = testDataAnalysis['base_price']

    st.subheader(" ###### Predicted Results ######")
    st.dataframe(dataFile.head())


elif choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the Creators</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    
    st.subheader("Mohit More - 19MIA1005")    
    st.subheader("Madasu Deepika- 19MIA1066")    
    st.subheader("G. Harinisri  - 19MIA1069")    

elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')

