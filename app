import streamlit as st
st.title("PROJECT II")



#pip install streamlit
#pip install pandas
#pip install sklearn




# IMPORT STATEMENTS



import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# FUNCTION
def user_report():
    
    pregnancies = st.sidebar.slider('Pregnancies', 0,17, 1 )
    glucose = st.sidebar.slider('Glucose', 0,200, 189 )
    bp = st.sidebar.slider('Blood Pressure', 0,122, 66 )
    skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 23 )
    insulin = st.sidebar.slider('Insulin', 0,846, 94 )
    bmi = st.sidebar.slider('BMI', 0,67, 28 )
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.000,2.400, 0.167 )
    age = st.sidebar.slider('Age', 21,88, 21 )



    user_report_data = {
        'pregnancies':pregnancies,
        'glucose':glucose,
        'bp':bp,
        'skinthickness':skinthickness,
        'insulin':insulin,
        'bmi':bmi,
        'dpf':dpf,
        'age':age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# READING DATA

df = pd.read_csv(r'C:\Users\sharmin\Downloads\diabetes.csv')
st.header('Dataset Contents')
st.dataframe(df)
st.markdown('###')
# HEADINGS
st.header('Statistics of the Data')
st.write(df.describe())
st.sidebar.header('Patient Data')





#DIABETES DISTRIBUTION
st.header('Diabetes Distribution')
fig = go.Figure(data=[go.Pie(labels=df['Outcome'].value_counts().index, values=df['Outcome'].value_counts(), hole=.3)])
fig.update_layout(legend_title_text='Outcome')
st.plotly_chart(fig)

# CORRELATION MATRIX
st.header('Correlation Matrix')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True)
st.write(fig)

# DISTRIBUTION OF CORRELATION OF FEATURES
st.header('Distribution of Correlation of Features')
df_corr_bar = abs(df.corr()['Outcome']).sort_values()[:-1]
fig = px.bar(df_corr_bar, orientation='h', color_discrete_sequence =['#4285f4']*len(df_corr_bar))
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

# BOX PLOTS
st.text('Outcome vs Blood Pressure')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="BloodPressure", data=df)
st.pyplot(fig)

st.text('Outcome vs Glucose')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="Glucose", data=df)
st.pyplot(fig)

st.text('Outcome vs BMI')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="BMI", data=df)
st.pyplot(fig)

st.text('Outcome vs Age')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="Age", data=df)
st.pyplot(fig)

st.text('Outcome vs Insulin')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="Insulin", data=df)
st.pyplot(fig)

st.text('Outcome vs DiabetesPedigreeFunction')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="DiabetesPedigreeFunction", data=df)
st.pyplot(fig)
st.text('Outcome vs SkinThickness')
fig, ax=plt.subplots()
sns.boxplot(x="Outcome", y="SkinThickness", data=df)
st.pyplot(fig)

# NORMALIZING CONTINUOUS FEATURES
st.header('Normalizing continuous features')
df['Glucose'] = df['Glucose']/df['Glucose'].max()
df['BloodPressure'] = df['BloodPressure']/df['BloodPressure'].max()    
df['SkinThickness'] = df['SkinThickness']/df['SkinThickness'].max()
df['Insulin'] = df['Insulin']/df['Insulin'].max()
df['BMI'] = df['BMI']/df['BMI'].max()
st.dataframe(df)

# SPLITTING X AND Y VALUES
y = df.iloc[:,8]
x = df.iloc[:,0:8]

# PREPARING TRAINING AND TESTING DATA
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



# PATIENT DATA
user_data = user_report()








# MODELS
# SVM
st.header('SVM')
SVM=SVC(kernel='linear')
SVM.fit(x_train, y_train)

# CLASSIFICATION REPORT FOR SVM
y_pred1 = SVM.predict(x_test)
SVM = [accuracy_score(y_test,y_pred1), f1_score(y_test,y_pred1, average='weighted')]
c_report=(classification_report(y_test,y_pred1))
st.text('Classification Report:\n' + c_report)
st.markdown('###')
# CONFUSION MATRIX FOR SVM
st.subheader('Confusion Matrix')
fig, ax=plt.subplots()
cm1 = confusion_matrix(y_test, y_pred1)
ax = sns.heatmap(cm1, annot=True, 
            cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
st.pyplot(fig)
st.markdown('###')
# RANDOM FOREST
st.header('RANDOM FOREST')
RF=RandomForestClassifier()
RF.fit(x_train,y_train)
user_result = RF.predict(user_data)
# CLASSIFICATION REPORT FOR RANDOM FOREST
y_pred2 = RF.predict(x_test)
RF = [accuracy_score(y_test,y_pred2), f1_score(y_test,y_pred2, average='weighted')]
c_report=(classification_report(y_test,y_pred2))
st.text('Classification Report:\n' + c_report)
st.markdown('##')
# CONFUSION MATRIX FOR RANDOM FOREST
st.subheader('Confusion Matrix')
fig, ax=plt.subplots()
cm1 = confusion_matrix(y_test, y_pred2)
ax = sns.heatmap(cm1, annot=True, 
            cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
st.pyplot(fig)

# MODELS COMPARISON
st.header('\t\tAccuracy Comparison for Models')
accuracy_SVM=accuracy_score(y_test,y_pred1)
accuracy_RF=accuracy_score(y_test,y_pred2)
models=[('Support Vector Machine(linear)',accuracy_SVM),
        ('Random Forest',accuracy_RF)]
predict=pd.DataFrame(data=models,columns=['Model','Accuracy'])
st.write(predict)







# COLOR FUNCTION
if user_result[0]==0:
    color = 'blue'
else:
    color = 'red'


# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.subheader(output)
#st.text('Accuracy: ')
#st.write(str(accuracy_score(y_test, SVM.predict(x_test))*100)+'%')

