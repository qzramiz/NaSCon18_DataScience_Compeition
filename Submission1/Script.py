# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:18:00 2018

@author: zaid
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from cross import cross_val
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
import glob
from collections import Counter
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE,ADASYN
#from help import *
#from test_predict import build_model

patient_record_train=pd.read_csv('Patient_Records_Train.csv')
Disease_Dictionay={'Myocardial infarction':1,'Healthy control':2,'Unidentified':3,'Dysrhythmia':4,
                   'Valvular heart disease':5,'Hypertrophy':6,'Myocarditis':7,'Cardiomyopathy':8,
                   'Bundle branch block':9}
model=0
def convert_to_integer_encoding(data,colname):
    temp=data[:]
    temp[colname]=temp[colname].astype('category')
    temp[colname]=temp[colname].cat.codes
    return temp


def columns_to_encode(Data):
    temp=Data[:]
    b=temp.select_dtypes(include=['object'])
    name_cols=b.columns.tolist()
    for columns in name_cols:
        temp=convert_to_integer_encoding(temp,columns)
    
    return temp

def columns_to_encode_test(TestData):
    testtemp=TestData[:]
    b=testtemp.select_dtypes(include=['object'])
    name_cols=b.columns.tolist()
    #print (len(b))
    for columns in name_cols:
        #print (columns)
        testtemp=convert_to_integer_encoding(testtemp,columns)
    
    return testtemp
    

def remove_null_columns(data):
    temp=data[:]
    for col in temp.columns.values:
        if(temp[col].isnull().sum()>len(data)-24):
            del (temp[col])
    
    return temp

def remove_null_columns_sample(data):
    temp=data[:]
    temp=temp.replace(to_replace=' n/a',value=np.nan)
    for col in temp.columns.values:
        if(temp[col].isnull().sum()>(len(data)/2)):
            del (temp[col])
    
    return temp

def fill_missing(data):
    temp=data[:]
    cols=temp.columns[temp.isnull().any()].tolist()
    for columns in cols:
        if (temp[columns].dtype=='float64' or temp[columns].dtype=='int8' or temp[columns].dtype=='int64'):
            temp[columns]=temp[columns].fillna(temp[columns].median())
    return temp
    


def build_model(column_list):
    Test_Data=pd.read_csv('Patient_Records_Test.csv')
    Ids=Test_Data['patientID']
    Testing_Data=Test_Data[column_list]
    Testing_Data=columns_to_encode_test(Testing_Data)
  #  Testing_Data=remove_null_columns(Testing_Data)
    Testing_Data=fill_missing(Testing_Data)
    TestSet=Testing_Data.values
    
    print ('normal')
    pred_ecg=ecg_incorporation()
    print ('ecg')
    pred_diff=build_different_samples_models()
    predictions=model.predict(TestSet)
    print ('end')
    
    return predictions,pred_ecg,pred_diff
    

def simple_sampling():
    patient_record=pd.read_csv('Patient_Records_Train.csv')

    labels=pd.unique(patient_record['Disease'].values)
    patient_1=patient_record[patient_record.Disease==labels[0]]
    patient_2=patient_record[patient_record.Disease==labels[1]]
    
    patient_3=patient_record[patient_record.Disease==labels[2]]
    patient_4=patient_record[patient_record.Disease==labels[3]]
    patient_5=patient_record[patient_record.Disease==labels[4]]
    patient_6=patient_record[patient_record.Disease==labels[5]]
    patient_7=patient_record[patient_record.Disease==labels[6]]
    patient_8=patient_record[patient_record.Disease==labels[7]]
    patient_9=patient_record[patient_record.Disease==labels[8]]
    
    
    
    patient_1_up = resample(patient_1, 
                                     replace=True,     # sample with replacement
                                     n_samples=98,    # to match majority class
                                     random_state=123) # reproducible results
    
    patient_2_up = resample(patient_2, 
                                     replace=True,     # sample with replacement
                                     n_samples=90,    # to match majority class
                                     random_state=123) # reproducible results
    patient_3_up = resample(patient_3, 
                                     replace=True,     # sample with replacement
                                     n_samples=80,    # to match majority class
                                     random_state=123) # reproducible results
    patient_4_up = resample(patient_4, 
                                     replace=True,     # sample with replacement
                                     n_samples=70,    # to match majority class
                                     random_state=123) # reproducible results
    patient_5_up = resample(patient_5, 
                                     replace=True,     # sample with replacement
                                     n_samples=50,    # to match majority class
                                     random_state=123) # reproducible results
    patient_6_up = resample(patient_6, 
                                     replace=True,     # sample with replacement
                                     n_samples=50,    # to match majority class
                                     random_state=123) # reproducible results
    patient_7_up = resample(patient_7, 
                                     replace=True,     # sample with replacement
                                     n_samples=50,    # to match majority class
                                     random_state=123) # reproducible results
    patient_8_up = resample(patient_8, 
                                     replace=True,     # sample with replacement
                                     n_samples=50,    # to match majority class
                                     random_state=123) # reproducible results
    patient_9_up = resample(patient_9, 
                                     replace=True,     # sample with replacement
                                     n_samples=50,    # to match majority class
                                     random_state=123) # reproducible results
    
    sampled_data=pd.concat((patient_1_up,patient_2_up,patient_3_up,patient_4_up,patient_5_up
                            ,patient_6_up,patient_7_up,patient_8_up,patient_9_up))
    sampled_data=sampled_data.sample(frac=1)
    Target_disease_sampled=remove_null_columns(sampled_data)
    Target_disease_sampled=fill_missing(Target_disease_sampled)
    Target_disease_sampled=Target_disease_sampled.drop(['Disease','Admission date','Catheterization date'
                                    ,'Infarction date','Infarction date (acute)','patientID'],axis=1)
    label_sampled=sampled_data['Disease'].values
    new_target_sampled=columns_to_encode(Target_disease_sampled)
    new_target_sampled=new_target_sampled.values
    #model_sample=XGBClassifier(objective='multi:sofprob',num_class=9,eval='mlogloss',max_depth=10,num_rounds=2000)
    model_sample_rf=RandomForestClassifier()
    model_sample_rf.fit(new_target_sampled,label_sampled)
    print (f1_score(label_sampled[150:],model_sample.predict(new_target_sampled[150:,:]),average='micro'))
    


def build_model_with_probabilities(column_list,model_ens):
    Test_Data=pd.read_csv('Patient_Records_Test.csv')
    Ids=Test_Data['patientID']
    Testing_Data=Test_Data[column_list]
    Testing_Data=columns_to_encode_test(Testing_Data)
    #Testing_Data=remove_null_columns(Testing_Data)
    Testing_Data=fill_missing(Testing_Data)
    TestSet=Testing_Data.values
    predictions_prob=model_ens.predict_proba(TestSet)
    pred=model_ens.predict(TestSet)
    #submit_frame=pd.DataFrame({'patientID':Ids,'Disease':pp})
    #submit_frame.to_csv('Submission_v8pc.csv',index=False)
    return predictions_prob,pred



def build_different_samples_models():
    patient_record=pd.read_csv('Patient_Records_Train.csv')
    
    label=patient_record['Disease']
    labels=pd.unique(patient_record['Disease'].values)
    
    
    patient_1=patient_record[patient_record.Disease==labels[0]]
    
    
    
    patient_2=patient_record[patient_record.Disease==labels[1]]
    
    patient_3=patient_record[patient_record.Disease==labels[2]]
    patient_4=patient_record[patient_record.Disease==labels[3]]
    patient_5=patient_record[patient_record.Disease==labels[4]]
    patient_6=patient_record[patient_record.Disease==labels[5]]
    patient_7=patient_record[patient_record.Disease==labels[6]]
    patient_8=patient_record[patient_record.Disease==labels[7]]
    patient_9=patient_record[patient_record.Disease==labels[8]]
    
    
    patient_3or=patient_3[:]
    
    
    patient_1=pd.concat((patient_1,patient_2))
    patient_2=pd.concat((patient_2,patient_3))
    patient_3=pd.concat((patient_3,patient_4))
    patient_4=pd.concat((patient_4,patient_5))
    patient_5=pd.concat((patient_5,patient_6))
    patient_6=pd.concat((patient_6,patient_7))
    patient_7=pd.concat((patient_7,patient_8))
    patient_8=pd.concat((patient_8,patient_9))
    patient_9=pd.concat((patient_9,patient_3or))    

    
    ## Disease 1
    patient_1_disease=remove_null_columns_sample(patient_1)
    patient_1_disease=fill_missing(patient_1_disease)
    label_patient_1_disease=patient_1_disease['Disease'].values
    patient_1_disease=patient_1_disease.drop(['Disease','patientID'],axis=1)
    patient_1_disease=columns_to_encode(patient_1_disease)
    column_list_disease1=patient_1_disease.columns
    model1=LogisticRegression()
    model1.fit(patient_1_disease.values,label_patient_1_disease)
    confidence_disease_1,pred_1=build_model_with_probabilities(column_list_disease1,model1)
    
    
    
    ## Disease 2
    patient_2_disease=remove_null_columns_sample(patient_2)
    patient_2_disease=fill_missing(patient_2_disease)
    label_patient_2_disease=patient_2_disease['Disease'].values
    patient_2_disease=patient_2_disease.drop(['Disease','patientID'],axis=1)
    patient_2_disease=columns_to_encode(patient_2_disease)
    column_list_disease2=patient_2_disease.columns
    model2=LogisticRegression()
    model2.fit(patient_2_disease.values,label_patient_2_disease)
    confidence_disease_2,pred_2=build_model_with_probabilities(column_list_disease2,model2)
    
    
    ## Disease 3
    patient_3_disease=remove_null_columns_sample(patient_3)
    patient_3_disease=fill_missing(patient_3_disease)
    label_patient_3_disease=patient_3_disease['Disease'].values
    patient_3_disease=patient_3_disease.drop(['Disease','patientID'],axis=1)
    patient_3_disease=columns_to_encode(patient_3_disease)
    column_list_disease3=patient_3_disease.columns
    model3=LogisticRegression()
    model3.fit(patient_3_disease.values,label_patient_3_disease)
    confidence_disease_3,pred_3=build_model_with_probabilities(column_list_disease3,model3)
    
    ## Disease 4
    patient_4_disease=remove_null_columns_sample(patient_4)
    patient_4_disease=fill_missing(patient_4_disease)
    label_patient_4_disease=patient_4_disease['Disease'].values
    patient_4_disease=patient_4_disease.drop(['Disease','patientID'],axis=1)
    patient_4_disease=columns_to_encode(patient_4_disease)
    column_list_disease4=patient_4_disease.columns
    model4=LogisticRegression()
    model4.fit(patient_4_disease.values,label_patient_4_disease)
    model=joblib.load('model.pkl')
    confidence_disease_4,pred_4=build_model_with_probabilities(column_list_disease4,model4)
    
    
    ## Disease 5
    patient_5_disease=remove_null_columns_sample(patient_5)
    patient_5_disease=fill_missing(patient_5_disease)
    label_patient_5_disease=patient_5_disease['Disease'].values
    patient_5_disease=patient_5_disease.drop(['Disease','patientID'],axis=1)
    patient_5_disease=columns_to_encode(patient_5_disease)
    column_list_disease5=patient_5_disease.columns
    model5=LogisticRegression()
    model5.fit(patient_5_disease.values,label_patient_5_disease)
    confidence_disease_5,pred_5=build_model_with_probabilities(column_list_disease5,model5)
    
    
    
    ## Disease 6
    patient_6_disease=remove_null_columns_sample(patient_6)
    patient_6_disease=fill_missing(patient_6_disease)
    label_patient_6_disease=patient_6_disease['Disease'].values
    patient_6_disease=patient_6_disease.drop(['Disease','patientID'],axis=1)
    patient_6_disease=columns_to_encode(patient_6_disease)
    column_list_disease6=patient_6_disease.columns
    model6=LogisticRegression()
    model6.fit(patient_6_disease.values,label_patient_6_disease)
    confidence_disease_6,pred_6=build_model_with_probabilities(column_list_disease6,model6)
    
    
    ## Disease 7
    patient_7_disease=remove_null_columns_sample(patient_7)
    patient_7_disease=fill_missing(patient_7_disease)
    label_patient_7_disease=patient_7_disease['Disease'].values
    patient_7_disease=patient_7_disease.drop(['Disease','patientID'],axis=1)
    patient_7_disease=columns_to_encode(patient_7_disease)
    column_list_disease7=patient_7_disease.columns
    model7=LogisticRegression()
    model7.fit(patient_7_disease.values,label_patient_7_disease)
    confidence_disease_7,pred_7=build_model_with_probabilities(column_list_disease7,model7)
    
    ## Disease 8
    patient_8_disease=remove_null_columns_sample(patient_8)
    patient_8_disease=fill_missing(patient_8_disease)
    label_patient_8_disease=patient_8_disease['Disease'].values
    patient_8_disease=patient_8_disease.drop(['Disease','patientID'],axis=1)
    patient_8_disease=columns_to_encode(patient_8_disease)
    column_list_disease8=patient_8_disease.columns
    model8=LogisticRegression()
    model8.fit(patient_8_disease.values,label_patient_8_disease)
    confidence_disease_8,pred_8=build_model_with_probabilities(column_list_disease8,model8)
    
    ## Disease 9
    patient_9_disease=remove_null_columns_sample(patient_9)
    patient_9_disease=fill_missing(patient_9_disease)
    label_patient_9_disease=patient_9_disease['Disease'].values
    patient_9_disease=patient_9_disease.drop(['Disease','patientID'],axis=1)
    patient_9_disease=columns_to_encode(patient_9_disease)
    column_list_disease9=patient_9_disease.columns
    model9=LogisticRegression()
    model9.fit(patient_9_disease.values,label_patient_9_disease)
    confidence_disease_9,pred_9=build_model_with_probabilities(column_list_disease9,model9)
    
    
    final_pred=[]

    for index in range(110):
        disease1=pred_1[index]
        disease1_prob=confidence_disease_1[index]
        
        disease2=pred_2[index]
        disease2_prob=confidence_disease_1[index]
        
    
        disease3=pred_3[index]
        disease3_prob=confidence_disease_3[index]
        
        disease4=pred_4[index]
        disease4_prob=confidence_disease_4[index]
        
        disease5=pred_5[index]
        disease5_prob=confidence_disease_5[index]
        
        disease6=pred_6[index]
        disease6_prob=confidence_disease_6[index]
        
        disease7=pred_7[index]
        disease7_prob=confidence_disease_7[index]
        
        disease8=pred_8[index]
        disease8_prob=confidence_disease_8[index]
        
        disease9=pred_9[index]
        disease9_prob=confidence_disease_9[index]
    
        probs=[np.max(disease1_prob),np.max(disease2_prob),np.max(disease3_prob),np.max(disease4_prob),np.max(disease5_prob),np.max(disease6_prob),
               np.max(disease7_prob),np.max(disease8_prob),np.max(disease9_prob)]
        diseases=[disease1,disease2,disease3,disease4,disease5,disease6,disease7,disease8,disease9]
        highest=np.argmax(probs)
        final_pred.append(diseases[highest])
    return final_pred
    


def build_ensemble(Datset,labels):
    model_lr=LogisticRegression()
    model_xgb=XGBClassifier(objective='multi:softmax',num_class=9,eval='merror')
    model_lda=LinearDiscriminantAnalysis()
    #model_svm=SVC(kernel='poly',degree=9)
    model_rf=RandomForestClassifier(n_estimators=9)
    
    ensemble=VotingClassifier(estimators=[('lr',model_lr),('xgb',model_xgb),('lda',model_lda),
                                          ('rf',model_rf)],voting='hard')
    
    ensemble.fit(Datset,labels)
    print (f1_score(label[150:],ensemble.predict(new_data[150:,:]),average='micro'))
    




## ECG Incorporation
def ecg_incorporation():
    Ids=patient_record_train['patientID']
    label=patient_record_train['Disease'].values
    counter=0
    array_ecg=np.zeros((173,115,15))
    for  ids in Ids:
        temp=pd.read_csv('TrainECGData/'+str(ids)+'/0.csv')
        count_ecgs=0
        for filename in glob.glob('TrainECGData/'+str(ids)+'/*.csv'):
            count_ecgs+=1
            if (filename != 'TrainECGData/'+str(ids)+'/0.csv'):
                filee=pd.read_csv(filename)
                temp=temp.add(filee)
        del (temp["'sample #'"])
        final_ecg=np.ceil(temp/count_ecgs)
        final_ecg=final_ecg.fillna(final_ecg.mean())
        
        temp_array_ecg=np.zeros((115,15))
        count=0
        for second in range(0,len(final_ecg)-200,int(len(final_ecg)/115)):
            temp_data=final_ecg[second:second+1000]
            temp_array_ecg[count,:]=np.ceil(temp_data.mean())
            count+=1
        array_ecg[counter,:,:]=temp_array_ecg
            
        counter+=1
    
    # building model
    models=[]
    for feature in range(0,15):
        vector=array_ecg[:,:,feature]
        vector_n,label_n=ADASYN(n_neighbors=1).fit_sample(vector,label)
        model=XGBClassifier()
        model.fit(vector_n,label_n)
        models.append(model)
    predc=ecg_test(models)
    return predc


def ecg_test(models):
    patient_record=pd.read_csv('Patient_Records_Test.csv')
    
    Ids=patient_record['patientID']
    #label=patient_record['Disease']
    counter=0
    array_ecg_test=np.zeros((110,115,15))
    for  ids in Ids:
        temp=pd.read_csv('TestECGData/'+str(ids)+'/0.csv')
        count_ecgs=0
        for filename in glob.glob('TestECGData/'+str(ids)+'/*.csv'):
            count_ecgs+=1
            if (filename != 'TestECGData/'+str(ids)+'/0.csv'):
                filee=pd.read_csv(filename)
                temp=temp.add(filee)
        del (temp["'sample #'"])
        final_ecg=np.ceil(temp/count_ecgs)
        final_ecg=final_ecg.fillna(final_ecg.mean())
        
        temp_array_ecg=np.zeros((115,15))
        count=0
        for second in range(0,len(final_ecg)-200,int(len(final_ecg)/115)):
            temp_data=final_ecg[second:second+1000]
            temp_array_ecg[count,:]=np.ceil(temp_data.mean())
            count+=1
        array_ecg_test[counter,:,:]=temp_array_ecg
            
        counter+=1
    
    # building model
    vector=array_ecg_test[:,:,0]
    model=models[0]
    prediction=model.predict(vector)
    
    for feature in range(1,15):
        vector=array_ecg_test[:,:,feature]
        model=models[feature]
        prediction_n=model.predict(vector)
        prediction=np.column_stack((prediction,prediction_n))
        
    final_pred_train=[];
    for i in range(len(prediction)):
        example=prediction[i,:];    #pick one row
        labs = Counter(example) #Elements with their frequencies
        mod=labs.most_common(1); #Pick out the most occuring element
        final_pred_train.append(mod[0][0]);    
        
    
    return final_pred_train





Target_disease=remove_null_columns(patient_record_train)
Target_disease=fill_missing(Target_disease)
Target_disease=Target_disease.drop(['Disease','Admission date','Catheterization date'
                                    ,'Infarction date','Infarction date (acute)','patientID'],axis=1)
#Target_disease=Target_disease.replace({'Disease':Disease_Dictionay})
label=patient_record_train['Disease'].values
new_data=columns_to_encode(Target_disease)

#data_train=Target_disease.as_matrix()
new_data=new_data.values

model=XGBClassifier(objective='multi:softmax',num_class=9,eval='merror')
model.fit(new_data,label)


column_list=Target_disease.columns

pred,pred1,pred2=build_model(column_list)


#pred=model.predict(new_data)
#print (f1_score(label,model.predict(new_data),average='macro'))