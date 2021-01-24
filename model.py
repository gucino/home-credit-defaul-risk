# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 10:58:09 2020

@author: Tisana
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#import file
application_train=pd.read_csv("application_train.csv")
bureau=pd.read_csv("bureau.csv")
bureau_balance=pd.read_csv("bureau_balance.csv")
previous_application=pd.read_csv("previous_application.csv")
POH_cash_balance=pd.read_csv("POS_CASH_balance.csv")
credit_card_balance=pd.read_csv("credit_card_balance.csv")
installments_payment=pd.read_csv("installments_payments.csv")

##############################################################################
##############################################################################
##############################################################################
#DATA EXPLORATION

#1.Gender and Payment difficulty:1 is male| 2 is female
target=application_train['TARGET'].values
gender=application_train['CODE_GENDER'].values
gender_index=np.zeros(len(gender))
gender_index[gender=="M"]=1
gender_index[gender=="F"]=2

total_male=gender_index==1
total_male=total_male.astype(int).sum()
total_female=gender_index==2
total_female=total_female.astype(int).sum()

gender_index=gender_index[target==1]

matrix=np.unique(gender_index,return_counts=True)
male_percent_payment_difficulties=matrix[1][0]/total_male
female_percent_payment_difficulties=matrix[1][1]/total_female

plt.figure()
plt.title("Payment difficulties between Male and Female")
plt.bar(0,male_percent_payment_difficulties,label="percent of Male that have payment difficulties",color="blue")
plt.bar(1,female_percent_payment_difficulties,label="percent of Female that have payment difficulties",color="pink")
plt.ylabel("percent")
plt.xticks(ticks=[0,1], labels=["Male","Female"])
plt.legend()
plt.show()

#2.Age vs Payment difficulties
age=application_train['DAYS_BIRTH'].values
age=(-age)/365
age_with_difficulty=age[target==1]
count_age=np.zeros((5))
count_age_dificulties=np.zeros((5))
#min age=20
#max age=70
#therefore: 20-30,30-40,40-50,50-60,60-70
for i in range(0,len(age)):
    a=age[i]
    if 20<=a<30:
        index=0
    if 30<=a<40:
        index=1
    if 40<=a<50:
        index=2
    if 50<=a<60:
        index=3
    if 60<=a<70:
        index=4
    count_age[index]+=1
    if target[i]==1:
        count_age_dificulties[index]+=1

percent_payment_diff_each_age=count_age_dificulties/count_age

plt.figure()
plt.title("Payment difficulties for different AGE group")
plt.bar(np.arange(0,5),percent_payment_diff_each_age)
plt.xticks(ticks=np.arange(0,5), labels=["20-30","30-40","40-50",
                                       "50-60","60-70"])  
plt.ylabel("percent")
plt.show()    
        
#3.relationship status VS payment difficulties
family_status=application_train["NAME_FAMILY_STATUS"].values
family_status_index=np.zeros(family_status.shape)
family_status_index[family_status=="Single / not married"]=1
family_status_index[family_status=="Married"]=2

total_single=family_status_index==1
total_married=family_status_index==2
total_other=family_status_index==0
total_single=total_single.astype(int).sum()
total_married=total_married.astype(int).sum()
total_other=total_other.astype(int).sum()

count_diff=[0,0,0]
for i in range(0,len(target)):
    if target[i]==1:
        if family_status_index[i]==1:
           count_diff[0]+=1 
        if family_status_index[i]==2:
           count_diff[1]+=1 
        if family_status_index[i]==0:
           count_diff[2]+=1 
percent_diff_each_fam_status=np.array(count_diff)/np.array([total_single,\
                                     total_married,total_other])

plt.figure()
plt.title("Payment difficulties for different Family Status")
plt.bar([0,1,2],percent_diff_each_fam_status)
plt.ylabel("percent")
plt.xticks(ticks=[0,1,2], labels=["SINGLE","MARRIED","others"])
plt.show()

#4.education status VS payment difficulties
educational_status=application_train["NAME_EDUCATION_TYPE"].values
educational_status_index=np.zeros(educational_status.shape)
educational_status_index[educational_status=="Secondary / secondary special"]=1
educational_status_index[educational_status=="Higher education"]=2

count_status=[0,0,0]
count_status_diff=[0,0,0]
for i in range(0,len(target)):
    if educational_status_index[i]==1:
        index=0
    if educational_status_index[i]==2:
        index=1
    if educational_status_index[i]==0:
        index=2
    count_status[index]+=1
    if target[i]==1:
        count_status_diff[index]+=1
percent_diff=np.array(count_status_diff)/np.array(count_status)

plt.figure()
plt.title("Payment difficulties for different EDUCATIONAL Status")
plt.bar([0,1,2],percent_diff)
plt.ylabel("percent")
plt.xticks(ticks=[0,1,2], labels=["SECONDARY","HIGHER","others"])
plt.show()


##############################################################################
##############################################################################
##############################################################################
#MODEL 1: consider only main table

#make application an array
a=np.zeros((application_train.values.shape[0],128))

a[:,[0,1]]=application_train.values[:,[0,1]]

cash_loan=application_train.values[:,2]=="Cash loans"
cash_loan=cash_loan.astype(int)
reloving_loan=np.zeros(cash_loan.shape)
reloving_loan[cash_loan==0]=1
a[:,2]=cash_loan
a[:,3]=reloving_loan

male=application_train.values[:,3]=="M"
female=application_train.values[:,3]=="F"
a[:,4]=male.astype(int)
a[:,5]=female.astype(int)

car=application_train.values[:,4]=="Y"
no_car=application_train.values[:,4]=="N"
a[:,6]=car.astype(int)
a[:,7]=no_car.astype(int)

realty=application_train.values[:,5]=="Y"
no_realty=application_train.values[:,5]=="N"
a[:,7]=realty.astype(int)
a[:,8]=no_realty.astype(int)

a[:,[9,10,11,12,13]]=application_train.values[:,[6,7,8,9,10]]

unaccompanied=application_train.values[:,11]=="Unaccompanied"
unaccompanied=unaccompanied.astype(int)
Family=application_train.values[:,11]=="Family"
Family=Family.astype(int)
other=np.zeros(unaccompanied.shape)
other[Family+unaccompanied==0]=1
a[:,14]=unaccompanied
a[:,15]=Family
a[:,16]=other

working=application_train.values[:,12]=="Working"
working=working.astype(int)
commercial=application_train.values[:,12]=="Commercial"
commercial=commercial.astype(int)
other=np.zeros(working.shape)
other[working+commercial==0]=1
a[:,17]=working
a[:,18]=commercial
a[:,19]=other

sec=application_train.values[:,13]=="Secondary / secondary special"
sec=sec.astype(int)
higher=application_train.values[:,13]=="Higher education"
higher=higher.astype(int)
other=np.zeros(working.shape)
other[higher+sec==0]=1
a[:,20]=sec
a[:,21]=higher
a[:,22]=other

maried=application_train.values[:,14]=="Married"
maried=maried.astype(int)
single=application_train.values[:,14]=="Single / not married"
single=single.astype(int)
other=np.zeros(working.shape)
other[single+maried==0]=1
a[:,23]=maried
a[:,24]=single
a[:,25]=other

house=application_train.values[:,15]=="House / apartment"
house=house.astype(int)
parent=application_train.values[:,15]=="With parents"
parent=parent.astype(int)
other=np.zeros(working.shape)
other[house+parent==0]=1
a[:,26]=house
a[:,27]=parent
a[:,28]=other

a[:,[28,29,30,31,32,33,34,35,36,37,38,39]]=application_train.values[:,[16,17,18,19,20,21,22,23,24,25,26,27]]
a[:,[40,41,42]]=application_train.values[:,[29,30,31]]

tuesday=application_train.values[:,32]=="TUESDAY"
tuesday=tuesday.astype(int)
wednesday=application_train.values[:,32]=="WEDNESDAY"
wednesday=wednesday.astype(int)
other=np.zeros(working.shape)
other[tuesday+wednesday==0]=1
a[:,43]=tuesday
a[:,44]=wednesday
a[:,45]=other

a[:,[46,47,48,49,50,51,52]]=application_train.values[:,[33,34,35,36,37,38,39]]

index=list(np.arange(41,121))
index.remove(86)
index.remove(87)
index.remove(89)
index.remove(90)

index_for_a=list(np.array(index)+7)
a[:,index_for_a]=application_train.values[:,index]

application_train_array=np.copy(a)

X=application_train_array[:,2:]
Y=application_train_array[:,1]
X[np.isnan(X)==True]=0

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.fit_transform(X_test)

classifier1=LogisticRegression()
classifier1.fit(scaled_X_train,Y_train)

Y_predicted_train=classifier1.predict(scaled_X_train)
Y_predicted_test=classifier1.predict(scaled_X_test)

check_train=Y_predicted_train==Y_train
check_train=check_train.astype(int)
print("training acc : ",check_train.sum()/len(check_train))

check_test=Y_predicted_test==Y_test
check_test=check_test.astype(int)
print("test acc : ",check_test.sum()/len(check_test))

prob_Y_test = classifier1.predict_proba(scaled_X_test)
prob_Y_test=prob_Y_test[:,1] #prob of result 1 only

from sklearn import metrics
fpr1, tpr1, thresholds = metrics.roc_curve(Y_test, prob_Y_test)

plt.figure()
plt.title("ROC curve for model 1")
plt.plot(fpr1,tpr1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


def find_bias(Y_actual,Y_predicted):
    #find bias in training set
    a=np.unique(Y_actual,return_counts=True)
    total_one=a[1][1]
    total_zero=a[1][0]
    
    #for one
    actually_one=Y_actual==1
    actually_one=actually_one.astype(int)
    
    predicted_zero=Y_predicted==0
    predicted_zero=predicted_zero.astype(int)
    predicted_zero[actually_one==0]=0
    
    predicted_zero_but_actually_one=predicted_zero.sum()
    
    #for zero
    actually_zero=Y_actual==0
    actually_zero=actually_zero.astype(int)
    
    predicted_one=Y_predicted==1
    predicted_one=predicted_one.astype(int)
    predicted_one[actually_zero==0]=0
    
    predicted_one_but_actually_zero=predicted_one.sum()
    print(predicted_zero_but_actually_one/total_one*100," of one is predicted as zero")
    print(predicted_one_but_actually_zero/total_zero*100," of zero is predicted as one")
    return total_one,total_zero,predicted_zero_but_actually_one,predicted_one_but_actually_zero


find_bias(np.concatenate((Y_train,Y_test)),np.concatenate((Y_predicted_train,Y_predicted_test)))

##############################################################################
##############################################################################
##############################################################################
#MODEL 2: reduce bias by making one and zero equal


#extract dimension information: num of loan in each revious is not the same

all_loan_ID=list((application_train.values)[:,0])
num_previous_loan_from_Credit_Bureau=np.unique(list(bureau.values[:,0]),return_counts=True)
num_previous_loan_from_Home_credit=np.unique(list(previous_application.values[:,1]),return_counts=True)



def count_one_and_zero(application_train):
    a=application_train.values

    SK_ID_CURR_one=list(np.unique(np.where(a[:,1]==1,a[:,0],0)))
    SK_ID_CURR_one.remove(0)
    SK_ID_CURR_zero=list(np.unique(np.where(a[:,1]==0,a[:,0],0)))
    SK_ID_CURR_zero.remove(0)
    
    num_one=len(SK_ID_CURR_one)
    num_zero=len(SK_ID_CURR_zero)
    
    return num_one,num_zero

one,zero=count_one_and_zero(application_train)
plt.figure()
plt.title("class 0 to class 1 ratio")
plt.pie([one,one],labels=["class 0","class 1"])
plt.legend()
plt.show()
#reduce the size of dataset by undersampling
    
a=application_train.values

SK_ID_CURR_one=list(np.unique(np.where(a[:,1]==1,a[:,0],0)))
SK_ID_CURR_one.remove(0)
SK_ID_CURR_zero=list(np.unique(np.where(a[:,1]==0,a[:,0],0)))
SK_ID_CURR_zero.remove(0)

num_one=len(SK_ID_CURR_one)
num_zero=len(SK_ID_CURR_zero)

random_SK_ID_CURR_zero=np.random.choice(SK_ID_CURR_zero,num_one,replace=False)

SK_ID_CURR_to_consider=list(random_SK_ID_CURR_zero)+SK_ID_CURR_one
np.random.shuffle(SK_ID_CURR_to_consider)

#upate new dataset with equal number of zero and one

application_train=application_train[application_train["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]

#make application an array
a=np.zeros((application_train.values.shape[0],128))

a[:,[0,1]]=application_train.values[:,[0,1]]

cash_loan=application_train.values[:,2]=="Cash loans"
cash_loan=cash_loan.astype(int)
reloving_loan=np.zeros(cash_loan.shape)
reloving_loan[cash_loan==0]=1
a[:,2]=cash_loan
a[:,3]=reloving_loan

male=application_train.values[:,3]=="M"
female=application_train.values[:,3]=="F"
a[:,4]=male.astype(int)
a[:,5]=female.astype(int)

car=application_train.values[:,4]=="Y"
no_car=application_train.values[:,4]=="N"
a[:,6]=car.astype(int)
a[:,7]=no_car.astype(int)

realty=application_train.values[:,5]=="Y"
no_realty=application_train.values[:,5]=="N"
a[:,7]=realty.astype(int)
a[:,8]=no_realty.astype(int)

a[:,[9,10,11,12,13]]=application_train.values[:,[6,7,8,9,10]]

unaccompanied=application_train.values[:,11]=="Unaccompanied"
unaccompanied=unaccompanied.astype(int)
Family=application_train.values[:,11]=="Family"
Family=Family.astype(int)
other=np.zeros(unaccompanied.shape)
other[Family+unaccompanied==0]=1
a[:,14]=unaccompanied
a[:,15]=Family
a[:,16]=other

working=application_train.values[:,12]=="Working"
working=working.astype(int)
commercial=application_train.values[:,12]=="Commercial"
commercial=commercial.astype(int)
other=np.zeros(working.shape)
other[working+commercial==0]=1
a[:,17]=working
a[:,18]=commercial
a[:,19]=other

sec=application_train.values[:,13]=="Secondary / secondary special"
sec=sec.astype(int)
higher=application_train.values[:,13]=="Higher education"
higher=higher.astype(int)
other=np.zeros(working.shape)
other[higher+sec==0]=1
a[:,20]=sec
a[:,21]=higher
a[:,22]=other

maried=application_train.values[:,14]=="Married"
maried=maried.astype(int)
single=application_train.values[:,14]=="Single / not married"
single=single.astype(int)
other=np.zeros(working.shape)
other[single+maried==0]=1
a[:,23]=maried
a[:,24]=single
a[:,25]=other

house=application_train.values[:,15]=="House / apartment"
house=house.astype(int)
parent=application_train.values[:,15]=="With parents"
parent=parent.astype(int)
other=np.zeros(working.shape)
other[house+parent==0]=1
a[:,26]=house
a[:,27]=parent
a[:,28]=other

a[:,[28,29,30,31,32,33,34,35,36,37,38,39]]=application_train.values[:,[16,17,18,19,20,21,22,23,24,25,26,27]]
a[:,[40,41,42]]=application_train.values[:,[29,30,31]]

tuesday=application_train.values[:,32]=="TUESDAY"
tuesday=tuesday.astype(int)
wednesday=application_train.values[:,32]=="WEDNESDAY"
wednesday=wednesday.astype(int)
other=np.zeros(working.shape)
other[tuesday+wednesday==0]=1
a[:,43]=tuesday
a[:,44]=wednesday
a[:,45]=other

a[:,[46,47,48,49,50,51,52]]=application_train.values[:,[33,34,35,36,37,38,39]]

index=list(np.arange(41,121))
index.remove(86)
index.remove(87)
index.remove(89)
index.remove(90)

index_for_a=list(np.array(index)+7)
a[:,index_for_a]=application_train.values[:,index]

application_train_array=np.copy(a)

X=application_train_array[:,2:]
Y=application_train_array[:,1]
X[np.isnan(X)==True]=0


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.fit_transform(X_test)

classifier2=LogisticRegression()
classifier2.fit(scaled_X_train,Y_train)

Y_predicted_train=classifier2.predict(scaled_X_train)
Y_predicted_test=classifier2.predict(scaled_X_test)

check_train=Y_predicted_train==Y_train
check_train=check_train.astype(int)
print("training acc : ",check_train.sum()/len(check_train))

check_test=Y_predicted_test==Y_test
check_test=check_test.astype(int)
print("test acc : ",check_test.sum()/len(check_test))

prob_Y_test = classifier2.predict_proba(scaled_X_test)
prob_Y_test=prob_Y_test[:,1] #prob of result 1 only

from sklearn import metrics
fpr2, tpr2, thresholds = metrics.roc_curve(Y_test, prob_Y_test)

plt.figure()
plt.title("ROC curve for model 2")
plt.plot(fpr2,tpr2)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

find_bias(np.concatenate((Y_train,Y_test)),np.concatenate((Y_predicted_train,Y_predicted_test)))

##############################################################################
##############################################################################
##############################################################################
#MODEL 3: include previous record of client

application_train=pd.read_csv("application_train.csv")
bureau=pd.read_csv("bureau.csv")
bureau_balance=pd.read_csv("bureau_balance.csv")
previous_application=pd.read_csv("previous_application.csv")
POH_cash_balance=pd.read_csv("POS_CASH_balance.csv")
credit_card_balance=pd.read_csv("credit_card_balance.csv")
installments_payment=pd.read_csv("installments_payments.csv")

a=application_train.values

SK_ID_CURR_one=list(np.unique(np.where(a[:,1]==1,a[:,0],0)))
SK_ID_CURR_one.remove(0)
SK_ID_CURR_zero=list(np.unique(np.where(a[:,1]==0,a[:,0],0)))
SK_ID_CURR_zero.remove(0)

num_one=len(SK_ID_CURR_one)
num_zero=len(SK_ID_CURR_zero)

random_SK_ID_CURR_zero=np.random.choice(SK_ID_CURR_zero,num_one,replace=False)

SK_ID_CURR_to_consider=list(random_SK_ID_CURR_zero)+SK_ID_CURR_one
np.random.shuffle(SK_ID_CURR_to_consider)

#upate new dataset with equal number of zero and one

application_train=application_train[application_train["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]
bureau=bureau[bureau["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]
SK_ID_BUREAU_to_consider=bureau["SK_ID_BUREAU"][bureau["SK_ID_CURR"].\
                                isin(pd.Series(SK_ID_CURR_to_consider))]
bureau_balance=bureau_balance[bureau_balance["SK_ID_BUREAU"].\
                                    isin(SK_ID_BUREAU_to_consider)]
previous_application=previous_application[previous_application["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]
POH_cash_balance=POH_cash_balance[POH_cash_balance["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]
credit_card_balance=credit_card_balance[credit_card_balance["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]
installments_payment=installments_payment[installments_payment["SK_ID_CURR"].\
                                    isin(pd.Series(SK_ID_CURR_to_consider))]


#recheck that each file contain same set of SK_ID_CURR

uni_SK_ID_CURR=np.unique(SK_ID_CURR_to_consider)
a=np.unique(application_train["SK_ID_CURR"].values)
b=np.unique(bureau["SK_ID_CURR"].values)
c=np.unique(previous_application["SK_ID_CURR"].values)
d=np.unique(POH_cash_balance["SK_ID_CURR"].values)
e=np.unique(credit_card_balance["SK_ID_CURR"].values)
f=np.unique(installments_payment["SK_ID_CURR"].values)

#MERGE TABLE: Aggregation

#create SK_ID_CURR to index translation
SK_ID_CURR_list=list(SK_ID_CURR_to_consider)
index_list=list(np.arange(0,len(SK_ID_CURR_to_consider)))


#makes bureau_balance array
bureau_balance_array=np.zeros((bureau.shape[0],4))
current_ID=1234567890
current_index=-1
for each_ID,each_status in zip(bureau_balance["SK_ID_BUREAU"],bureau_balance["STATUS"]):
    if each_ID!=current_ID:
        current_ID=each_ID
        current_index+=1
        bureau_balance_array[current_index,0]=current_ID
    if each_status=='C':
        bureau_balance_array[current_index,1]+=1
        
    if each_status=='0':
        bureau_balance_array[current_index,2]+=1
    else:
        bureau_balance_array[current_index,3]+=1


remove_index=list(np.squeeze(np.argwhere(bureau_balance_array[:,0]==0)))
bureau_balance_array=bureau_balance_array[:remove_index[0],:]

#make bureau array
bureau_array=np.zeros((bureau.shape[0],19))

bureau_array[:,[0,1]]=bureau.values[:,[0,1]]
bureau_array[:,[4,5,6,7,8,9,10,11,12,13]]=bureau.values[:,[4,5,6,7,8,9,10,11,12,13]]
bureau_array[:,[17,18]]=bureau.values[:,[15,16]]

credit_active=bureau.values[:,2]=="Closed"
credit_active_closed=np.zeros(credit_active.shape)
credit_active_active=np.zeros(credit_active.shape)
credit_active_closed[credit_active==True]=1
credit_active_active[credit_active==False]=1
bureau_array[:,2]=credit_active_closed
bureau_array[:,3]=credit_active_active

credit_type_consumer_credit=bureau.values[:,14]=="Consumer credit"
credit_type_consumer_credit=credit_type_consumer_credit.astype(int)
credit_type_credit_card=bureau.values[:,14]=="Credit card"
credit_type_credit_card=credit_type_credit_card.astype(int)
summ=credit_type_credit_card+credit_type_consumer_credit
credit_type_other=np.zeros(credit_type_consumer_credit.shape)
credit_type_other[summ==0]=1
bureau_array[:,14]=credit_type_consumer_credit
bureau_array[:,15]=credit_type_credit_card
bureau_array[:,16]=credit_type_other

bureau_array[np.isnan(bureau_array)==True]=0

#make application an array
a=np.zeros((application_train.values.shape[0],128))

a[:,[0,1]]=application_train.values[:,[0,1]]

cash_loan=application_train.values[:,2]=="Cash loans"
cash_loan=cash_loan.astype(int)
reloving_loan=np.zeros(cash_loan.shape)
reloving_loan[cash_loan==0]=1
a[:,2]=cash_loan
a[:,3]=reloving_loan

male=application_train.values[:,3]=="M"
female=application_train.values[:,3]=="F"
a[:,4]=male.astype(int)
a[:,5]=female.astype(int)

car=application_train.values[:,4]=="Y"
no_car=application_train.values[:,4]=="N"
a[:,6]=car.astype(int)
a[:,7]=no_car.astype(int)

realty=application_train.values[:,5]=="Y"
no_realty=application_train.values[:,5]=="N"
a[:,7]=realty.astype(int)
a[:,8]=no_realty.astype(int)

a[:,[9,10,11,12,13]]=application_train.values[:,[6,7,8,9,10]]

unaccompanied=application_train.values[:,11]=="Unaccompanied"
unaccompanied=unaccompanied.astype(int)
Family=application_train.values[:,11]=="Family"
Family=Family.astype(int)
other=np.zeros(unaccompanied.shape)
other[Family+unaccompanied==0]=1
a[:,14]=unaccompanied
a[:,15]=Family
a[:,16]=other

working=application_train.values[:,12]=="Working"
working=working.astype(int)
commercial=application_train.values[:,12]=="Commercial"
commercial=commercial.astype(int)
other=np.zeros(working.shape)
other[working+commercial==0]=1
a[:,17]=working
a[:,18]=commercial
a[:,19]=other

sec=application_train.values[:,13]=="Secondary / secondary special"
sec=sec.astype(int)
higher=application_train.values[:,13]=="Higher education"
higher=higher.astype(int)
other=np.zeros(working.shape)
other[higher+sec==0]=1
a[:,20]=sec
a[:,21]=higher
a[:,22]=other

maried=application_train.values[:,14]=="Married"
maried=maried.astype(int)
single=application_train.values[:,14]=="Single / not married"
single=single.astype(int)
other=np.zeros(working.shape)
other[single+maried==0]=1
a[:,23]=maried
a[:,24]=single
a[:,25]=other

house=application_train.values[:,15]=="House / apartment"
house=house.astype(int)
parent=application_train.values[:,15]=="With parents"
parent=parent.astype(int)
other=np.zeros(working.shape)
other[house+parent==0]=1
a[:,26]=house
a[:,27]=parent
a[:,28]=other

a[:,[28,29,30,31,32,33,34,35,36,37,38,39]]=application_train.values[:,[16,17,18,19,20,21,22,23,24,25,26,27]]
a[:,[40,41,42]]=application_train.values[:,[29,30,31]]

tuesday=application_train.values[:,32]=="TUESDAY"
tuesday=tuesday.astype(int)
wednesday=application_train.values[:,32]=="WEDNESDAY"
wednesday=wednesday.astype(int)
other=np.zeros(working.shape)
other[tuesday+wednesday==0]=1
a[:,43]=tuesday
a[:,44]=wednesday
a[:,45]=other

a[:,[46,47,48,49,50,51,52]]=application_train.values[:,[33,34,35,36,37,38,39]]

index=list(np.arange(41,121))
index.remove(86)
index.remove(87)
index.remove(89)
index.remove(90)

index_for_a=list(np.array(index)+7)
a[:,index_for_a]=application_train.values[:,index]

application_train_array=np.copy(a)


#make POS cash column an array
POS_cash_balance_array=np.zeros((POH_cash_balance.values.shape[0],POH_cash_balance.values.shape[1]+2))

POS_cash_balance_array[:,[0,1,2,3,4]]=POH_cash_balance.values[:,[0,1,2,3,4]]

active=POH_cash_balance.values[:,5]=="Active"
active=active.astype(int)
completed=POH_cash_balance.values[:,5]=="Completed"
completed=completed.astype(int)
other=np.zeros(completed.shape)
other[active+completed==0]=1
POS_cash_balance_array[:,5]=active
POS_cash_balance_array[:,6]=completed
POS_cash_balance_array[:,7]=other

POS_cash_balance_array[:,[8,9]]=POH_cash_balance.values[:,[6,7]]

#make installment payment an array
installments_payment_array=installments_payment.values

#make credit card balance an array
credit_card_balance_array=np.zeros((credit_card_balance.shape[0],credit_card_balance.shape[1]+2))

index=list(np.arange(0,20))
credit_card_balance_array[:,index]=credit_card_balance.values[:,index]

active=credit_card_balance.values[:,20]=="Active"
active=active.astype(int)
completed=credit_card_balance.values[:,20]=="Completed"
completed=completed.astype(int)
other=np.zeros(completed.shape)
other[active+completed==0]=1
credit_card_balance_array[:,20]=active
credit_card_balance_array[:,21]=completed
credit_card_balance_array[:,22]=other

credit_card_balance_array[:,[23,24]]=credit_card_balance.values[:,[21,22]]

#make previous an array
previous_application_array=np.zeros((previous_application.shape[0],70))


previous_application_array[:,[0,1]]=previous_application.values[:,[0,1]]


cash=previous_application.values[:,2]=="Cash loans"
cash=cash.astype(int)
consumer=previous_application.values[:,2]=="Comsumer loans"
consumer=consumer.astype(int)
other=np.zeros(cash.shape)
other[cash+consumer==0]=1
previous_application_array[:,2]=cash
previous_application_array[:,3]=consumer
previous_application_array[:,4]=other


previous_application_array[:,[5,6,7,8,9]]=previous_application.values[:,[3,4,5,6,7]]


tuesday=previous_application.values[:,8]=="TUESDAY"
tuesday=tuesday.astype(int)
wednesday=previous_application.values[:,8]=="WEDNESDAY"
wednesday=wednesday.astype(int)
other=np.zeros(tuesday.shape)
other[tuesday+wednesday==0]=1
previous_application_array[:,11]=tuesday
previous_application_array[:,12]=wednesday
previous_application_array[:,13]=other

previous_application_array[:,14]=previous_application.values[:,9]

yes=previous_application.values[:,10]=="Y"
yes=yes.astype(int)
no=previous_application.values[:,10]=="N"
no=no.astype(int)
other=np.zeros(yes.shape)
other[yes+no==0]=1
previous_application_array[:,15]=yes
previous_application_array[:,16]=no
previous_application_array[:,17]=other

previous_application_array[:,[18,19,20,21]]=previous_application.values[:,[11,12,13,14]]


XAP=previous_application.values[:,15]=="XAP"
XAP=XAP.astype(int)
XNA=previous_application.values[:,15]=="XNA"
XNA=XNA.astype(int)
other=np.zeros(XAP.shape)
other[XAP+XNA==0]=1
previous_application_array[:,22]=XAP
previous_application_array[:,23]=XNA
previous_application_array[:,24]=other

cancel=previous_application.values[:,16]=="Canceled"
cancel=cancel.astype(int)
approved=previous_application.values[:,16]=="Approved"
approved=approved.astype(int)
other=np.zeros(cancel.shape)
other[cancel+approved==0]=1
previous_application_array[:,25]=cancel
previous_application_array[:,26]=approved
previous_application_array[:,27]=other


previous_application_array[:,28]=previous_application.values[:,17]


cash=previous_application.values[:,18]=="Cash through the bank"
cash=cash.astype(int)
XNA=previous_application.values[:,18]=="XNA"
XNA=XNA.astype(int)
other=np.zeros(cash.shape)
other[cash+XNA==0]=1
previous_application_array[:,29]=cash
previous_application_array[:,30]=XNA
previous_application_array[:,31]=other

XAP=previous_application.values[:,19]=="XAP"
XAP=XAP.astype(int)
HC=previous_application.values[:,19]=="HC"
HC=HC.astype(int)
other=np.zeros(XAP.shape)
other[XAP+HC==0]=1
previous_application_array[:,32]=XAP
previous_application_array[:,33]=HC
previous_application_array[:,34]=other

un=previous_application.values[:,20]=="Unaccompanied"
un=un.astype(int)
Family=previous_application.values[:,20]=="Family"
Family=Family.astype(int)
other=np.zeros(un.shape)
other[un+Family==0]=1
previous_application_array[:,35]=un
previous_application_array[:,36]=Family
previous_application_array[:,37]=other


Repeater=previous_application.values[:,21]=="Repeater"
Repeater=Repeater.astype(int)
New=previous_application.values[:,21]=="New"
New=New.astype(int)
other=np.zeros(Repeater.shape)
other[Repeater+New==0]=1
previous_application_array[:,38]=Repeater
previous_application_array[:,39]=New
previous_application_array[:,40]=other

XNA=previous_application.values[:,22]=="XNA"
XNA=XNA.astype(int)
Mobile=previous_application.values[:,22]=="Mobile"
Mobile=Mobile.astype(int)
other=np.zeros(XNA.shape)
other[XNA+Mobile==0]=1
previous_application_array[:,41]=XNA
previous_application_array[:,42]=Mobile
previous_application_array[:,43]=other

POS=previous_application.values[:,23]=="POS"
POS=POS.astype(int)
Cash=previous_application.values[:,23]=="Cash"
Cash=Cash.astype(int)
other=np.zeros(POS.shape)
other[POS+Cash==0]=1
previous_application_array[:,44]=POS
previous_application_array[:,45]=Cash
previous_application_array[:,46]=other


XNA=previous_application.values[:,24]=="XNA"
XNA=XNA.astype(int)
xsell=previous_application.values[:,24]=="x-sell"
xsell=xsell.astype(int)
other=np.zeros(XNA.shape)
other[XNA+xsell==0]=1
previous_application_array[:,47]=XNA
previous_application_array[:,48]=xsell
previous_application_array[:,49]=other

credit=previous_application.values[:,25]=="Credit and cash offices"
credit=credit.astype(int)
country=previous_application.values[:,25]=="Country-wide"
country=country.astype(int)
other=np.zeros(credit.shape)
other[credit+country==0]=1
previous_application_array[:,50]=credit
previous_application_array[:,51]=country
previous_application_array[:,52]=other

previous_application_array[:,53]=previous_application.values[:,26]

XNA=previous_application.values[:,27]=="XNA"
XNA=XNA.astype(int)
consumer=previous_application.values[:,27]=="Consumer electronics"
consumer=consumer.astype(int)
other=np.zeros(XNA.shape)
other[XNA+consumer==0]=1
previous_application_array[:,54]=XNA
previous_application_array[:,55]=consumer
previous_application_array[:,56]=other

previous_application_array[:,57]=previous_application.values[:,28]

XNA=previous_application.values[:,29]=="XNA"
XNA=XNA.astype(int)
middle=previous_application.values[:,29]=="middle"
middle=middle.astype(int)
other=np.zeros(XNA.shape)
other[XNA+middle==0]=1
previous_application_array[:,58]=XNA
previous_application_array[:,59]=middle
previous_application_array[:,60]=other

cash=previous_application.values[:,30]=="Cash"
cash=cash.astype(int)
POS=previous_application.values[:,30]=="POS household with interest"
POS=POS.astype(int)
other=np.zeros(cash.shape)
other[cash+POS==0]=1
previous_application_array[:,61]=cash
previous_application_array[:,62]=POS
previous_application_array[:,63]=other

previous_application_array[:,[64,65,66,67,68,69]]=previous_application.values[:,[31,32,33,34,35,36]]

#all array list
all_table=[bureau_balance_array,bureau_array,application_train_array,\
           POS_cash_balance_array,installments_payment_array,\
           credit_card_balance_array,previous_application_array]


#SK_ID_CURR(value) to SK_BUREAU_ID(key)
dic={}
for each_UD_CURR,each_BUREAU_ID in zip(bureau_array[:,0],bureau_array[:,1]):
    dic[each_BUREAU_ID]=each_UD_CURR
    
SK_ID_CURR_column=[]
for each_BUREAU_ID in bureau_balance_array[:,0]:
    SK_ID_CURR_column.append(dic[each_BUREAU_ID])
SK_ID_CURR_column=np.array(SK_ID_CURR_column)

bureau_balance_array=np.concatenate((bureau_balance_array,SK_ID_CURR_column[:,np.newaxis]),axis=1)
#merge all table
one_table=[]
i=0
for each_SK_ID_CURR in SK_ID_CURR_list:
    if i%10==0:
        print(i/len(SK_ID_CURR_list)," of 1")
    a=np.squeeze(application_train_array[application_train_array[:,0]==each_SK_ID_CURR,:])   
    b=previous_application_array[previous_application_array[:,1]==each_SK_ID_CURR,3:].sum(axis=0)    
    c=credit_card_balance_array[credit_card_balance_array[:,1]==each_SK_ID_CURR,3:].sum(axis=0)   
    d=installments_payment_array[installments_payment_array[:,1]==each_SK_ID_CURR,3:].sum(axis=0)    
    e=POS_cash_balance_array[POS_cash_balance_array[:,1]==each_SK_ID_CURR,3:].sum(axis=0)    
    f=bureau_array[bureau_array[:,0]==each_SK_ID_CURR,3:].sum(axis=0)    
    g=bureau_balance_array[bureau_balance_array[:,-1]==each_SK_ID_CURR,1:-1].sum(axis=0)
    one_row=[list(a)+list(b)+list(c)+list(d)+list(e)+list(f)+list(g)]
    one_table.append(np.array(one_row))
    
    i+=1
    
one_table=np.array(one_table)
one_table=np.squeeze(one_table)
one_table[np.isnan(one_table)==True]=0

X=one_table[:,2:]
Y=one_table[:,1]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.fit_transform(X_test)

classifier3=LogisticRegression()
classifier3.fit(scaled_X_train,Y_train)

Y_predicted_train=classifier3.predict(scaled_X_train)
Y_predicted_test=classifier3.predict(scaled_X_test)

check_train=Y_predicted_train==Y_train
check_train=check_train.astype(int)
print("training acc : ",check_train.sum()/len(check_train))

check_test=Y_predicted_test==Y_test
check_test=check_test.astype(int)
print("test acc : ",check_test.sum()/len(check_test))

prob_Y_test = classifier3.predict_proba(scaled_X_test)
prob_Y_test=prob_Y_test[:,1] #prob of result 1 only

from sklearn import metrics
fpr3, tpr3, thresholds = metrics.roc_curve(Y_test, prob_Y_test)

plt.figure()
plt.title("ROC curve for model 3")
plt.plot(fpr3,tpr3)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

find_bias(np.concatenate((Y_train,Y_test)),np.concatenate((Y_predicted_train,Y_predicted_test)))

##############################################################################
##############################################################################
##############################################################################
#MODEL 4: remove unimportance feature

from sklearn.ensemble import ExtraTreesClassifier

#split train|test|validation
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.2)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.fit_transform(X_test)
scaled_X_val=scaler.fit_transform(X_val)


# feature extraction
model=ExtraTreesClassifier(n_estimators=10)
model.fit(scaled_X_train,Y_train)

a=model.feature_importances_
a[np.isnan(a)]=0

sorted_index=list(np.argsort(a)) #small to big
sorted_index.reverse() #big to small
b=a[sorted_index]
''' 
num_to_remove=1
acc_train_list=[]
acc_val_list=[]
for num_to_remove in range(1,scaled_X_val.shape[1]):

    important_index=sorted_index[:-num_to_remove]
    
    
    new_X_train=scaled_X_train[:,important_index]
    new_X_val=scaled_X_val[:,important_index]
    
    classifier4=LogisticRegression()
    classifier4.fit(new_X_train,Y_train)
    
    Y_predicted_train=classifier4.predict(new_X_train)
    Y_predicted_val=classifier4.predict(new_X_val)
    
    check_train=Y_predicted_train==Y_train
    check_train=check_train.astype(int)
    print("remove: ",num_to_remove)
    print("training acc : ",check_train.sum()/len(check_train))
    
    check_val=Y_predicted_val==Y_val
    check_val=check_val.astype(int)
    print("val acc : ",check_val.sum()/len(check_val))

    acc_train_list.append(check_train.sum()/len(check_train))
    acc_val_list.append(check_val.sum()/len(check_val))
    


plt.figure()
plt.title("training accuracy for different number of features consdiered")
plt.plot(np.arange(0,len(acc_train_list)),acc_train_list,label="train acc")
plt.plot(np.arange(0,len(acc_val_list)),acc_val_list,label="val acc")
plt.legend()
plt.ylabel("accuracy")
plt.xlabel("number of feature excluded")
plt.show()

#sort index
feature_reduction=np.arange(1,scaled_X_val.shape[1])

train_val_acc_gap=np.array(acc_train_list)-np.array(acc_val_list)
train_val_acc_gap[train_val_acc_gap<0]=10000

sorted_index_overfitting=np.argsort(train_val_acc_gap) #least overfit to highest overfit


sorted_index_val_accuarcy=np.argsort(acc_val_list)
sorted_index_val_accuarcy=list(sorted_index_val_accuarcy)
sorted_index_val_accuarcy.reverse()
'''
#select based on least overfitting
num_reduction=102

important_index=sorted_index[:-num_reduction]


new_X_train=scaled_X_train[:,important_index]
new_X_val=scaled_X_val[:,important_index]
new_X_test=scaled_X_test[:,important_index]

classifier4=LogisticRegression()
classifier4.fit(new_X_train,Y_train)

Y_predicted_train=classifier4.predict(new_X_train)
Y_predicted_val=classifier4.predict(new_X_val)
Y_predicted_test=classifier4.predict(new_X_test)

check_train=Y_predicted_train==Y_train
check_train=check_train.astype(int)
print("training acc : ",check_train.sum()/len(check_train))

check_val=Y_predicted_val==Y_val
check_val=check_val.astype(int)
print("val acc : ",check_val.sum()/len(check_val))

check_test=Y_predicted_test==Y_test
check_test=check_test.astype(int)
print("test acc : ",check_test.sum()/len(check_test))

find_bias(np.concatenate((Y_train,Y_test,Y_val)),np.concatenate((Y_predicted_train,Y_predicted_test,Y_predicted_val)))
print("ROCCCCC MODEL 4!!!!!!!!!")

prob_Y_test = classifier4.predict_proba(new_X_test)
prob_Y_test=prob_Y_test[:,1] #prob of result 1 only

from sklearn import metrics
fpr4, tpr4, thresholds = metrics.roc_curve(Y_test, prob_Y_test)

plt.figure()
plt.title("ROC curve for model 4")
plt.plot(fpr4,tpr4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


##############################################################################
##############################################################################
##############################################################################
#roc comparison
plt.figure()
plt.title("ROC comparison")
plt.plot(fpr1,tpr1,label="model1")
plt.plot(fpr2,tpr2,label="model2")
plt.plot(fpr3,tpr3,label="model3")
plt.plot(fpr4,tpr4,label="model4")
plt.legend()
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

#bias comparison
plt.figure()
plt.title("MODEL 1")
plt.bar(0,99.88,label="class 1 predicted as class 0")
plt.bar(1,0.024,label="class 0 predicted as class 1")
plt.legend()
plt.show()

plt.figure()
plt.title("MODEL 2")
plt.bar(0,33.76,label="class 1 predicted as class 0")
plt.bar(1,32.34,label="class 0 predicted as class 1")
plt.legend()
plt.show()

#accuracy comaprison
plt.figure()
plt.title("training set accuracy")
plt.bar(0,0.67019,label="MODEL 2")
plt.bar(1,0.69179,label="MODEL 3")
plt.ylabel("accuracy")
plt.ylim(ymin=0.6)
plt.legend()
plt.show()

plt.figure()
plt.title("test set accuracy")
plt.bar(0,0.666666,label="MODEL 2")
plt.bar(1,0.6864,label="MODEL 3")
plt.ylim(ymin=0.6)
plt.ylabel("accuracy")
plt.legend()
plt.show()




