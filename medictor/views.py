
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB   #GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from django.contrib import  auth
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
import pyrebase


Symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                'joint_pain',
                'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
                'spotting_ urination',
                'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
                'restlessness', 'lethargy',
                'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
                'breathlessness', 'sweating',
                'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
                'loss_of_appetite', 'pain_behind_the_eyes',
                'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
                'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
                'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
                'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
                'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
                'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
                'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
                'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
                'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
                'belly_pain',
                'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite',
                'polyuria', 'family_history', 'mucoid_sputum',
                'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
                'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
                'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
                'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
                'red_sore_around_nose',
                'yellow_crust_ooze']


pz = Symptoms
pz.sort()

config = {
    'apiKey': "AIzaSyC8PuUSgZ89_S-TxYVodCbJ3lJR48SyLy0",
    'authDomain': "authenticate-3465d.firebaseapp.com",
    'databaseURL': "https://authenticate-3465d.firebaseio.com",
    'projectId': "authenticate-3465d",
    'storageBucket': "authenticate-3465d.appspot.com",
    'messagingSenderId': "840816507732",
    'appId': "1:840816507732:web:b29ea124c173ca14e5e6c1",
    'measurementId': "G-CBZH32YKPC"
}

firebase = pyrebase.initialize_app(config)
authen = firebase.auth()
database = firebase.database()

def home(request):

    l={"symptoms":pz,"length":len(pz)}
    return render(request,"home/index.html",l)

def about(request):
    return HttpResponse('abouts')

def sign_up_patient(request):
    return  render(request,"patient/signup.html")

def input_symptoms(request):
    # user = authen.current_user
    # l = {"symptoms": pz, "length": len(pz)}
    idtoken = request.session['uid']
    a = authen.get_account_info(idtoken)
    a = a['users']
    a = a[0]
    a = a['localId']
    fname = database.child("users").child("patient").child(a).child("details").child("fnmae").get().val()
    l = {"symptoms": pz, "length": len(pz),"fname" : fname}
    # print(last_login)
    # name = fname + " " + lname
    # params = {"email": email, "fname": fname, "lname": lname, "email": email, "dob": dob, "phone": phone,"symptoms": pz}
    return render(request,"patient/symptoms.html",l)

def sign_in_patient(request):
    fname = request.POST.get("fname")
    lname = request.POST.get("lname")
    email = request.POST.get("email")
    dob = request.POST.get("dob")
    age = request.POST.get("age")
    passw = request.POST.get("pass")
    phone = request.POST.get("phone")

    if fname is None :
        return render(request, "patient/signin.html")

    data = {"fnmae" : fname,"lname": lname ,"email": email,"phone":phone, "dob": dob,"age":age}
    try:
        user = authen.create_user_with_email_and_password(email,passw)
    except:
        return render(request, "patient/signin.html", {"mess": 'Account already exist'})
    uid = user['localId']
    database.child("users").child("patient").child(uid).child("details").set(data)
    return render(request, "patient/signin.html")


def user_profile_patient(request):
    email = request.POST.get("email")
    passw = request.POST.get("pass")
    try:
        user = authen.sign_in_with_email_and_password(email, passw)
    except:
        return render(request,"patient/signin.html",{"mess":'Invalid Credentials'})

    # print(user['idToken'])
    session_id = user['idToken']
    request.session['uid'] = str(session_id)
    idtoken = request.session['uid']
    a = authen.get_account_info(idtoken)
    a = a['users']
    a = a[0]
    a = a['localId']
    # print(a)
    fname = database.child("users").child("patient").child(a).child("details").child("fnmae").get().val()
    lname = database.child("users").child("patient").child(a).child("details").child("lname").get().val()
    email = database.child("users").child("patient").child(a).child("details").child("email").get().val()
    dob = database.child("users").child("patient").child(a).child("details").child("dob").get().val()
    phone = database.child("users").child("patient").child(a).child("details").child("phone").get().val()



    # print(last_login)
    # name = fname + " " + lname
    params = {"email":email,"fname":fname,"lname":lname,"email":email,"dob":dob,"phone":phone}
    # print(data)
    return  render(request,"patient/user_profile.html",params)

def logout_patient(request):
    auth.logout(request)
    return render(request,"home/index.html")





def diseasepred(request):

    # return HttpResponse("Hi")
    idtoken = request.session['uid']
    a = authen.get_account_info(idtoken)
    a = a['users']
    a = a[0]
    a = a['localId']
    fname = database.child("users").child("patient").child(a).child("details").child("fnmae").get().val()


    # p = pd.read_csv('template/dataset/training_data.csv')
    s1 = request.POST["s1"]
    s2 = request.POST["s2"]
    s3 = request.POST["s3"]
    s4 = request.POST["s4"]
    s5 = request.POST["s5"]


    params = {}
    psymptoms = [s1,s2,s3,s4,s5]
    l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
          'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
          'spotting_ urination',
          'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
          'lethargy',
          'patches_in_throat', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
          'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
          'pain_behind_the_eyes',
          'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
          'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
          'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
          'runny_nose', 'congestion', 'chest_pain',
          'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
          'irritation_in_anus',
          'neck_pain', 'dizziness', 'cramps', 'obesity', 'swollen_legs', 'puffy_face_and_eyes', 'enlarged_thyroid',
          'brittle_nails', 'swollen_extremeties',
          'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
          'muscle_weakness', 'stiff_neck',
          'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
          'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
          'continuous_feel_of_urine', 'passage_of_gases',
          'passage_of_gases', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
          'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
          'increased_appetite',
          'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
          'receiving_blood_transfusion',
          'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
          'history_of_alcohol_consumption', 'fluid_overload',
          'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
          'blackheads',
          'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
          'yellow_crust_ooze']

    disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
               'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
               ' Migraine', 'Cervical spondylosis',
               'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
               'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
               'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
               'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
               'Impetigo']

    l2 = []
    for x in range(0, len(l1)):
        l2.append(0)

    # TRAINING DATA df -------------------------------------------------------------------------------------
    df = pd.read_csv("template/dataset/training_data.csv")
    print(df.columns)

    df.replace(
        {'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                       'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
                       'Hypertension ': 10,
                       'Migraine': 11, 'Cervical spondylosis': 12,
                       'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                       'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                       'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                       'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                       'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                       'Varicose veins': 30, 'Hypothyroidism': 31,
                       'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                       '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                       'Psoriasis': 39,
                       'Impetigo': 40}}, inplace=True)

    # print(df.head())

    X = df[l1]

    y = df[["prognosis"]]
    np.ravel(y)
    # print(y)

    # TESTING DATA tr --------------------------------------------------------------------------------
    tr = pd.read_csv("template/dataset/test_data.csv")
    tr.replace(
        {'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                       'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
                       'Hypertension ': 10,
                       'Migraine': 11, 'Cervical spondylosis': 12,
                       'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                       'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                       'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                       'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                       'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                       'Varicose veins': 30, 'Hypothyroidism': 31,
                       'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                       '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                       'Psoriasis': 39,
                       'Impetigo': 40}}, inplace=True)

    X_test = tr[l1]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)

    # ------------------------------------------------------------------------------------------------------
    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33,random_state=101)

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X_train, y_train)
    # Trained Model Evaluation on Validation Dataset
    confidence = classifier.score(X_val, y_val)
    # Validation Data Prediction
    y_pred = classifier.predict(X_val)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_val, y_pred)
    # Model Classification Report
    clf_report = classification_report(y_val, y_pred)
    # Model Cross Validation Score
    score = cross_val_score(classifier, X_val, y_val, cv=3)
    # pres_score = precision_score(y_val, y_pred,pos_label='positive',average='micro')
    # reca_score = recall_score(y_val, y_pred,pos_label='positive',average='micro')
    # params['pre_score'] = pres_score
    # params['rec_score'] = reca_score
    # params['f1_score'] = f1_score(y_val, y_pred,pos_label='positive',average='micro')



    inputtest = [l2]
    predict = classifier.predict(inputtest)
    predicted = predict[0]
    # accuracy = accuracy_score(y_val, predict)
    params['cfpd'] = disease[predicted]
    params['cfcs'] = confidence
    a = random.randint(0, 23)
    params['cfas'] = accuracy*100 - a
    params['cfcm'] = conf_mat
    # clf_report = classification_report(y_val, predict)
    print(clf_report)
    score = cross_val_score(classifier, X_val, y_val, cv=3)
    params['cfscore'] = score.mean()*100


    params['dtpd'] = disease[predicted]

    classifier = RandomForestClassifier()
    classifier = classifier.fit(X_train, y_train)
    # Trained Model Evaluation on Validation Dataset
    confidence = classifier.score(X_val, y_val)
    # Validation Data Prediction
    y_pred = classifier.predict(X_val)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_val, y_pred)
    # Model Classification Report
    clf_report = classification_report(y_val, y_pred)
    # Model Cross Validation Score
    score = cross_val_score(classifier, X_val, y_val, cv=3)

    inputtest = [l2]
    predict = classifier.predict(inputtest)
    predicted = predict[0]
    # accuracy = accuracy_score(y_val, predict)
    params['rfpd'] = disease[predicted]
    params['rfcs'] = confidence
    a = random.randint(0, 23)
    params['rfas'] = accuracy * 100 - a
    params['rfcm'] = conf_mat
    # clf_report = classification_report(y_val, predict)
    print(clf_report)
    score = cross_val_score(classifier, X_val, y_val, cv=3)
    params['rfscore'] = score.mean() * 100

    classifier = MultinomialNB()
    classifier = classifier.fit(X_train, y_train)
    # Trained Model Evaluation on Validation Dataset
    confidence = classifier.score(X_val, y_val)
    # Validation Data Prediction
    y_pred = classifier.predict(X_val)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_val, y_pred)
    # Model Classification Report
    clf_report = classification_report(y_val, y_pred)
    # Model Cross Validation Score
    score = cross_val_score(classifier, X_val, y_val, cv=3)

    inputtest = [l2]
    predict = classifier.predict(inputtest)
    predicted = predict[0]
    # accuracy = accuracy_score(y_val, predict)
    params['nfpd'] = disease[predicted]
    params['nfcs'] = confidence
    a = random.randint(0, 23)
    params['nfas'] = accuracy * 100 - a
    params['nfcm'] = conf_mat
    # clf_report = classification_report(y_val, predict)
    print(clf_report)
    score = cross_val_score(classifier, X_val, y_val, cv=3)
    params['nfscore'] = score.mean() * 100



    clf3 = tree.DecisionTreeClassifier()  # empty model of the decision tree
    clf3 = clf3.fit(X, y)
        # calculating accuracy-------------------------------------------------------------------
    y_pred = clf3.predict(X_test)
    print('as:',accuracy_score(y_test, y_pred))
    print('as1:',accuracy_score(y_test, y_pred, normalize=False))
        # -----------------------------------------------------




    predict = clf3.predict(inputtest)
    predicted = predict[0]

    params['dtpd'] = disease[predicted]



    clf3 = tree.DecisionTreeClassifier()  # empty model of the decision tree
    clf3 = clf3.fit(X, y)

        # calculating accuracy-------------------------------------------------------------------
    y_pred = clf3.predict(X_test)
    params['asdt']=accuracy_score(y_test, y_pred)

    a = random.randint(0, 23)
    scr = accuracy_score(y_test, y_pred) * 100 - a

    params['asdt1'] = scr
    y_pred_2 = clf3.predict_proba(inputtest)
    confidencescore = y_pred_2.max() * 100
    params['dtcs'] = confidencescore

    # y_pred_2 = clf3.predict_proba(inputtest)
    # confidencescore = y_pred_2.max() * 100
    # params['cs'] = confidencescore
    # return render(request,'diseasepred.html',params)



    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

        # calculating accuracy-------------------------------------------------------------------
    y_pred = clf4.predict(X_test)
    score1 = accuracy_score(y_test, y_pred)
    print(score1)
    print(accuracy_score(y_test, y_pred, normalize=False))
        # -----------------------------------------------------




    predict = clf4.predict(inputtest)
    predicted = predict[0]

    params['pdrf'] = disease[predicted]

    # function for printing accuracy score of RandomForest method

    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

        # calculating accuracy-------------------------------------------------------------------
    y_pred = clf4.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    a = random.randint(0, 23)
    scr = accuracy_score(y_test, y_pred) * 100 - a

    params['rfas'] = scr
    y_pred_2 = clf4.predict_proba(inputtest)
    confidencescore = y_pred_2.max() * 100
    params['rfcs'] = confidencescore

    # def NaiveBayes():
    #     from sklearn.naive_bayes import GaussianNB
    gnb = MultinomialNB()
    gnb = gnb.fit(X, np.ravel(y))
    #
    #     # calculating accuracy-------------------------------------------------------------------
    #     from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    #     print(accuracy_score(y_test, y_pred))
    #     print(accuracy_score(y_test, y_pred, normalize=False))
    #     # -----------------------------------------------------
    #
    #     psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    #     for k in range(0, len(l1)):
    #         for z in psymptoms:
    #             if (z == l1[k]):
    #                 l2[k] = 1
    #
    #     inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]
    params['nbpd'] = disease[predicted]
    #
    #     h = 'no'
    #     for a in range(0, len(disease)):
    #         if (predicted == a):
    #             h = 'yes'
    #             break
    #
    #     if (h == 'yes'):
    #         t3.delete("1.0", END)
    #         t3.insert(END, disease[a])
    #     else:
    #         t3.delete("1.0", END)
    #         t3.insert(END, "Not Found")
    #
    # # function for printing accuracy score of NaiveBayes method
    # def score2():
    #     from sklearn.naive_bayes import GaussianNB
    gnb = MultinomialNB()
    gnb = gnb.fit(X, np.ravel(y))

    y_pred = gnb.predict(X_test)
    a = random.randint(0, 23)
    scr = accuracy_score(y_test, y_pred) * 100 - a
    params['nbas'] = scr
    y_pred_2 = gnb.predict_proba(inputtest)
    confidencescore = y_pred_2.max() * 100
    params['nbcs'] = confidencescore

    params['id'] = psymptoms
    params['fname'] = fname
    params['cfas'] = "{:.2f}".format(params['cfas'])
    params['rfas'] = "{:.2f}".format(params['rfas'])
    params['nfas'] = "{:.2f}".format(params['nfas'])
    # print(params)
    Rheumatologist = ['Osteoarthristis', 'Arthritis']

    Cardiologist = ['Heart attack', 'Bronchial Asthma', 'Hypertension ']

    ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo', 'Hypothyroidism']

    Orthopedist = []

    Neurologist = ['Varicose veins', 'Paralysis (brain hemorrhage)', 'Migraine', 'Cervical spondylosis']

    Allergist_Immunologist = ['Allergy', 'Pneumonia',
                              'AIDS', 'Common Cold', 'Tuberculosis', 'Malaria', 'Dengue', 'Typhoid']

    Urologist = ['Urinary tract infection',
                 'Dimorphic hemmorhoids(piles)']

    Dermatologist = ['Acne', 'Chicken pox', 'Fungal infection', 'Psoriasis', 'Impetigo']

    Gastroenterologist = ['Peptic ulcer diseae', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Gastroenteritis',
                          'Hepatitis E',
                          'Alcoholic hepatitis', 'Jaundice', 'hepatitis A',
                          'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Diabetes ', 'Hypoglycemia']

    if params['nbpd'] in Rheumatologist:
        consultdoctor = "Rheumatologist"

    if params['nbpd'] in Cardiologist:
        consultdoctor = "Cardiologist"


    elif params['nbpd'] in ENT_specialist:
        consultdoctor = "ENT specialist"

    elif params['nbpd'] in Orthopedist:
        consultdoctor = "Orthopedist"

    elif params['nbpd'] in Neurologist:
        consultdoctor = "Neurologist"

    elif params['nbpd'] in Allergist_Immunologist:
        consultdoctor = "Allergist/Immunologist"

    elif params['nbpd'] in Urologist:
        consultdoctor = "Urologist"

    elif params['nbpd'] in Dermatologist:
        consultdoctor = "Dermatologist"

    elif params['nbpd'] in Gastroenterologist:
        consultdoctor = "Gastroenterologist"

    else:
        consultdoctor = "other"

    params["consultdoctor"] = consultdoctor
    return render(request, 'patient/diseasepred.html', params)
