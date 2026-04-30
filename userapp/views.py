from django.shortcuts import render, redirect
from mainapp.models import *
from userapp.models import Feedback
from django.contrib import messages
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


# ================= DASHBOARD =================
def userdashboard(request):
    images_count = User.objects.all().count()
    user_id = request.session["User_id"]
    user = User.objects.get(User_id=user_id)
    return render(request, 'user/user-dashboard.html', {'detect': images_count, 'la': user})


# ================= PROFILE =================
def profile(request):
    user_id = request.session["User_id"]
    user = User.objects.get(User_id=user_id)

    if request.method == 'POST':
        user.Full_name = request.POST.get('userName')
        user.Age = request.POST.get('userAge')
        user.Phone_Number = request.POST.get('userPhNum')
        user.Email = request.POST.get('userEmail')
        user.Address = request.POST.get("userAddress")

        if len(request.FILES) != 0:
            user.Image = request.FILES['profilepic']

        user.save()
        messages.success(request, 'Updated Successfully!')

    return render(request, 'user/user-profile.html', {"i": user})


# ================= LOGOUT =================
def userlogout(request):
    user_id = request.session["User_id"]
    user = User.objects.get(User_id=user_id)

    t = time.localtime()
    user.Last_Login_Time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Date = time.strftime('%Y-%m-%d')

    user.save()
    messages.info(request, 'You are logged out..')
    return redirect('login')


# ================= FEEDBACK =================
def userfeedbacks(request):
    user = User.objects.get(User_id=request.session["User_id"])

    if request.method == "POST":
        rating = request.POST.get("rating")
        review = request.POST.get("review")

        if not rating:
            messages.info(request, 'Give rating')
            return redirect('userfeedbacks')

        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)

        if score['compound'] >= 0.5:
            sentiment = 'very positive'
        elif score['compound'] > 0:
            sentiment = 'positive'
        elif score['compound'] <= -0.5:
            sentiment = 'very negative'
        elif score['compound'] < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        Feedback.objects.create(
            Rating=rating,
            Review=review,
            Sentiment=sentiment,
            Reviewer=user
        )

        messages.success(request, 'Feedback recorded')
        return redirect('userfeedbacks')

    return render(request, 'user/user-feedbacks.html')


# ================= MAIN ML FUNCTION =================
def user_Gradiant_Boost(request):
    if request.method == 'POST':

        # Load dataset
        train_df = pd.read_csv('E:/1/Break_fault_in_Heavy_Transports/filtered_aps_failure_training_set.csv', na_values=["na"])
        test_df = pd.read_csv('E:/1/Break_fault_in_Heavy_Transports/filtered_aps_failure_test_set.csv', na_values=["na"])

        train_df.columns = train_df.columns.str.strip()
        test_df.columns = test_df.columns.str.strip()

        df = pd.concat([train_df, test_df], ignore_index=True)

        df['class'] = df['class'].apply(lambda x: 1 if x == 'pos' else 0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(df.mean())

        X = df.drop(columns=['class'])
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # ========= USER INPUT =========
        aa_000 = float(request.POST.get('aa_000'))
        ag_002 = float(request.POST.get('ag_002'))
        ai_000 = float(request.POST.get('ai_000'))
        bj_000 = float(request.POST.get('bj_000'))
        cc_000 = float(request.POST.get('cc_000'))

        # Create full input
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)

        input_data['aa_000'] = aa_000
        input_data['ag_002'] = ag_002
        input_data['ai_000'] = ai_000
        input_data['bj_000'] = bj_000
        input_data['cc_000'] = cc_000

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        input_scaled = scaler.transform(input_data)

        # Load model
        model = pickle.load(open('E:/1/Break_fault_in_Heavy_Transports/gradient_boostinggg_model.pkl', 'rb'))

        # ========= PREDICTION =========
        prediction = model.predict(input_scaled)

        # ========= FAULT DETECTION =========
        likely_faulty_features = {}

        if aa_000 > 50:
            likely_faulty_features['aa_000'] = "Brake cylinder air pressure issue"

        if ag_002 > 50:
            likely_faulty_features['ag_002'] = "Brake pads wear issue"

        if ai_000 > 50:
            likely_faulty_features['ai_000'] = "ABS system issue"

        if bj_000 > 50:
            likely_faulty_features['bj_000'] = "Wheel alignment issue"

        if cc_000 > 50:
            likely_faulty_features['cc_000'] = "Control valve leakage"

        # ========= FINAL RESULT =========
        if prediction[0] == 1 or len(likely_faulty_features) > 0:
            result = "Fault Detected ❌"
        else:
            result = "No Fault ✅"

        # ========= METRICS =========
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f2 = fbeta_score(y_test, y_pred, beta=2) * 100

        return render(request, 'user/Gboost_prediction.html', {
            "result": result,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f2_score": f2,
            "likely_faulty_features": likely_faulty_features
        })

    return render(request, 'user/Gboost_prediction.html')