from django.shortcuts import render,redirect
from mainapp.models import*
from userapp.models import*
from adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.contrib import messages
from keras.models import load_model


#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('admin')
def admindashboard(req):
    all_users_count =  User.objects.all().count()
    pending_users_count = User.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User.objects.filter(User_Status = 'removed').count()
    accepted_users_count =User.objects.filter(User_Status = 'accepted').count()
    Feedbacks_users_count= Feedback.objects.all().count()
    user_uploaded_images =Dataset.objects.all().count()
    return render(req,'admin/admin-dashboard.html',{'a' : all_users_count, 'b' : pending_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e':Feedbacks_users_count, 'f':user_uploaded_images})

def pendingusers(req):
    pending = User.objects.filter(User_Status = 'Pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin-pending-users.html', { 'user' : post})

def delete_user(req, id):
    User.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('manageusers')

def accept_user(req, id):
    status_update = User.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pendingusers')

def manageusers(req):
    manage_users  = User.objects.all()
    paginator = Paginator(manage_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-manage-users.html', {"allu" : manage_users, 'user' : post})

def reject_user(req, id):
    status_update2 = User.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pendingusers')

def admin_datasetupload(req):
    return render(req,'admin/admin-upload-dataset.html')
def admin_dataset_btn(req):
    messages.success(req, 'Dataset uploaded successfully..!')
    return redirect('admin_datasetupload')







def adminfeedback(req):
    feed =Feedback.objects.all()
    return render(req,'admin/user-feedback.html', {'back':feed})

def adminsentiment(req):
    fee = Feedback.objects.all()
    return render(req,'admin/user-sentiment.html' , {'cat':fee})

def usergraph(req):
    positive = Feedback.objects.filter(Sentiment = 'positive').count()
    very_positive = Feedback.objects.filter(Sentiment = 'very positive').count()
    negative = Feedback.objects.filter(Sentiment = 'negative').count()
    very_negative = Feedback.objects.filter(Sentiment = 'very negative').count()
    neutral = Feedback.objects.filter(Sentiment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'n':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(req,'admin/user-sentiment-graph.html',context)

def RF_alg(req):
  return render(req,'admin/Random_forest.html')

def DT_alg(req):
  return render(req,'admin/Decision_tree.html')

def LR_alg(req):
  return render(req,'admin/Logistic.html')

def GBoost_alg(req):
  return render(req,'admin/GBoost.html')





def RF_btn(req):
    import pickle
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Load the saved models and scaler
    with open('rf_model_all_features.pkl', 'rb') as model_file:
        rf_model_all_features = pickle.load(model_file)

    with open('rf_model_top_features.pkl', 'rb') as model_file_top:
        rf_model_top_features = pickle.load(model_file_top)

    # Load the dataset again to ensure you have the data for making predictions (if not already available)
    train_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_training_set.csv', na_values=["na"])
    test_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_test_set.csv', na_values=["na"])

    # Combine the datasets and preprocess
    df = pd.concat([train_df, test_df], ignore_index=True)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'pos' else 0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    X = df.drop(columns=['class'])
    y = df['class']

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (same scaler used before)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Evaluating the model with all features
    y_pred_all_features = rf_model_all_features.predict(X_test_scaled)
    accuracy_all_features = accuracy_score(y_test, y_pred_all_features)*100
    precision_all_features = precision_score(y_test, y_pred_all_features)*100
    recall_all_features = recall_score(y_test, y_pred_all_features)*100
    f2_all_features = fbeta_score(y_test, y_pred_all_features, beta=2)*100

    # 2. Identifying top features based on importance scores from the model
    feature_importances = rf_model_top_features.feature_importances_
    top_features_indices = feature_importances.argsort()[-170:][::-1]  # Get indices of top 170 features

    # Get the names of the top 170 features
    top_features = X.columns[top_features_indices].tolist()

    # Debugging: Print the top features to ensure they are correctly selected
    print("Top features:", top_features)

    # Ensure that X_test has these top features
    X_test_top_features = X_test[top_features]  # Subset the test data based on top features
    print("Selected columns in X_test:", X_test_top_features.columns)  # Print selected columns in X_test

    # Now, we scale the top features based on the scaler fitted on the training set's top features
    X_train_top_features = X_train[top_features]  # Subset the training data based on top features
    X_train_top_features_scaled = scaler.fit_transform(X_train_top_features)  # Scale the top features

    # Now, apply the scaler to the top features in the test set
    X_test_top_features_scaled = scaler.transform(X_test_top_features)  # Scale the top features

    # Predict with the top features
    y_pred_top_features = rf_model_top_features.predict(X_test_top_features_scaled)

    accuracy_top_features = accuracy_score(y_test, y_pred_top_features)*100
    precision_top_features = precision_score(y_test, y_pred_top_features)*100
    recall_top_features = recall_score(y_test, y_pred_top_features)*100
    f2_top_features = fbeta_score(y_test, y_pred_top_features, beta=2)*100


    # Save results to the database
    name = "Random Forest Algorithm"
    RF.objects.create(accuracy=accuracy_top_features, Precession=precision_top_features, f1_score=f2_top_features, recall=recall_top_features, Name=name)
    data = RF.objects.last()

    # Success message
    messages.success(req, 'Algorithm executed successfully')
    return render(req, 'admin/Random_forest.html', {
        'accuracy': accuracy_top_features,
        'precision': precision_top_features,
        'recall': recall_top_features,
        'f1': f2_top_features,
        'feature1': top_features[0],
        'feature2': top_features[1],
        'Name':name
    })



def DT_btn(req):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
    from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
    import shap
    import pickle

    # Load datasets
    train_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_training_set.csv', na_values=["na"])
    test_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_test_set.csv', na_values=["na"])

    # Combine the datasets for consistent preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Convert target column ('class') to numeric, with 'pos' as 1 and 'neg' as 0
    df['class'] = df['class'].apply(lambda x: 1 if x == 'pos' else 0)

    # Convert non-numeric data to NaN (if any columns have mixed types)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values by filling NaNs with the mean of each column
    df = df.fillna(df.mean())

    # Separate features and target
    X = df.drop(columns=['class'])  # Assuming 'class' is the target column
    y = df['class']  # Encoded target variable

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for Decision Tree (not mandatory but keeping it for consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)  # Changed to DecisionTreeClassifier
    dt_model.fit(X_train_scaled, y_train)

    # Predict on test set and calculate metrics
    y_pred = dt_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f2 = fbeta_score(y_test, y_pred, beta=2)*100

    # Feature Importance using SHAP
    explainer = shap.TreeExplainer(dt_model)  # Use TreeExplainer for Decision Tree
    shap_values = explainer.shap_values(X_test_scaled)

    # For binary classification, shap_values is a list with two elements: one for each class.
    # We will take the SHAP values for the positive class (index 1).
    shap_importances = pd.DataFrame(list(zip(X.columns, abs(shap_values[1]).mean(axis=0))), columns=['feature', 'importance'])

    # Sort by importance and select top N features (you can adjust N based on your preference)
    top_features_count = 30  # Select the top 30 features, you can change this number
    top_features = shap_importances.sort_values(by='importance', ascending=False).head(top_features_count)['feature'].tolist()

    print(f"Top {top_features_count} Features based on SHAP values:", top_features)

    # Retrain the model using only the top N features
    X_train_top_features = X_train[top_features]
    X_test_top_features = X_test[top_features]

    X_train_top_features_scaled = scaler.fit_transform(X_train_top_features)  # Scale the top features
    X_test_top_features_scaled = scaler.transform(X_test_top_features)

    dt_model_top_features = DecisionTreeClassifier(random_state=42)  # Train Decision Tree with top features
    dt_model_top_features.fit(X_train_top_features_scaled, y_train)


    # Evaluate the model with top features
    y_pred_top_features = dt_model_top_features.predict(X_test_top_features_scaled)
    accuracy_top_features = accuracy_score(y_test, y_pred_top_features)*100
    precision_top_features = precision_score(y_test, y_pred_top_features)*100
    recall_top_features = recall_score(y_test, y_pred_top_features)*100
    f2_top_features = fbeta_score(y_test, y_pred_top_features, beta=2)*100

    
    # Save results to the database
    name = "Decision Tree Algorithm"
    DT.objects.create(accuracy=accuracy_top_features, Precession=precision_top_features, f1_score=f2_top_features, recall=recall_top_features, Name=name)
    data = DT.objects.last()
    
    # Success message
    messages.success(req, 'Algorithm executed successfully')
    return render(req, 'admin/Decision_tree.html', {
        'accuracy': accuracy_top_features,
        'precision': precision_top_features,
        'recall': recall_top_features,
        'f1': f2_top_features,
        'feature1': top_features[0],
        'feature2': top_features[1],
        'Name':name
    })


def LR_btn(req):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
    from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
    import shap
    import pickle

    # Load datasets
    train_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_training_set.csv', na_values=["na"])
    test_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_test_set.csv', na_values=["na"])

    # Combine the datasets for consistent preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Convert target column ('class') to numeric, with 'pos' as 1 and 'neg' as 0
    df['class'] = df['class'].apply(lambda x: 1 if x == 'pos' else 0)

    # Convert non-numeric data to NaN (if any columns have mixed types)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values by filling NaNs with the mean of each column
    df = df.fillna(df.mean())

    # Separate features and target
    X = df.drop(columns=['class'])  # Assuming 'class' is the target column
    y = df['class']  # Encoded target variable

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Logistic Regression model
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)

    # Predict on test set and calculate metrics
    y_pred = log_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f2 = fbeta_score(y_test, y_pred, beta=2)*100


    # Feature Importance using SHAP
    explainer = shap.LinearExplainer(log_model, X_train_scaled)  # Use LinearExplainer for Logistic Regression
    shap_values = explainer.shap_values(X_test_scaled)

    # Identify top features by SHAP importance
    shap_importances = pd.DataFrame(list(zip(X.columns, abs(shap_values).mean(axis=0))), columns=['feature', 'importance'])

    # Sort by importance and select top N features (you can adjust N based on your preference)
    top_features_count = 30  # Select the top 30 features, you can change this number
    top_features = shap_importances.sort_values(by='importance', ascending=False).head(top_features_count)['feature'].tolist()

    print(f"Top {top_features_count} Features based on SHAP values:", top_features)

    # Retrain the model using only the top N features
    X_train_top_features = X_train[top_features]
    X_test_top_features = X_test[top_features]

    X_train_top_features_scaled = scaler.fit_transform(X_train_top_features)  # Scale the top features
    X_test_top_features_scaled = scaler.transform(X_test_top_features)

    log_model_top_features = LogisticRegression(max_iter=1000, random_state=42)
    log_model_top_features.fit(X_train_top_features_scaled, y_train)


    # Evaluate the model with top features
    y_pred_top_features = log_model_top_features.predict(X_test_top_features_scaled)
    accuracy_top_features = accuracy_score(y_test, y_pred_top_features)*100
    precision_top_features = precision_score(y_test, y_pred_top_features)*100
    recall_top_features = recall_score(y_test, y_pred_top_features)*100
    f2_top_features = fbeta_score(y_test, y_pred_top_features, beta=2)*100


    
    # Save results to the database
    name = "Logistic Algorithm"
    LR.objects.create(accuracy=accuracy_top_features, Precession=precision_top_features, f1_score=f2_top_features, recall=recall_top_features, Name=name)
    data = LR.objects.last()
    
    # Success message
    messages.success(req, 'Algorithm executed successfully')
    return render(req, 'admin/Logistic.html', {
        'top_feature_accuracy': accuracy_top_features,
        'top_feature_precision': precision_top_features,
        'top_feature_recall': recall_top_features,
        'top_feature_f1': f2_top_features,
        'feature1': top_features[0],
        'feature2': top_features[1],
        'feature3': top_features[2],
        'feature4': top_features[3],
        'feature5': top_features[4],
        'Name': name
    })



# Django view for blending predictions
def GBoost_btn(req):
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load datasets
    train_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_training_set.csv', na_values=["na"])
    test_df = pd.read_csv('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/Dataset/aps_failure_test_set.csv', na_values=["na"])

    # Combine the datasets for consistent preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Convert target column ('class') to numeric, with 'pos' as 1 and 'neg' as 0
    df['class'] = df['class'].apply(lambda x: 1 if x == 'pos' else 0)

    # Convert non-numeric data to NaN (if any columns have mixed types)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values by filling NaNs with the mean of each column
    df = df.fillna(df.mean())

    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']

    # Load the saved models and scaler
    with open('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/gradient_boosting_model.pkl', 'rb') as model_file:
        gb_model = pickle.load(model_file)

    with open('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/gradient_boosting_top_features_model.pkl', 'rb') as top_features_model_file:
        gb_model_top_features = pickle.load(top_features_model_file)

    with open('C:/Users/system -2/Desktop/Break_fault_in_Heavy_Transports/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Ensure you are scaling the data using the same feature order as when the model was trained
    # Using the exact features that the model was trained on (which were used to train the scaler)
    X_full_ordered = X[scaler.feature_names_in_]  # This makes sure the order is correct.

    # Scale the full feature set using the scaler
    X_full_scaled = scaler.transform(X_full_ordered)

    # Predict using the full model (with all features)
    y_pred_gb_model = gb_model.predict(X_full_scaled)
    accuracy_gb_model = accuracy_score(y, y_pred_gb_model)
    precision_gb_model = precision_score(y, y_pred_gb_model)
    recall_gb_model = recall_score(y, y_pred_gb_model)
    f1_gb_model = f1_score(y, y_pred_gb_model)

    print("Gradient Boosting Model (All Features) Performance:")
    print(f"Accuracy: {accuracy_gb_model * 100:.2f}%")
    print(f"Precision: {precision_gb_model * 100:.2f}")
    print(f"Recall: {recall_gb_model * 100:.2f}")
    print(f"F1 Score: {f1_gb_model * 100:.2f}")

    top_features=['aa_000', 'bt_000', 'ay_008', 'ag_002', 'ag_006', 'am_0', 'ci_000', 'ck_000', 'cc_000', 'az_002', 'bj_000', 'ai_000', 'ay_006', 'cn_004', 'ee_005', 'dx_000', 'cn_001', 'dt_000', 'ay_009', 'ee_007', 'bs_000', 'ay_005', 'do_000', 'al_000', 'aq_000', 'dg_000', 'bc_000', 'ba_007', 'ag_001', 'cn_002', 'ay_000', 'ao_000', 'an_000', 'ag_004', 'cg_000', 'cp_000', 'bx_000', 'az_000', 'de_000', 'ah_000', 'ag_005', 'cj_000', 'ba_001', 'dv_000', 'ba_000', 'ad_000', 'ay_001', 'cs_003', 'ee_003', 'cn_003', 'ba_006', 'bh_000', 'cm_000', 'ay_002', 'aj_000', 'bk_000', 'cl_000', 'ay_007', 'ee_008', 'cs_005', 'bq_000', 'bg_000', 'dy_000', 'bi_000', 'dd_000', 'dr_000', 'di_000', 'az_001', 'ag_007', 'bn_000', 'au_000', 'bf_000', 'cn_007', 'az_004', 'az_003', 'bl_000', 'ba_009', 'cy_000', 'ag_009', 'ee_009', 'cs_004', 'ee_004', 'db_000', 'bz_000', 'br_000', 'du_000', 'as_000', 'bb_000', 'co_000', 'cz_000', 'ec_00', 'ee_001', 'dq_000', 'bp_000', 'bo_000', 'dl_000', 'ay_003', 'ca_000', 'ac_000', 'ba_003', 'dj_000', 'ay_004', 'ee_000', 'az_006', 'ba_008', 'ce_000', 'ag_003', 'ba_002', 'ba_004', 'ds_000', 'at_000', 'dk_000', 'cr_000', 'cn_000', 'be_000', 'bm_000', 'az_007', 'cs_009', 'cs_008', 'cs_006', 'ct_000', 'cn_009', 'ed_000', 'ef_000', 'cx_000', 'dz_000', 'ag_008', 'az_005', 'cb_000', 'ee_002', 'av_000', 'ap_000', 'by_000', 'cs_007', 'ab_000', 'ae_000', 'af_000', 'ag_000', 'ar_000', 'ax_000', 'ak_000', 'ch_000', 'cn_006', 'cn_005', 'cd_000', 'bd_000', 'ba_005', 'az_008', 'az_009', 'cf_000', 'bv_000', 'bu_000', 'cn_008', 'cq_000', 'dh_000', 'df_000', 'dc_000', 'da_000', 'cv_000', 'cu_000', 'cs_001', 'cs_002', 'cs_000', 'dn_000', 'dp_000', 'dm_000', 'eb_000', 'ea_000', 'ee_006', 'eg_000']

    # Select only the top features and order them correctly for the top-features model
    X_top_features_ordered = X[top_features]  # Ensure this matches the feature names used in training
    X_top_features_scaled = scaler.transform(X_top_features_ordered)

    # Predict using the top features model
    y_pred_top_features_model = gb_model_top_features.predict(X_top_features_scaled)
    accuracy_top_features_model = accuracy_score(y, y_pred_top_features_model)*100
    precision_top_features_model = precision_score(y, y_pred_top_features_model)*100
    recall_top_features_model = recall_score(y, y_pred_top_features_model)*100
    f1_top_features_model = f1_score(y, y_pred_top_features_model)*100


    # Save results to the database
    name = "GBoost Algorithm"
    GBoost.objects.create(accuracy=accuracy_top_features_model, Precession=precision_top_features_model, f1_score=f1_top_features_model, recall=recall_top_features_model, Name=name)
    data = GBoost.objects.last()
    
    # Success message
    messages.success(req, 'Algorithm executed successfully')
    return render(req, 'admin/GBoost.html', {
        'Name':name,
        'accuracy': accuracy_top_features_model,
        'precision': precision_top_features_model,
        'recall': recall_top_features_model,
        'f1': f1_top_features_model,
        'feature1': top_features[0],
        'feature2': top_features[1],
        'feature3': top_features[2],
        'feature4': top_features[3],
        'feature5': top_features[4],
    })





def admin_graph(request):
    details1 = RF.objects.last()
    rf = details1.accuracy


    details2 = DT.objects.last()
    dt = details2.accuracy

    details3 = LR.objects.last()
    lr = details3.accuracy


    details4 = GBoost.objects.last()
    gboost = details4.accuracy


    print('RF','DT','LR','GBoost')
    return render(request,'admin/admin-graph-analysis.html',{'RF':rf,'DT':dt,'LR':lr, 'GBoost':gboost })

















