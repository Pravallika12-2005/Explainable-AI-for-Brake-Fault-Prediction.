"""
URL configuration for EVs Price Analysis project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home') 
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from mainapp import views as main_views
from userapp import views as user_views
from adminapp import views as admin_views
urlpatterns = [
    path('admin/', admin.site.urls),
    # Main
    path('',main_views.Home, name ='home'),
    path('contact',main_views.Contact, name ='contact'),
    path('register',main_views.UserRegister, name ='register'),
    path('admin',main_views.AdminLogin, name ='admin'),
    path('login/',main_views.UserLogin, name ='login'),
    path('about', main_views.about, name = 'about' ),
    # path('otpverify',main_views.otpverify,name ='otpverify'),
    path('forgotpwd',main_views.forgotpwd,name='forgotpwd'),
    
    # User
    path('userdashboard',user_views.userdashboard,name='userdashboard'),
    path('profile', user_views.profile, name = 'profile' ),
    path('userlogout', user_views.userlogout, name = 'userlogout'),
    # path('userdetect', user_views.userdetect, name = 'userdetect'),
    path('userfeedbacks',user_views.userfeedbacks,name='userfeedbacks'),
    path('user_Gradiant_Boost',user_views.user_Gradiant_Boost,name='user_Gradiant_Boost'),

    
    
    #admin
    path('admin-dashboard', admin_views.admindashboard, name = 'admindashboard'),
    path('pending-users', admin_views.pendingusers, name = 'pendingusers'),
    path('manage-users', admin_views.manageusers, name = 'manageusers'),
    path('admin-graph',admin_views.admin_graph, name='admingraph'),
    path('admin-feedback',admin_views.adminfeedback, name='adminfeedback'),
    path('adminsentiment',admin_views.adminsentiment, name='adminsentiment'),
    path('user-graph',admin_views.usergraph, name='usergraph'),
    # path('admin-traintest',admin_views.admintraintest, name='admintraintest'),
    # path('admin-cnn',admin_views.admincnn, name='admincnn'),
    # path('traintest_btn', admin_views.traintest_btn, name='traintest_btn'),
    # path('cnn_btn', admin_views.cnn_btn, name='cnn_btn'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('delete-user/<int:id>', admin_views.delete_user, name = 'delete_user'),
    path('admin-datasetupload',admin_views.admin_datasetupload, name='admin_datasetupload'),
    path('admin_dataset_btn', admin_views.admin_dataset_btn, name ='admin_dataset_btn'),
   

    path('RF_alg/',admin_views.RF_alg,name='RF_alg'),
    path('DT_alg/',admin_views.DT_alg,name='DT_alg'),
    path('LR_alg/',admin_views.LR_alg,name='LR_alg'), 
    path('GBoost_alg/',admin_views.GBoost_alg,name='GBoost_alg'),
    path('RF_btn/',admin_views.RF_btn,name='RF_btn'),
    path('DT_btn/',admin_views.DT_btn,name='DT_btn'),
    path('LR_btn/',admin_views.LR_btn,name='LR_btn'),
    path('GBoost_btn/',admin_views.GBoost_btn,name='GBoost_btn'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


