from django.shortcuts import render,redirect
import urllib.request
import urllib.parse
from mainapp.models import *
from userapp.models import *
from adminapp.models import *
import random 
import ssl
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
# Create your views here.
# def sendSMS(user, otp, mobile):
#     data = urllib.parse.urlencode({
#         'username': 'Codebook',
#         'apikey': '56dbbdc9cea86b276f6c',
#         'mobile': mobile,
#         'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
#         'senderid': 'CODEBK'
#     })
#     data = data.encode('utf-8')
#     # Disable SSL certificate verification
#     context = ssl._create_unverified_context()
#     request = urllib.request.Request("https://smslogin.co/v3/api.php?")
#     f = urllib.request.urlopen(request, data,context=context)
#     return f.read()

def Home(req):
    return render(req,'main/main-home.html')
def Contact(req):
    if req.method == 'POST':
        # Handle form submission here
        full_name = req.POST.get('Name')
        phone_number = req.POST.get('Phone Number')
        email = req.POST.get('Email')
        message = req.POST.get('message')

        # For demonstration purposes, let's just display a success message
        User.objects.create(Full_name = full_name, Phone_Number = phone_number, Email=email, Message=message)
        messages.success(req, 'Your message has been submitted successfully.')
        return redirect('contact') 
    return render(req,'main/main-contact.html')
def UserRegister(req):
    if req.method == 'POST' :
        name = req.POST.get('myName')
        age = req.POST.get('myAge')
        password = req.POST.get('myPwd')
        phone = req.POST.get('myPhone')
        email = req.POST.get('myEmail')
        address = req.POST.get("address")
        image = req.FILES['image']
        print(name,age,password,phone,email,address,image)
        image = req.FILES['image']
        number = random.randint(1000,9999)
        
        print(number)
        # try:
        #     user_data = User.objects.get(Email = email)
        #     if user_data!=None:
        #         messages.warning(req, 'Email was already registered, choose another email..!')
        #         return redirect("register")
        # except:
            # sendSMS(name,number,phone)
        User.objects.create(Full_name = name, Image = image, Age = age, Password = password, Address = address, Email = email, Phone_Number = phone)
        mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {number}'
        print(mail_message)
        # send_mail("User Password", mail_message , settings.EMAIL_HOST_USER, [email])
        req.session['Email'] = email
        messages.success(req, 'Your account was created..')
            # return redirect('otpverify')
    return render(req,'main/main-register.html')
def AdminLogin(req):
    admin_name = 'admin'
    admin_pwd = 'admin'
    if req.method == 'POST':
        admin_n = req.POST.get('adminName')
        admin_p = req.POST.get('adminPwd')
        if (admin_n == admin_name and admin_p == admin_pwd):
            messages.success(req, 'You are logged in..')
            return redirect('admindashboard')
        else:
            messages.error(req, 'You are trying to loging with wrong details..')
            return redirect('admin')
    return render(req,'main/main-admin.html')
def UserLogin(req):
    if req.method == 'POST':
        u_email = req.POST.get('uemail')
        u_password = req.POST.get('upwd')
        print( u_email,u_password)
        
        user_data = User.objects.get(Email = u_email)
        print(user_data)
        if user_data.Password == u_password:
            if user_data.User_Status=='accepted':
                req.session['User_id'] = user_data.User_id
                messages.success(req, 'You are logged in..')
                user_data.No_Of_Times_Login += 1
                user_data.save()
                return redirect('userdashboard')
            elif user_data.User_Status=='pending':
                messages.info(req, 'Your Status is in pending')
                return redirect('login')
            else:
                # messages.warning(req, 'verifyOTP...!')
                req.session['Email'] = user_data.Email
                # return redirect('otpverify')
        else:
            messages.error(req, 'incorrect credentials...!')
            return redirect('login')
    return render(req,'main/main-user.html')
def about(req):
    return render(req,'main/main-about.html')

def otpverify(req):
    user_id = req.session['Email']
    user_o = User.objects.get(Email = user_id)
    print(user_o.Otp_Num,'data otp')
    if req.method == 'POST':
        user_otp = req.POST.get('otp')
        u_otp = int(user_otp)
        if u_otp == user_o.Otp_Num:
            user_o.Otp_Status = 'verified'
            user_o.save()
            messages.success(req, 'OTP verification was Success. Now you can continue to login..!')
            return redirect('home')
        else:
            messages.error(req, 'OTP verification was Faild. You entered invalid OTP..!')
            return redirect('otpverify')
    return render(req,'main/main-otpverify.html')
def forgotpwd(req):
    if req.POST:
        email=req.POST.get("uemail")
        user = User.objects.get(Email=email)
        if(not user):
            messages.success(req, 'No user found..!')
            return render("forgotPassword.html")
        else:
            newPassword = user.password
            send_mail('Password Recovery', 'The password for your site is '+ newPassword, 'rv_nair@gmail.com',
    ['rv_ks@gmail.com'], fail_silently=False) 
            messages.success(req, 'OTP verification was Success. Now you can continue to login..!')  
            return render("passwordRecovery.html")
    return render(req,'main/main-forgot-password.html')
