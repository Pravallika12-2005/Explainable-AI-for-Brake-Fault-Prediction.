from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'



class DT(models.Model):
    accuracy = models.FloatField(max_length=100)  # Store accuracy as a floating-point number
    Precession = models.FloatField(max_length=100)  
    recall = models.FloatField(max_length=100)  
    f1_score=models.FloatField(max_length=100)
    Name = models.TextField(max_length = 100)

    
    class Meta:
        db_table = 'DT_algo'



class RF(models.Model):
    accuracy = models.FloatField(max_length=100)  # Store accuracy as a floating-point number
    Precession = models.FloatField(max_length=100)  
    recall = models.FloatField(max_length=100)  
    f1_score=models.FloatField(max_length=100)
    Name = models.TextField(max_length = 100)

    
    class Meta:
        db_table = 'RF_algo'



class LR(models.Model):
    accuracy = models.FloatField(max_length=100)  # Store accuracy as a floating-point number
    Precession = models.FloatField(max_length=100)  
    recall = models.FloatField(max_length=100)  
    f1_score=models.FloatField(max_length=100)
    Name = models.TextField(max_length = 100)

    
    class Meta:
        db_table = 'LR_algo'

class GBoost(models.Model):
    accuracy = models.FloatField(max_length=100)  # Store accuracy as a floating-point number
    Precession = models.FloatField(max_length=100)  
    recall = models.FloatField(max_length=100)  
    f1_score=models.FloatField(max_length=100)
    Name = models.TextField(max_length = 100)

    
    class Meta:
        db_table = 'GBoost_algo'