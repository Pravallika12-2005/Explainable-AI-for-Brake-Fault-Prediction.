from django.db import models
from mainapp.models import *

# Create your models here.
class Feedback(models.Model):
    Feed_id = models.AutoField(primary_key=True)
    Rating=models.CharField(max_length=100,null=True)
    Review=models.CharField(max_length=225,null=True)
    Sentiment=models.CharField(max_length=100,null=True)
    Reviewer=models.ForeignKey(User, on_delete=models.CASCADE,null=True)
    datetime=models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'feedback_details'
        
class Dataset(models.Model):
   Data_id = models.AutoField(primary_key=True)
   Image = models.ImageField(upload_to='media/') 
   class Meta:
        db_table = "upload" 


        

# class HealthData(models.Model):
#     heart_rate = models.IntegerField()
#     pulse = models.IntegerField()
#     bp_sys = models.IntegerField()
#     bp_dia = models.IntegerField()
#     respiratory_rate = models.IntegerField()
#     oxygen_saturation = models.IntegerField()
#     respiratory_imbalance = models.TextField()

#     def __str__(self):
#         return f"Health Data (HR: {self.heart_rate}, BP: {self.bp_sys}/{self.bp_dia})"
