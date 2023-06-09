# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

class RationDetails(models.Model):
    ration_card = models.CharField(max_length=15)
    received_date = models.DateField()

    class Meta:
        db_table = 'ration_details'

class User(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20, blank=True, null=True)
    phone_number = models.IntegerField()
    ration_card = models.CharField(primary_key=True, max_length=15)
    hint_question = models.TextField()
    hint_answer = models.TextField()
    password = models.TextField()

    class Meta:
        db_table = 'user'
