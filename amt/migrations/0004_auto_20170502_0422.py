# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2017-05-02 04:22
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('amt', '0003_feedback'),
    ]

    operations = [
        migrations.AlterField(
            model_name='feedback',
            name='bot',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='gameround',
            name='bot',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='imageranking',
            name='bot',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
