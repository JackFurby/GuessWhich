from __future__ import absolute_import

import os
import sys
sys.path.append('..')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()

from django.conf import settings
from amt.utils import log_to_terminal

import amt.constants as constants
import torch
#import PyTorchHelpers
import pika
import time
import yaml
import json
import traceback
import signal
import requests
import atexit
from chatbot.visDialModel import VisDialModel as VisDialModel

VisDialATorchModel = VisDialModel()

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='sl_chatbot_queue', durable=True)


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        body['history'] = body['history'].split("||||")
        caption = body['history'].pop(0)  # First item in history is image caption
        dialog = []
        # seperate question and answer history
        for i in body['history']:
            i = i.split("~~~~")
            dialog.append({"question": i[0], "answer": i[1]})

        history = {"caption": caption, "dialog": dialog}

        # get the image url to be process by the agent
        image_id = body['image_path'].split("/")[-1]
        image_url = constants.POOL_IMAGES_URL + image_id

        result = VisDialATorchModel.abot(
            image_url,
            history,
            body['input_question']
        )
        result['input_image'] = body['image_path']
        result['question'] = str(result['question'])
        result['answer'] = str(result['answer'])
        result['history'] = result['history'].replace("<START>", "")
        result['history'] = result['history'].replace("<END>", "")

        # Store the result['predicted_fc7'] in the database after each round
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception:
        print(str(traceback.print_exc()))


channel.basic_consume('sl_chatbot_queue', callback)

channel.start_consuming()
