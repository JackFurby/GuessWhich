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

RLVisDialATorchModel = VisDialModel()

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='rl_chatbot_queue', durable=True)


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        body['history'] = body['history'].split("||||")

        # get the image url to be process by the agent
        image_id = body['image_path'].split("/")[-1]
        image_url = constants.POOL_IMAGES_URL + image_id

        result = RLVisDialATorchModel.abot(
            image_url,
            body['history'],
            body['input_question']
        )
        result['question'] = str(result['question'])
        result['answer'] = str(result['answer'])
        result['history'] = result['history']
        result['history'] = result['history'].replace("<START>", "")
        result['history'] = result['history'].replace("<END>", "")

        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception:
        print(str(traceback.print_exc()))


channel.basic_consume('rl_chatbot_queue', callback)

channel.start_consuming()
