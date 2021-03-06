from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.db.models import Sum

from .utils import (
    log_to_terminal,
    get_pool_images,
    get_url_of_image,
    set_qualification_to_worker
)

from .models import Feedback, ImageRanking

import amt.constants as constants

import sys
import uuid
import os
import traceback
import random
import urllib
import redis
import json


r = redis.Redis(host='localhost', port=6379, db=0)


class PoolImage:
    """
    Class to store the details related to a particular pool
    """

    def __init__(self, image_path, score, img_id, rank):
        self.image_path = image_path
        self.score = score
        self.img_id = img_id
        self.rank = rank


def home(request, template_name="amt/index.html"):
    """
    Method called when the game starts
    """

    worker_id = request.GET.get('workerId', "default")

    if worker_id == "default":
        # default is used for the debug mode
        disabled = True
    if worker_id != "default":
        disabled = False
        try:
            # Set the qualification so that worker cannot do the HIT again
            set_qualification_to_worker(
                worker_id=worker_id, qualification_value=1)
            print("Success: Setting Qualification for worker ", worker_id)
        except Exception as e:
            print("Error: Cannot Set Qualification for worker ", worker_id)

    '''
    Possible values of level:
    - easy
    - medium
    - hard
    '''
    level = request.GET.get("level", "medium")
    hitId = request.GET.get('hitId')
    assignmentId = request.GET.get('assignmentId')
    turkSubmitTo = request.GET.get('turkSubmitTo')
    bot = request.GET.get('bot')#

    # generate next hit ID
    if hitId is None:
        args = ImageRanking.objects.exclude(hit_id="")  # remove any value which is empty
        args = args.extra(select={'hit_id': 'CAST(hit_id AS INTEGER)'}).values("hit_id")  # convert to int and only keep hit_id
        hitId = int((args.order_by('-hit_id')[0]).get("hit_id")) + 1

    # generate next assignment ID
    if assignmentId is None:
        args = ImageRanking.objects.exclude(assignment_id="")  # remove any value which is empty
        args = args.extra(select={'assignment_id': 'CAST(assignment_id AS INTEGER)'}).values("assignment_id")  # convert to int and only keep assignment_id
        assignmentId = int((args.order_by('-assignment_id')[0]).get("assignment_id")) + 1

    socketid = uuid.uuid4()

    # Fetch previous games played by this user
    prev_games_of_this_hit = ImageRanking.objects.filter(assignment_id=assignmentId).filter(worker_id=worker_id).filter(hit_id=hitId).filter(bot=bot)
    prev_game_ids = prev_games_of_this_hit.values_list('game_id', flat=True)
    prev_game_ids = [int(i) for i in prev_game_ids]

    print(ImageRanking.objects.values("assignment_id", "worker_id", "hit_id", "bot", "game_id"))

    try:
        # Compute the next GameID to show the new pool of images to play with
        next_game_id = max(prev_game_ids)
    except:
        # If exception, start from the very beginning i.e game_id=0
        next_game_id = 0

    if next_game_id == 10:
        next_game_id = 9
        # If this is the last game, show the modal to fill feedback after this
        # game
        show_feedback_modal = True
    else:
        show_feedback_modal = False

    # Get the pool details for the particular game_id
    image_pool = get_pool_images(pool_id=int(next_game_id))

    # Fetch the images of particular difficulty from the pool json data
    image_list = image_pool['pools'][level][:20]
    image_list = sorted(image_list)
    img_list = []
    for i in range(len(image_list)):
        img_path = constants.POOL_IMAGES_URL + str(image_list[i]) + ".jpg"
        img = PoolImage(img_path, 0, image_list[i], i+1)
        img_list.append(img)
    image_path_list = [constants.POOL_IMAGES_URL +
                       str(s) + ".jpg" for s in image_list][:20]
    target_image = image_pool['target']
    target_image_url = get_url_of_image(target_image)
    # Assign 0 rank to all of the images
    scores = [0] * 20
    caption = image_pool['gen_caption']

    r.set("target_{}".format(str(socketid)), target_image)

    intro_message = random.choice(constants.BOT_INTORDUCTION_MESSAGE)

    # Compute the comulative bonus for previous games that he has played before
    total_bonus_so_far = ImageRanking.objects.filter(
        assignment_id=assignmentId, worker_id=worker_id, hit_id=hitId, bot=bot).aggregate(score=Sum('score'))

    # If this is the first game for the user, set the total bonus to 0
    if total_bonus_so_far['score'] is None:
        total_bonus_so_far = 0
    else:
        total_bonus_so_far = total_bonus_so_far['score']

    return render(request, template_name, {
        "socketid": socketid,
        "bot_intro_message": intro_message,
        "img_list": img_list,
        "target_image": target_image_url,
        "target_image_id": target_image,
        "scores": scores,
        "img_id_list": json.dumps(image_list),
        "caption": caption,
        "max_rounds": constants.NUMBER_OF_ROUNDS_IN_A_GAME,
        "num_of_games_in_a_hit": constants.NUMBER_OF_GAMES_IN_A_HIT,
        "disabled": disabled,
        "total_bonus_so_far": total_bonus_so_far,
        "max_game_bonus": constants.MAX_BONUS_IN_A_GAME,
        "bonus_deduction_on_each_click": constants.BONUS_DEDUCTION_FOR_EACH_CLICK,
        "next_game_id": next_game_id,
        "bonus_for_correct_image_after_each_round": constants.BONUS_FOR_CORRECT_IMAGE_AFTER_EACH_ROUND,
        "show_feedback_modal": show_feedback_modal,
        "hitId": hitId,
        "assignmentId": assignmentId,
        "worker_id": worker_id,
    })


def feedback(request):
    """
    View to collect the feedback provided by the Mechanical Turk Workers
    """
    hitId = request.POST.get('hitId')
    assignmentId = request.POST.get('assignmentId')
    workerId = request.POST.get('workerId')
    understand_question = request.POST.get('understand_question')
    understand_image = request.POST.get('understand_image')
    fluency = request.POST.get('fluency')
    detail = request.POST.get('detail')
    accurate = request.POST.get('accurate')
    consistent = request.POST.get('consistent')
    comments = request.POST.get('comments')
    game_id = request.POST.get('game_id')
    level = request.POST.get('level')
    bot = request.POST.get('bot')
    task = request.POST.get('task')

    Feedback.objects.create(
        hit_id=hitId,
        assignment_id=assignmentId,
        worker_id=workerId,
        understand_question=understand_question,
        understand_image=understand_image,
        fluency=fluency,
        detail=detail,
        accurate=accurate,
        consistent=consistent,
        comments=comments,
        level=level,
        game_id=game_id,
        bot=bot,
        task=task
    )
    return JsonResponse({'success': True})
