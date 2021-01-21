import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from PIL import Image
import requests
from io import BytesIO
import chatbot.models.encoders.hre as hre_enc
import chatbot.models.decoders.gen as gen_dec
# from chatbot.models.questioner import Questioner
# from chatbot.models.answerer import Answerer
from utils import utilities as utils
from chatbot.dataloader import VisDialDataset
import chatbot.options as options
import random
from chatbot.prepro_ques import main as prepro_ques
import numpy as np
from sklearn.preprocessing import normalize

from nltk.tokenize import word_tokenize
import skimage.io
from skimage.transform import resize


def var_map(tensor):
    return Variable(tensor.unsqueeze(0))


class VisDialModel():
    # initialize
    def __init__(self):
        '''
        '''

        # Read the command line options and create options table to initialize dataloader
        self.opt = options.readCommandLine()

        splits = ['train', 'val', 'test']

        self.dataloader = VisDialDataset(self.opt, subsets=splits)

        # Params to transfer from dataset
        transfer = ['vocabSize', 'numOptions', 'numRounds']
        for key in transfer:
            if hasattr(self.dataloader, key):
                self.opt[key] = getattr(self.dataloader, key)

        # Seed rng for reproducibility
        random.seed(self.opt['randomSeed'])
        torch.manual_seed(self.opt['randomSeed'])
        if self.opt['useGPU']:
            torch.cuda.manual_seed_all(self.opt['randomSeed'])

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.opt['useGPU']) else "cpu")

        print("Device:", self.device)

        # -- Load Questioner and Answerer model
        self.questionerModel, loadedParams, optim_state = utils.loadModel(self.opt, 'qbot')
        self.questionerModelParams = loadedParams

        self.answererModel, loadedParams, optim_state = utils.loadModel(self.opt, 'abot')
        self.answererModelParams = loadedParams

        # -- changing savepath in checkpoints
        self.questionerModelParams['model_name'] = 'im-hist-enc-dec-questioner'
        self.answererModelParams['model_name'] = 'im-hist-enc-dec-answerer'

        # -- Print Questioner and Answerer
        print('Questioner', self.questionerModelParams['model_name'])
        print('Answerer', self.answererModelParams['model_name'])

        # -- Add flags for various configurations
        if 'hist' in self.questionerModelParams['model_name']:
            self.questionerModelParams['useHistory'] = True
        if 'hist' in self.answererModelParams['model_name']:
            self.answererModelParams['useHistory'] = True
        if 'im' in self.answererModelParams['model_name']:
            self.answererModelParams['useIm'] = True

        # -- Setup both Qbot and Abot
        print('Using models from', self.questionerModelParams['model_name'])
        print('Using models from', self.answererModelParams['model_name'])

        self.questionerModel.to(self.device)
        self.answererModel.to(self.device)

        self.questionerModel.eval()
        self.answererModel.eval()

        # load pre-trained VGG 19 (used to extract image features)
        print("Loading image feature extraction model")
        self.feat_extract_model = torchvision.models.vgg19(pretrained=True)
        self.feat_extract_model.classifier = nn.Sequential(*list(self.feat_extract_model.classifier.children())[:-3])
        self.feat_extract_model.eval()
        self.feat_extract_model.to(self.device)

    def abot(self, imgURL, history, question_asked):
        ############################################# history and question as JSON???????????
        ############################################# Do I need to change the JSON format???????

        print("history:", history)
        print("question:", question_asked)

        history = {"caption": history[0], "dialog": []}  # THIS NEEDS UPDATING TO THE PROPER FORMAT https://github.com/batra-mlp-lab/visdial-rl/blob/master/demo/hist.json
        question = {"question": question_asked}  # THIS NEEDS UPDATING TO THE PROPER FORMAT https://github.com/batra-mlp-lab/visdial-rl/blob/master/demo/ques.json

        print("history:", history)
        print("question:", question)

        # Download image from URL
        response = requests.get(imgURL)
        img = Image.open(BytesIO(response.content))

        ind_map = lambda words: np.array([self.dataloader.word2ind.get(word, self.dataloader.word2ind['UNK'])
                                  for word in words], dtype='int64')

        tokenize = lambda string: ['<START>'] + word_tokenize(string) + ['<END>']

        # Process image
        def transform(img):
            img = img.astype("float")/255
            img = resize(img, (224, 224), mode='constant')
            img[:,:,0] -= 0.485
            img[:,:,1] -= 0.456
            img[:,:,2] -= 0.406
            return img.transpose([2,0,1])

        raw_img = transform(skimage.io.imread(BytesIO(response.content)))

        # Process caption
        caption_tokens = tokenize(history['caption'])
        caption = ind_map(caption_tokens)

        # Process history
        h_question_tokens = []
        h_questions = []
        h_answer_tokens = []
        h_answers = []
        for round_idx, item in enumerate(history['dialog']):
            ans_tokens = tokenize(item['answer'])
            h_answer_tokens.append(ans_tokens)
            h_answers.append(ind_map(ans_tokens))

            ques_tokens = tokenize(item['question'])
            h_question_tokens.append(ques_tokens)
            h_questions.append(ind_map(ques_tokens))

        # Process question
        question_tokens = tokenize(question['question'])
        question = ind_map(question_tokens)

        img_tensor = var_map((torch.from_numpy(raw_img).float()).to(self.device))
        img_feats = self.feat_extract_model(img_tensor)
        _norm = torch.norm(img_feats, p=2, dim=1)
        img_feats = img_feats.div(_norm.expand_as(img_feats))

        caption_tensor = var_map(torch.from_numpy(caption).to(self.device))
        caption_lens = var_map(torch.LongTensor([len(caption)]).to(self.device))

        question_tensor = var_map(torch.from_numpy(question).to(self.device))
        question_lens = var_map(torch.LongTensor([len(question)]).to(self.device))

        hist_ans_tensors = [var_map(torch.from_numpy(ans).to(self.device)) for ans in h_answers]
        hist_ans_lens = [var_map(torch.LongTensor([len(h_ans)]).to(self.device)) for h_ans in h_answer_tokens]
        hist_ques_tensors = [var_map(torch.from_numpy(ques).to(self.device)) for ques in h_questions]
        hist_ques_lens = [var_map(torch.LongTensor([len(h_ques)]).to(self.device)) for h_ques in h_question_tokens]

        to_str_pred = lambda w, l: str(" ".join([self.dataloader.ind2word[x] for x in list( filter(
            lambda x:x>0,w.data.cpu().numpy()))][:l.item()]))[8:]
        to_str_gt = lambda w: str(" ".join([self.dataloader.ind2word[x] for x in filter(
            lambda x:x>0,w.data.cpu().numpy())]))[8:-6]

        self.answererModel.reset()
        self.answererModel.observe(
            -1, image=img_feats, caption=caption_tensor, captionLens=caption_lens)

        numRounds = len(history['dialog'])
        beamSize = 5
        for round in range(numRounds):
            self.answererModel.observe(
                round,
                ques=hist_ques_tensors[round],
                quesLens=hist_ques_lens[round])
            self.answererModel.observe(
                round,
                ans=hist_ans_tensors[round],
                ansLens=hist_ans_lens[round])
            _ = self.answererModel.forward()
            answers, ansLens = self.answererModel.forwardDecode(
                inference='greedy', beamSize=beamSize)

        # After processing history
        self.answererModel.observe(
            numRounds,
            ques=question_tensor,
            quesLens=question_lens)
        answers, ansLens = self.answererModel.forwardDecode(
            inference='greedy', beamSize=beamSize)

        result = {}
        result['answer'] = to_str_pred(answers[0], ansLens[0])
        result['question'] = question_asked
        result['history'] = question_asked + ' ' + to_str_pred(answers[0], ansLens[0])
        result['history'] = result['history'].replace('<START>','')

        return result
