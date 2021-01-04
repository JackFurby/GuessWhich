import torch
import torch.nn as nn
import chatbot.models.encoders.hre as hre_enc
import chatbot.models.decoders.gen as gen_dec
from chatbot.models.questioner import Questioner
from chatbot.models.answerer import Answerer
from utils import utilities as utils
from chatbot.dataloader import VisDialDataset
import chatbot.options as options
import random


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

        self.questionerModel.eval()
        self.answererModel.eval()
