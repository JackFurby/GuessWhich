# GuessWhich

## Introduction

**Evaluating Visual Conversational Agents via Cooperative Human-AI Games**  
Prithvijit Chattopadhyay*, Deshraj Yadav*, Viraj Prabhu, Arjun Chandrashekharan, Abhishek Das, Stefan Lee, Dhruv Batra, Devi Parikh  
[HCOMP 2017][4]

This repository contains code for setting up the **GuessWhich Game** along with Amazon Mechinical Turk (AMT) integration for real time data collection. The data collection settings can be changed easily by modifying certain configurations defined [here](https://github.com/VT-vision-lab/GuessWhich/blob/master/amt/constants.py).

## Abstract

As AI continues to advance, human-AI teams are inevitable. However, progress in AI is routinely measured in isolation, without a human in the loop. It is important to measure how progress in AI translates to humans being able to accomplish tasks better; i.e., the performance of human-AI teams. In this work, we design a cooperative game – GuessWhich to measure human-AI team performance in the specific context of the AI being a visual conversational agent. The AI, which we call ALICE, is provided an image which is unseen by the human. The human then asks ALICE questions about this secret image to identify it from a fixed pool of images.

We measure performance of the human-ALICE team by the number of guesses it takes the human to correctly identify the secret image after a fixed number of dialog rounds with ALICE. We compare performance of the human-ALICE teams for two versions of ALICE. While AI literature shows that one version outperforms the other when paired with another AI, we find that this improvement in AI-AI performance does not translate to improved human-AI performance.


## Installation Instructions

### Installing the essential requirements

```shell
sudo dnf install -y git python3.7 python-pip python-devel
sudo dnf install -y autoconf automake libtool curl make g++ unzip
sudo dnf install -y gflags-devel glog-devel lmdb-libs
sudo dnf install protobuf-devel leveldb-devel snappy-devel opencv-devel hdf5-devel protobuf-compiler
sudo dnf install postgresql postgresql-contrib postgresql-server openssl-devel
```

### Install Torch

```shell
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
```

### Install PyTorch(Python Lua Wrapper)

```shell
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```


### Install Pip packages

(update for conda)

```
pip install -r requirements.txt
```

### Install RabbitMQ and Redis Server

```shell
sudo dnf install -y redis rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
sudo service rabbitmq-server restart
sudo systemctl start redis.service
```

### Lua dependencies

```shell
luarocks install loadcaffe
```

The below two dependencies are only required if you are going to use GPU

```shell
luarocks install cudnn
luarocks install cunn
```

### Cuda Installation

Note: CUDA and cuDNN is only required if you are going to use GPU

Download and install CUDA and cuDNN from [nvidia website](https://developer.nvidia.com/cuda-downloads)

### Install dependencies

```shell
git clone https://github.com/Cloud-CV/GuessWhich.git
cd GuessWhich
sh download_models.sh
pip install -r requirements.txt
```

### Create the database

```shell
sudo postgresql-setup initdb
sudo systemctl start postgresql
sudo su - postgres
psql
CREATE DATABASE guesswhich;
CREATE USER demo WITH PASSWORD 'demo';
ALTER ROLE demo SET client_encoding TO 'utf8';
ALTER ROLE demo SET default_transaction_isolation TO 'read committed';
GRANT ALL PRIVILEGES ON DATABASE guesswhich TO demo;
\q
exit

python manage.py makemigrations amt
python manage.py migrate
```

You will need to change authentication methods in `pg_hba.conf`. `ident` for `IPv4` and `IPv6` local connections should be changed to `md5`.

```
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
# IPv6 local connections:
host    all             all             ::1/128                 md5
```

Once done you will need to restart Postgress with the command

```shell
sudo systemctl restart postgresql
```

### Running the RabbitMQ workers and Development Server

Open 3 different terminal sessions and run the following commands:

```shell
cd chatbot && python sl_worker.py -qstartFrom="chatbot/data/sl_qbot.vd" -startFrom="chatbot/data/sl_abot.vd"
cd chatbot && python rl_worker.py  -qstartFrom="chatbot/data/rl_qbot.vd" -startFrom="chatbot/data/rl_abot.vd"
python manage.py runserver
```

You are all set now. Visit http://127.0.0.1:8000 and you will have your demo running successfully.


## Cite this work

If you find this code useful, consider citing our work:

```
@inproceedings{visdial_eval,
  title={Evaluating Visual Conversational Agents via Cooperative Human-AI Games},
  author={Prithvijit Chattopadhyay and Deshraj Yadav and Viraj Prabhu and Arjun Chandrasekaran and Abhishek Das and Stefan Lee and Dhruv Batra and Devi Parikh},
  booktitle={Proceedings of the Fifth AAAI Conference on Human Computation and Crowdsourcing (HCOMP)},
  year={2017}
}
```

## Contributors

* [Deshraj Yadav][2] (deshraj@gatech.edu)

## License

BSD

## Credits

- Vicki Image: "[Robot-clip-art-book-covers-feJCV3-clipart](https://commons.wikimedia.org/wiki/File:Robot-clip-art-book-covers-feJCV3-clipart.png)" by [Wikimedia Commons](https://commons.wikimedia.org) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)

[1]: https://arxiv.org/abs/1611.08669
[2]: http://deshraj.github.io
[4]: http://www.humancomputation.com/2017/
