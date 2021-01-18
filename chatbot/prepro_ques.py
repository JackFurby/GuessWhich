import json
import argparse
import re
import nltk

regex = re.compile('[^\sa-zA-Z]')
nltk.download('punkt')


def main(question_args='', ques_len=15, history_args='', fact_len=30, delimiter='||||'):
    ques = regex.sub('', question_args) + ' ?'
    question = nltk.tokenize.word_tokenize(ques.replace('?', ' ? ').strip().lower())[:ques_len]
    history_facts = history_args.replace('?', ' ? ').split(delimiter)
    history, questions = [], []
    for i in history_facts:
        fact = nltk.tokenize.word_tokenize(i.strip().lower())[:fact_len]
        if len(fact) != 0:
            history.append(fact)
            try:
                questions.append(fact[:fact.index('?')+1])
            except:
                pass

    num_hist = min(len(history), 10)
    num_ques = num_hist - 1 if num_hist > 0 else 0
    return(json.dump({'question': question, 'history': history[-num_hist:], 'questions': questions[-num_ques:]}, open('ques_feat.json', 'w')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=str, default='')
    parser.add_argument('-ques_len', type=int, default=15)
    parser.add_argument('-history', type=str, default='')
    parser.add_argument('-fact_len', type=int, default=30)
    parser.add_argument('-delimiter', type=str, default='||||')
    args = parser.parse_args()
    main(args.question, args.ques_len, args.history, args.fact_len, args.delimiter)
