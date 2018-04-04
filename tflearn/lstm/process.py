import os
import sys
import json

file_path = './projectnextnlp-chat-dialogue-corpus/json/rest1046/'
file_dir = os.listdir(file_path)

# store machine's answer flag and human's utterance text
label_text = []

for file in file_dir:
    r = open(file_path + file, 'r', encoding='utf-8')
    json_data = json.load(r)

    for turn in json_dat['turns']:
        turn_index = turn['turn-index']
        speaker = turn['speaker']
        utterance = turn['utterance']

        #exclude the first line
        if turn_index != 0:
            #extract human's utterance
            if speaker == 'U':
                u_text = ''
                u_text = utterance
            #extract machine's answer
            else:
                a = ''
                for annotate in turn['annotattions']:
                    a = annotate['breakdown']
                    if a == '0':
                        a = 0 #0 means the answer is inappropriate
                    else:
                        a = 1 #1 means the answer is appropriate

                    temp = str(a) + '\t' + u_text
                    temp2 = temp.split('\t')
                    label_text.append(temp2)
