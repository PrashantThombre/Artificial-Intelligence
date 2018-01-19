import nltk
import urllib
from nltk import word_tokenize, sent_tokenize, pos_tag
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import markovify
import random
import os, webbrowser
def main():
    characters = ['Holmes','Moriarty','Watson', 'Mycroft', 'Victoria', 'Gregory', 'Dartmoor' , 'John' , 'Simpson', 'Tavistock', 'Pullman']

    with open("sherlock.txt") as f:
        response = f.read()

    raw = response.decode('utf8').encode("ascii", "ignore")
    raw = raw.replace("!",".").replace("?",".")
    raw = raw.replace("(","").replace(")","").replace("[","").replace("]","").replace(":","").replace(";","").replace("-","")\
        .replace("201d","")

    sentences = sent_tokenize(raw)
    interaction_list = list()
    cp = nltk.RegexpParser('CHUNK: {<NNP><.*>*<V.*>+<.*>*<NNP>}')
    for sentence in sentences:
        tagged_sentences = pos_tag(word_tokenize(sentence))
        tree = cp.parse(tagged_sentences)
        for subtree in tree.subtrees():
            interaction_set = set()
            if subtree.label() == 'CHUNK':
                interaction_set.add(subtree[0][0])
                interaction_set.add(subtree[-1][0])
                count = 0
                if len(interaction_list)!= 0:
                    decisionvar = False
                    for item in interaction_list:
                        if interaction_set.issubset(item[0]):
                            item[1] += 1
                            decisionvar = True
                            break
                    if decisionvar == False:
                        newList = []
                        newList.append(interaction_set)
                        newList.append(1)
                        interaction_list.append(newList)
                else:
                    newList = []
                    newList.append(interaction_set)
                    newList.append(1)
                    interaction_list.append(newList)

    character_list = list()
    counts = list()
    for interactions in interaction_list:
        try:
            person1 = interactions[0].pop()
            person2 = interactions[0].pop()
        except KeyError:
            continue
        else:
            character_list.append([person1, person2])
            counts.append(interactions[1])

    print 'Printing Interaction Frequencies: '
    for i,val in enumerate(character_list):
        print val
        print counts[i]

    person_name = random.choice(characters)
    text_model = markovify.Text(raw)
    script = ""
    for i in range(random.randint(2,20)):
        script += (text_model.make_sentence_with_start(person_name, strict=False))+"<br>"

    f = open('sherlock.html', 'w')
    message = """<HTML>
           <HEAD></HEAD>
           <BODY style="background-image:url('https://img00.deviantart.net/1b85/i/2014/012/f/3/sherlock_wallpaper_by_youngfab4fan-d71vzx4.png');">
               <H3>Our Markovified Script For """+person_name+"""</H3>"""
    message += """<div>"""+script+"""</div></BODY></HTML>"""

    f.write(message)
    f.close()
    webbrowser.open('file://' + os.path.realpath('sherlock.html'))
if __name__=='__main__':
    main()
