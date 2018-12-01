# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 04:50:03 2018

@author: Half_BlooD PrincE
"""


from eval_subtitles import *
import os.path
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import groupby

def parse_subtitle(filename):
    # "chunk" our input file, delimited by blank lines
    with open(filename, 'rb') as f:
        res = [list(g) for b,g in groupby(f, lambda x: bool(x.strip())) if b]

    content=""
    subs = []
    number = 0
    count=0
    for sub in res:
        if len(sub) >= 3: # not strictly necessary, but better safe than sorry
            sub = [x.strip() for x in sub]
            try:
                number = sub[0].decode("UTF-8")
            except:
                number += 1            
            content += " "
            
#            content += sub[2].decode("utf-8")
            content += "".join(map(chr, sub[2]))
#            if len(start_end.split(' --> ')) == 2:
#                start, end = start_end.split(' --> ') # e.g. 02:14:53,085
#                
#                if len(start) >= 12 and len(end) >= 12:
#                    start = start[:12] #for truncating unnecessary fields, if any
#                    end = end[:12] #for truncating unnecessary fields, if any
#                    try:
#                        at_minute = int(start[:2]) * 60 + int(start[3:5])
#                        at_seconds = int(start[:2]) * 3600 + int(start[3:5]) * 60 + int(start[6:8])
#                    except:
#                        at_minute = 0
#                        at_seconds = 0
#                        #continue
#                    subs.append(Subtitle(number, start, end, content, at_minute, at_seconds))
#            
#            elif len(start_end.split(' -> ')) == 2:
#                start, end = start_end.split(' -> ')
#
#                if len(start) >= 12 and len(end) >= 12:
#                    start = start[:12] #for truncating unnecessary fields, if any
#                    end = end[:12] #for truncating unnecessary fields, if any
#                    try:
#                        at_minute = int(start[:2]) * 60 + int(start[3:5])
#                        at_seconds = int(start[:2]) * 3600 + int(start[3:5]) * 60 + int(start[6:8])
#                    except:
#                        at_minute = 0
#                        at_seconds = 0
#                        #continue
#                    subs.append(Subtitle(number, start, end, content, at_minute, at_seconds))
            

    return content


labels, texts = [], []
for dirpath, dirnames, filenames in os.walk("../Subtitles"):
        dirname = dirpath.split("/")[-1]

        print(dirname)

        cnt = 0
        indices = []

        files = [f for f in filenames if f.endswith(".srt")]
        for index, filename in enumerate(files):
            # if filename == 'Broken Arrow (IMPAIRED).srt':
            #     print(os.path.join(dirpath, filename))
            
            content = parse_subtitle(os.path.join(dirpath, filename))




            labels.append(dirname)
            #texts.append(content[1:])
            texts.append(''.join(content))
            
            
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

            
            
            
            
            
            