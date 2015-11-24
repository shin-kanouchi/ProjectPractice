#!/usr/bin/python
#-*-coding:utf-8-*-

import sys
import MeCab
tagger = MeCab.Tagger("-Owakati")


#time,A,10月23日,12:23,国語は好きですか？,好きです,5
#time,A,10月23日,12:23,数学は好きですか？,好きです,4

def do_mecab():
    for line in open(sys.argv[1]):
        # time,C,10月27日,14:18,動物は好きですか？,はい,4
        item = line.strip().split(',')
        wakati1 = tagger.parse(item[4])
        wakati2 = tagger.parse(item[5])
        print "%s,%s,%s,%s,%s,%s,%s" % (item[0],item[1],item[2],item[3],wakati1.strip() , wakati2.strip(), item[6])


do_mecab()