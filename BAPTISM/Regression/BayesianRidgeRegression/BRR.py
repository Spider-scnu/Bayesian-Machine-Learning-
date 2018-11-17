from __future__ import print_function
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
#import json
from urllib.parse import parse_qs
import os
import re
import sys
import threading
import cx_Oracle as oracle
#import grpc
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.keras as kr
import predict_RNN
from predict_RNN import RnnModel
import chardet

#---------------------------------------------------------- #
''' 连接Oracle数据库 '''
#result_dir = '/home/whale/Desktop/WiseFly-EMR-NLP-master/results-documents/EMR.txt'
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK' # change the observation encode
db=oracle.connect('orcltest', 'orcl','192.168.1.137:1521/orcl') 
cursor1=db.cursor()
cursor1.execute("select t.WORD_CONTENT from HAI_WARNING_KEYWORD t where t.WORD_TYPE=2 and t.OPER_TYPE=1")
KeyWords = cursor1.fetchall()

#-----------------------------------------------------------#
''' 导入长短时记忆神经网络 '''
tf.reset_default_graph()
Rnn = RnnModel()

def work(text, keywords = KeyWords, connection = 'database'):
    if text is None:
        return 0
    else:
        cases, _, indexs, ColorContent, RelatedText, _, _ = \
                        Rnn.lookfor(text, keywords, connection)
        return cases, indexs , ColorContent, RelatedText


        
def NLPModel(PatID, RecordID=''):
    cursor2 = db.cursor()
    Non = "None"
    dot = "'"
    And = " and t.record_id="
    top = "select t.RECORD_CONTENT from HAI_V_EMR_RECORD t where t.PATIENT_ID="
    tail = str(PatID)
    if RecordID == '':
        try:
            cursor2.execute(top+dot+tail+dot)
            Contexts = cursor2.fetchall()
            Content = str(Contexts[0][0])
            #print(Content)
            cases, indexs, ColorContent, _ = work(Content, KeyWords, connection = 'database')
            a = ''
            for i in range(int(len(indexs)/2)):
                sent = indexs[2*i] + '|' + str(indexs[2*i - 1]) + ';'
                a += sent
            #print(a)    
            if len(a) == 0:
                a = '该病人没有潜在关键病症.'
            return ColorContent
        except:
            return Non
    else:
        #print("There was Record ID.")
        tail1 = str(RecordID)
        try:
            cursor2.execute(top+dot+tail+dot+And+dot+tail1+dot)
            Contexts = cursor2.fetchall()
            Content = str(Contexts[0][0])
            #print(Content)
            cases, indexs, ColorContent, _ = work(Content, KeyWords, connection = 'database')
            a = ''
            for i in range(int(len(indexs)/2)):
                sent = indexs[2*i] + '|' + str(indexs[2*i - 1]) + ';'
                a += sent
            #print(a)    
            if len(a) == 0:
                a = '该病人没有潜在关键病症.'
            return ColorContent
        except:
            return Non
    
    
    

class TodoHandler(BaseHTTPRequestHandler):
    Todos = []
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/xml; charset=utf-8' )
        self.end_headers()
    def do_HEAD(self):
        self._set_headers()
    def do_GET(self):
        self._set_headers()
        print(self.Todos)
        print(self.path)
        print(parse_qs(self.path[2:]))
        try:
            try:
                PatID, RecordID = parse_qs(self.path[2:])['PatID'][0], parse_qs(self.path[2:])['RecordID'][0]
                #PatID, RecordID = parse_qs(self.path[2:])['name'][0], parse_qs(self.path[2:])['name1'][0]
                ColorContent = NLPModel(PatID, RecordID)
                #print(ColorContent)
                #print(chardet.detect(ColorContent.encode('ascii')), chardet.detect(str.encode("<html><body><h1>")))
                self.wfile.write(str.encode("<html><body><h1>"))
                self.wfile.write(str.encode(ColorContent))
                self.wfile.write(str.encode("</h1></body></html>"))
            except:
                PatID = parse_qs(self.path[2:])['PatID'][0]
                #PatID = parse_qs(self.path[2:])['name'][0]
                ColorContent = NLPModel(PatID)
                #print(chardet.detect(str.encode(ColorContent)))
                self.wfile.write(str.encode("<html><body><h1>"))
                self.wfile.write(str.encode(ColorContent))
                self.wfile.write(str.encode("</h1></body></html>"))
        except:
            #PatID = parse_qs(self.path[2:])['name'][0]
            self.wfile.write(str.encode(<?xml version="1.0"?>))
            self.wfile.write(str.encode(<data>))
            self.wfile.write(str.encode(<country name="Liechtenstein">))
            self.wfile.write(str.encode(<rank>1</rank>))
            self.wfile.write(str.encode(<year>2008</year>))
            self.wfile.write(str.encode(<gdppc>141100</gdppc>))
            self.wfile.write(str.encode(<neighbor name="Austria" direction="E"/>))
            self.wfile.write(str.encode(<neighbor name="Switzerland" direction="W"/>))
            self.wfile.write(str.encode(</country>))
            self.wfile.write(str.encode(<country name="Singapore">))
            self.wfile.write(str.encode(<rank>4</rank>))
            self.wfile.write(str.encode(<year>2011</year>))
            self.wfile.write(str.encode(<gdppc>59900</gdppc>))
            self.wfile.write(str.encode(<neighbor name="Malaysia" direction="N"/>))
            self.wfile.write(str.encode(</country>))
            self.wfile.write(str.encode(<country name="Panama">))
            self.wfile.write(str.encode(<rank>68</rank>))
            self.wfile.write(str.encode(<year>2011</year>))
            self.wfile.write(str.encode(<gdppc>13600</gdppc>))
            self.wfile.write(str.encode(<neighbor name="Costa Rica" direction="W"/>))
            self.wfile.write(str.encode(<neighbor name="Colombia" direction="E"/>))
            self.wfile.write(str.encode(</country>))
            self.wfile.write(str.encode(</data>))
            #self.wfile.write(str.encode("<html><body><h1>"))
            #self.wfile.write(str.encode("Get Request Receivded!\n"))
            #self.wfile.write(str.encode("Error Get!"))
            #self.wfile.write(str.encode("</h1></body></html>"))
            
        #except:
            
        
    def do_POST(self):
        self._set_headers()
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers = (self.headers),
            environ={str('REQUEST_METHOD'):str('POST')}
        )
        self.wfile.write(str.encode("<html><body><h1>"))
        self.Todos.append(form.getvalue("name"))
        self.wfile.write(str.encode("Hello "+form.getvalue("name")))
        try:
            self.wfile.write(str.encode("Hello "+form.getvalue("name")))
        except:
            self.wfile.write(str.encode("None this input"))
        #print(form.getvalue("bin"))
        self.wfile.write(str.encode("         POST Request Received!" ))
        self.wfile.write(str.encode("</h1></body></html>" ))

if __name__ == '__main__':
    server = HTTPServer(('192.168.219.1', 8080), TodoHandler)
    print("Starting server, use <Ctrl-C> to stop")
    server.serve_forever()
        