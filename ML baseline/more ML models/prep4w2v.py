# -*- coding: utf-8 -*-
import re
import nltk
from file_io import load_text_classification_data_xlsx


def clinic_text_processing(clinic_text):
    # clinic_text.encode('utf-8').replace("&nbsp;", " ").replace("<P>","").replace("</P>", "").strip()
    # print(clinic_text.encode('utf-8'))
    clinic_text = re.sub(r'\<.*?\>', '', clinic_text)
    clinic_text = clinic_text.replace("\\0xc2\\xa0", " ").replace("0xb0", " ").replace("\\xa0", " ").replace("&nbsp;", " ").replace("<P>","").replace("</P>", "").strip()
    clinic_text =  clinic_text.decode('utf-8',errors='ignore')
    clinic_text = " ".join((nltk.word_tokenize(clinic_text)))
    return clinic_text
    



big_file = "../report_from_final-annotated.txt"
anno_file = "../../9K_reports_20181228.xlsx"
X, Y  = load_text_classification_data_xlsx(anno_file,"E",'D', True)
sent_dict = {}
for x in X:
	x = " ".join((nltk.word_tokenize(x))).lower()
	if x not in sent_dict:
		sent_dict[x] = 1
with open(big_file,'r') as fin:
	fins = fin.readlines()
	for line in fins:
		line = clinic_text_processing(line.strip('\n')).lower()
		if line not in sent_dict:
			sent_dict[line] = 1
print("Total line:", len(sent_dict))
fout = open("MGH.Report.lower.4w2v",'w')
for k, v in sent_dict.iteritems():
	fout.write(k.encode('utf-8')+"\n")








