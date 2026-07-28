import re
from tqdm import tqdm

global boundaries_regex,whitespace_regex
boundaries_regex = re.compile(r'''(?x)(?=[^\.])([\r\n])+(?![a-z0-9\(]|[A-Z][^a-z]+ )|\.+(\s\*\*\*)?([\r\n\s])+|(?<!(D|d)r)\.+\s+(?=[a-zA-Z])|(?<=:)\s{2,}(?![a-zA-Z0-9])|(?<!:)\s{2,}(?![a-z])''')
whitespace_regex = re.compile(r'([a-zA-Z0-9,]{2,})[ \t]{2,}([a-z0-9+])')


def find_sentences(raw, char_level=False):
	# remove excess whitespace
	raw = re.sub(whitespace_regex, r'\1 \2', raw)
	raw = re.sub(whitespace_regex, r'\1 \2', raw)
	#keeps ending " within sentence
	raw = raw.replace('."', '".')
	#removing \r
	raw = re.sub(r'\r', r'', raw)
	#hides periods in Dr. and Vs.
	raw = re.sub(r'((D|d)(R|r)).',r'\1<keepperiod>',raw)
	raw = re.sub(r'((V|v)(S|s)).',r'\1<keepperiod>',raw)
	#hides periods following single initials
	raw = re.sub(r'([A-Z][a-z]+\s)([A-Z])\.(\s[A-Z][a-z]+)',r'\1\2<keepperiod>\3',raw)
	raw = re.sub(r'\s([A-Z]|[a-z])\.\s',r' \1<keepperiod> ',raw)
	#hides periods from abbreviations
	raw = re.sub(r'(?<=\.[a-zA-Z])(\.)|(?<=[a-zA-Z])(\.)(?=\S)', r'<keepperiod>', raw)
	#condenses multiple blank lines (for double spaced notes)
	raw = re.sub(r'\n\n|\n +(?=[a-z])',r'\n',raw)
	#finds end of sent when sent ends with in-line code being displayed as a dict
	raw = re.sub(r'}(\.|\s+(?=[A-Z]))','}<><>', raw)
	raw = re.sub(boundaries_regex, '<><>', raw)
	raw = [tokenize(x.replace('\n',' ').strip().replace('<keepperiod>','.'), char_level=char_level) for x in raw.split('<><>') if x.replace('\n',' ').strip()!='']
	return raw


def tokenize(string, char_level=False):
	nums = '12345678990'
	lowlets = 'qwertyuiopasdfghjklzxcvbnm'
	uplets = 'QWERTYUIOPASDFGHJKLZXCVBNM'
	newstring = ''
	for i,char in enumerate(string):
		#if punctuation or special character, encase in spaces
		if char not in ' '+nums+lowlets+uplets:
			#exception for decimal points
			if char == '.' and (i==0 or (i>0 and string[i-1] in ' '+nums)) and (i<len(string)-1 and string[i+1] in nums):
				newstring += char
			#exception for negative numbers
			elif char == '-' and (i==0 or (i>0 and string[i-1] in ' ')) and (i<len(string)-1 and string[i+1] in nums):
				newstring += ' ' if (i>0 and string[i-1]!=' ') else ''
				newstring += char
			else:
				#only add spaces if space isn't already there
				newstring += ' ' if (i>0 and string[i-1]!=' ') else ''
				newstring += char
				newstring += ' ' if (i<len(string)-1 and string[i+1]!=' ') else ''
		#else just add the character
		else:
			newstring += char
	newstring = newstring.strip()
	if char_level:
		return [char for char in newstring]
	return newstring.split()


if __name__ == '__main__':
	readfile = '../falls/newnotes/joined_ascii_PHS_CONCERN_STUDY_2016-q1.txt'
	read_sep = '|'
	docID_inx = 9
	text_inx = 11

	docs = {}
	with open(readfile) as rf:
		skip = True
		for line in tqdm(rf):
			if skip:
				skip = False

			else:
				splitline = line.split(read_sep)
				doc_ID = splitline[docID_inx].strip()
				doc_text = splitline[text_inx]
				docs[doc_ID] = doc_text

	sents = []
	for doc_ID in tqdm(docs):
		sents += [' '.join(x) for x in find_sentences(docs[doc_ID]) if len(x)>2]

	sents = list(set(sents))
	with open('processed_files.tsv','w+') as wf:
		for i,sent in tqdm(enumerate(sents)):
			wf.write('{}\t{}\n'.format(i,sent))