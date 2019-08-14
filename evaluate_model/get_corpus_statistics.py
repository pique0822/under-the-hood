import json
import argparse
from nltk import pos_tag

def load_json(path):
	with open(path, 'r') as f:
		d = json.load(f)
	return d

def write_lines(path, lines):
	with open(path, 'w') as f:
		for l in lines:
			f.write('{}\n'.format(l))

def top_n_words(vocab_dict, n):
	return [vocab_dict['idx2word'][str(i)] for i in range(n)]

def tag(words):
	# don't tag words directly because NLTK may try to interpret as a sentence
	return [pos_tag([w])[0] for w in words]

def count(texts, token):
	count = sum(
		sum(1 for t in text.strip().split(' ') if t == token)
	for text in texts)
	return count

def check_counts():
	print('loading train data')
	with open('/om/user/jennhu/colorlessgreenRNNs/data/wiki/train.txt', 'r') as f:
		train = f.readlines()
	train = [t.strip() for t in train]
	import pandas as pd
	print('loading verbs')
	df = pd.read_csv('wiki_vbdvbn_all.csv')
	print('getting counts')
	counts = {t:0 for t in df.token.values}
	for sent in train:
		for t in sent.strip().split(' '):
			if t in counts:
				counts[t] += 1
	# counts = [count(train, token) for token in df.token.values]
	# df['train_count'] = counts
	# df.to_csv('wiki_vbdvbn_all_counts.csv')
	with open('wiki_vbdvbn_all_counts.json', 'w') as f:
		json.dump(counts, f, indent=4)
	print('saved')

# def main(vocab, n, pos, outfile):
# 	print('Loading dictionary from ' + vocab)
# 	d = load_json(vocab)

# 	print('Getting top %d words' % n)
# 	top_words = top_n_words(d, n)

# 	print('Getting POS tags')
# 	tags = tag(top_words)

# 	print('Writing tokens from {}'.format(pos))
# 	tokens = ['{},{}'.format(t[0],t[1]) for t in tags if t[1] in pos]
# 	tokens.insert(0, 'token,POS')

# 	write_lines(outfile, tokens)

def main(vocab, n, pos, outfile):
	print('Loading dictionary from ' + vocab)
	d = load_json(vocab)
	words = d['word2idx'].keys()

	print('Getting POS tags')
	tags = tag(words)

	print('Keeping tokens from {}'.format(pos))
	# tags = [t for t in tags if t[1] in pos]
	lines = ['{},{}'.format(t[0],t[1]) for t in tags]
	lines.insert(0, 'token,POS')

	write_lines(outfile, lines)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get top verbs.')
	parser.add_argument('--vocab', '-vocab', type=str,
						default='/om/user/jennhu/colorlessgreenRNNs/data/wiki/vocab_dict.json',
	                    help='path to vocab dict')
	parser.add_argument('--n', '-n', type=int, default=10000,
	                    help='number of top words in vocab to consider')
	parser.add_argument('--pos', '-pos', nargs='+', default=['VBD', 'CC', 'IN', 'RB'],
						help='POS tags to include in output file')
	parser.add_argument('--outfile', '-outfile', default='wiki.csv',
						help='file to write data')
	args = parser.parse_args()
	# main(**vars(args))
	check_counts()
