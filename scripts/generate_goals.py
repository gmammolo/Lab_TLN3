"""Genera un CSV con le parole ambigue (più di un synset) da semcor_index.sqlite3.

Output: outputs/goals.csv (delimiter `;`) con colonne:
parola;frase;synset

Usage:
	python3 scripts/generate_goals.py [--db PATH] [--out PATH]
"""

from pathlib import Path
import argparse
import sqlite3
import json
import csv
import sys

try:
	import nltk
	from nltk.corpus import wordnet as wn
except Exception:
	nltk = None
	wn = None


def ensure_wordnet():
	if wn is None:
		print("Required package 'nltk' not installed. Install with 'pip install nltk'", file=sys.stderr)
		sys.exit(3)
	try:
		# try a cheap access
		wn.synsets('test')
	except LookupError:
		nltk.download('wordnet')


def penn_to_wn_pos(penn):
	"""Map Penn Treebank POS to WordNet POS tag (n, v, a, r) or None."""
	if not penn:
		return None
	p = penn.upper()
	if p.startswith('NN'):
		return 'n'
	if p.startswith('VB'):
		return 'v'
	if p.startswith('JJ'):
		return 'a'
	if p.startswith('RB'):
		return 'r'
	return None


def normalize_word(word):
	"""Normalizza la parola per la ricerca in WordNet: lowercasing, rimozione underscores."""
	if not word:
		return ''
	return word.lower().replace('_', ' ').strip()


def split_synsets_field(val):
	if val is None:
		return []
	if isinstance(val, (list, tuple)):
		return [str(v) for v in val if v]
	s = str(val).strip()
	if not s:
		return []
	# Prefer splitting on explicit separators; keep colon-containing ids intact
	if '|' in s:
		parts = [p.strip() for p in s.split('|') if p.strip()]
	elif ',' in s:
		parts = [p.strip() for p in s.split(',') if p.strip()]
	elif ';' in s:
		parts = [p.strip() for p in s.split(';') if p.strip()]
	elif ' ' in s:
		parts = [p.strip() for p in s.split() if p.strip()]
	else:
		parts = [s]
	return parts


def extract_ambiguous_tokens_from_row(sentence_text, metadata_json, lang='eng'):
	results = []
	try:
		data = json.loads(metadata_json)
	except Exception:
		return results
	tokens = data.get('tokens') or []
	# Ensure WordNet data available before any wn lookups
	ensure_wordnet()
	for tok in tokens:
		word = tok.get('word') or tok.get('lemma') or ''
		if not word:
			continue

		# Get lemma and normalize for WordNet lookup
		lemma = tok.get('lemma') or tok.get('word')
		normalized_lemma = normalize_word(lemma) if lemma else ''
		if not normalized_lemma:
			continue

		# Get POS tag
		penn = tok.get('pos')
		wn_pos = penn_to_wn_pos(penn)

		# Check WordNet synsets for the normalized word
		syns_list = wn.synsets(normalized_lemma, pos=wn_pos, lang=lang)

		# CRITICAL: Only process if word has >= 2 synsets in WordNet
		if len(syns_list) < 2:
			continue

		# Now check metadata for synsets
		wnsn = tok.get('wnsn')
		lexsn = tok.get('lexsn')

		# Case 1: token has numeric WordNet sense number
		if wnsn is not None:
			try:
				idx = int(str(wnsn))
				# Validate that index is within bounds
				if 1 <= idx <= len(syns_list):
					syn_name = syns_list[idx - 1].name()
					results.append((word, sentence_text, syn_name))
			except (ValueError, IndexError):
				pass
			continue

		# Case 2: token has lexsn (fallback)
		if lexsn:
			syns = split_synsets_field(lexsn)
			if not syns:
				continue

			# deduplicate while preserving order
			seen = set()
			syns_unique = []
			for s in syns:
				if s not in seen:
					seen.add(s)
					syns_unique.append(s)

			# Validate that synsets are correct (must exist in WordNet for this word)
			valid_syns = []
			for syn_str in syns_unique:
				try:
					# Try to load the synset and verify it matches the word
					syn = wn.synset(syn_str)
					# Check if synset is in the word's synsets
					if syn in syns_list:
						valid_syns.append(syn_str)
				except Exception:
					pass

			# Add row only if we have at least one valid synset
			if valid_syns:
				results.append((word, sentence_text, '|'.join(valid_syns)))

	return results


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--db', default=None, help='Path to semcor_index sqlite DB')
	parser.add_argument('--out', default=None, help='Output CSV path')
	args = parser.parse_args()

	repo_root = Path(__file__).resolve().parents[1]
	db_path = Path(args.db) if args.db else repo_root / 'data' / 'semcor_index.sqlite3'
	out_dir = Path(args.out).parent if args.out else repo_root / 'outputs'
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = Path(args.out) if args.out else out_dir / 'goals.csv'

	if not db_path.exists():
		print(f"Database not found: {db_path}", file=sys.stderr)
		sys.exit(2)

	conn = sqlite3.connect(str(db_path))
	cur = conn.cursor()
	cur.execute('SELECT id, sentence_text, metadata FROM sentences')

	# write CSV with semicolon delimiter
	with open(out_path, 'w', newline='') as fh:
		writer = csv.writer(fh, delimiter=';')
		# header
		writer.writerow(['parola', 'frase', 'synset'])
		for row in cur:
			sid, sentence_text, metadata = row
			if not metadata:
				continue
			ambs = extract_ambiguous_tokens_from_row(sentence_text, metadata)
			for word, sent, syns in ambs:
				if(len(wn.synsets(word))>1):
					writer.writerow([word, sent, syns])

	conn.close()
	print(f'Wrote {out_path}')


if __name__ == '__main__':
	main()

