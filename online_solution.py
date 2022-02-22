import argparse
import kenlm
import numpy as np
import pandas as pd
import pickle as pkl
import pymorphy2
import torch
from functools import lru_cache
from pymystem3 import Mystem
from sklearn.neighbors import KDTree
from torch import softmax, sigmoid
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TOXIC_CLASS=-1
TOKENIZATION_TYPE='sentencepiece'

ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


def get_w2v_indicies(a):
    res = []
    if isinstance(a, str):
        a = a.split()
    for w in a:
        if w in embs_voc:
            res.append((w, embs_voc[w]))
        else:
            for lemma in stemmer.lemmatize(w):
                if lemma.isalpha():
                    res.append((w, embs_voc.get(lemma, None)))
    return res
    

def calc_embs(words):
    words = ' '.join(map(normalize, words))
    inds = get_w2v_indicies(words)
    return [(w, i if i is None else embs_vectors[i]) for w, i in inds]


def calc_single_embedding_dist(a, b):
    a_s, a_v = a
    b_s, b_v = b
    if a_s == b_s:
        return 0.0
    if a_v is None or b_v is None:
        return 1.0
    a = a_v
    b = b_v
    # inexact match is punished by 0.1
    return 0.1 + 0.9 * (1 - a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)) / 2


def greedy_match_embs(a, b, max_dist=99999, cache=None, a_ind=0, b_ind=0):
    a_len = len(a) - a_ind
    b_len = len(b) - b_ind
    minlen = min(a_len, b_len)
    maxlen = max(a_len, b_len)
    if minlen == 0:
        return np.minimum(maxlen, max_dist)
    if maxlen - minlen >= max_dist:
        return max_dist
    
    if cache is None:
        cache = {}
    
    cache_key = (a_len, b_len)
    if cache_key in cache:
        return cache[cache_key]
        
    min_dist = max_dist
    
    first_dist = calc_single_embedding_dist(a[a_ind], b[b_ind])
    if max_dist >= first_dist:
        min_dist = np.minimum(min_dist, first_dist + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind + 1
        ))
    
    if first_dist > 0 and max_dist >= 1:
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind
        ))
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind, b_ind + 1
        ))
    
    cache[cache_key] = min_dist
    
    return min_dist


def calc_semantic_distance(a, b):
    a_embs = calc_embs(a)
    b_embs = calc_embs(b)
    
    clip_distance = 5  # this clips long computations
    return np.exp(-(greedy_match_embs(a_embs, b_embs, max_dist=clip_distance) / (0.6 * np.log(1 + len(a)))) ** 2)


def distance_score(original, fixed):
    original = original.split()
    fixed = fixed.split()
    
    return calc_semantic_distance(original, fixed)


def is_word_start(token):
    if TOKENIZATION_TYPE == 'sentencepiece':
        return token.startswith('▁')
    if TOKENIZATION_TYPE == 'bert':
        return not token.startswith('##')
    raise ValueError("Unknown tokenization type")


def normalize(sentence, max_tokens_per_word=20):
    def validate_char(c):
        return c in ALLOWED_ALPHABET
    
    sentence = ''.join(map(lambda c: c if validate_char(c) else ' ', sentence.lower()))
    ids = tokenizer(sentence)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)[1:-1]
    
    result = []
    num_continuation_tokens = 0
    for token in tokens:
        if not is_word_start(token):
            num_continuation_tokens += 1
            if num_continuation_tokens < max_tokens_per_word:
                result.append(token.lstrip('#▁'))
        else:
            num_continuation_tokens = 0
            result.extend([' ', token.lstrip('▁#')])
    
    return ''.join(result).strip()


def lm_score(original, fixed):
    original_lm_logproba = lm.score(original, bos=True, eos=True)
    fixed_lm_logproba = lm.score(fixed, bos=True, eos=True)
    
    probability_fraction = 10**((fixed_lm_logproba - original_lm_logproba) / 25)
    
    return np.clip(probability_fraction, 0.0, 1.0)


def get_lexemes(word):
    pym_word = morph.parse(word)[0]
    lexemes = {word}
    for x in pym_word.lexeme:
        lemma = stemmer.lemmatize(x.word)[0]
        if lemma == word and x.word not in embs_voc:
            lexemes.add(x.word)
    
    word_lemma = stemmer.lemmatize(word)[0]
    if word_lemma != word:
        pym_word = morph.parse(word_lemma)[0]
        for x in pym_word.lexeme:
            lemma = stemmer.lemmatize(x.word)[0]
            if lemma == word_lemma and x.word not in embs_voc:
                lexemes.add(x.word)
    
    return list(lexemes)


def get_best_from_vocabs(word, norm_word):
    if word not in vocab_005:
        word = norm_word
        
    # 48 in total
    #res = set(map(lambda x: x[0], vocab_005[word][:20] + vocab_005_01[word][:20] + vocab_01_02[word][:6] + vocab_0_1[word][:2]))
    
    # Пока только с первого словаря 10 ближайших
    res = set(map(lambda x: x[0], vocab_005[word][:10]))

    lexemes = set()
    for w in vocab_005[word][:10]:
        lexemes.update(get_lexemes(w[0]))
    
    res.update(lexemes)
        
    return list(res)


def get_sub_sents(i, word, norm_word, sent):
    word_subs = get_best_from_vocabs(word, norm_word)

    sub_sents = list()

    for sub in word_subs:
        new_sent = sent.split()
        new_sent[i] = sub
        sub_sents.append(' '.join(new_sent))
        
    sub_sents = sorted(map(lambda x: (lm_score(sent, x), x), sub_sents), reverse = True)
    sub_sents_f = list(filter(lambda x: True if x[0] > 0.85 else False, sub_sents))
    if len(sub_sents_f) == 0:
        sub_sents_f = sub_sents[:10]
    # sub_sents = list(map(lambda x: x[1], sub_sents_f))[:10]
    sub_sents = sub_sents_f[:10]
    return sub_sents


def get_vocab_words(sent):
    res = []
    if isinstance(sent, str):
        sent = sent.split()
    for w in sent:
        if w in vocab_005:
            res.append(w)
        else:
            lemma = stemmer.lemmatize(w)[0]
            # Тут замена на лемму
            # res.append(w if lemma in vocab_005 else None)
            res.append(lemma if lemma in vocab_005 else None)
    return res


def order_words_by_tox(sent):
    vocab_sent = get_vocab_words(sent)
    sent_words_tox = list()

    for i, word in enumerate(vocab_sent):
        if word:
            # Тут замена get_toxicity на поиск по словарю
            # sent_words_tox.append((i, word, stemmer.lemmatize(word)[0], get_toxicity(word)[word]))
            lemma = stemmer.lemmatize(word)[0]
            sent_words_tox.append((i, word, lemma, vocab_toxicity.get(word, vocab_toxicity.get(lemma, None))))
    sent_words_tox = sorted(sent_words_tox, key = lambda x: x[-1], reverse = True)

    return sent_words_tox


def beam_search(sent):
    tox_words_order = order_words_by_tox(sent)

    original_sent = sent

    sub_sents = list()
    max_iter = 0 if len(sent.split()) < 9 else 1 if len(sent.split()) < 28 else 2
    for iteration, (i, word, norm_word, tox) in enumerate(tox_words_order):
        if tox > 0.3:
            if iteration == 0:
                sent = get_sub_sents(i, word, norm_word, sent)[0][1]
            else:
                sub_sents = get_sub_sents(i, word, norm_word, sent)
                lm_dist_sub = list()
                for score, sent in sub_sents:
                    lm_dist_sub.append((score * distance_score(original_sent, sent), sent))
                sent = sorted(lm_dist_sub, reverse=True)[0][1]

        if iteration == max_iter:
            break

    # if sub_sents:
        # sent = sub_sents[0]
    return sent


def load_embeddings(path):
    embs_file = np.load(path, allow_pickle=True)
    embs_vectors = embs_file['vectors']
    embs_voc = embs_file['voc'].item()

    embs_voc_by_id = [None for i in range(len(embs_vectors))]
    for word, idx in embs_voc.items():
        if embs_voc_by_id[idx] is None:
            embs_voc_by_id[idx] = word
    return embs_vectors, embs_voc, embs_voc_by_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_texts', type=argparse.FileType('r'))
    parser.add_argument('fixed_texts', type=argparse.FileType('w'))
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--tokenizer', required=True, type=str)
    parser.add_argument('--root', required=True, type=str)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.tokenizer).to(device)

    vocab_toxicity = np.load(args.root + '/vocab_words_toxicity.npy', allow_pickle=True).item()
    vocab_005 = np.load(args.root + '/vocab_005_extended.npy', allow_pickle=True).item()

    embs_vectors, embs_voc, embs_voc_by_id = load_embeddings(args.embeddings)
    embs_vectors_normed = embs_vectors / np.linalg.norm(embs_vectors, axis=1, keepdims=True)

    morph = pymorphy2.MorphAnalyzer()

    lm = kenlm.Model(args.root + "/lm.binary")

    stemmer = Mystem()

    with args.original_texts, args.fixed_texts:
        for line in tqdm(args.original_texts):
            line = normalize(line.strip())
            if len(line.split()) < 80:
                print(beam_search(line), file=args.fixed_texts)
            else:
                print(line, file=args.fixed_texts)
