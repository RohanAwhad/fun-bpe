# Create a Byte Pair Encoding (BPE) tokenizer using the Shakespeare dataset.

from loguru import logger
logger.add('logs/main.log', format='{time} {level} {message}', level='DEBUG', rotation='1 MB')

import argparse
from collections import Counter, defaultdict
import re


def get_word_freq_dict(filename):
    # read file
    fstream = open(filename, 'r')
    vocab = defaultdict(int)

    for line in fstream:
        line = line.strip('\r\n ')

        # remove punctuation
        line = re.sub(r'[^\w\s]', '', line)
    
        # convert to lowercase
        line = line.lower()
    
        # split into words
        words = line.split()
        
        # create a dictionary
        word_dict = Counter(words)

        for word, freq in word_dict.items():
            vocab[word] += freq

    fstream.close()
    logger.info('Read file: {}'.format(filename))
    logger.info('Vocab size: {}'.format(len(vocab)))
    return vocab


def get_stats(vocab):
    """
    Returns a stats and indices dictionary for the given vocabulary.
    Stats is a dictionary with tuple(letter_1, letter_2) as key and frequency as value.
    TODO: Indices I will fill later.
    """
    stats = defaultdict(int)
    indices = defaultdict(dict)

    for i, (word, freq) in enumerate(vocab):
        for prev_char, curr_char in zip(word, word[1:]):
            stats[(prev_char, curr_char)] += freq

            # get cnt of each pair for each word
            cnt = indices[(prev_char, curr_char)].get(i, 0)
            indices[(prev_char, curr_char)][i] = cnt + 1

    return stats, indices


def replace_string(vocab, pair, indices):
    """
    Replace the first_sym with second_sym in the vocab.
    """
    # Create a regex pattern that matches 'first_sym second_sym'
    # with no non-whitespace characters just before and after.
    first_sym, second_sym = pair
    new_sym = ''.join(pair)
    pattern = re.compile(r'(?<!\S)' + re.escape(f'{first_sym} {second_sym}') + r'(?!\S)')

    for j, _ in indices[pair].items():
        word, freq = vocab[j]
        x = ' '.join(word)
        x = pattern.sub(new_sym, x)
        word = tuple(x.split(' '))
        vocab[j] = (word, freq)



def learn_bpe():
    filename = 'shakespeare.txt'
    vocab = get_word_freq_dict(filename)

    # add '</w>' to the end of each word
    # and change the key from string to tuple(word[:-1], word[-1]+'</w>')
    vocab = dict(((tuple(word[:-1])+ (word[-1]+'</w>', )), freq) for word, freq in vocab.items())
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    # BPE
    n_bpe_operations = 3000
    tokens = []
    for i in range(n_bpe_operations):
        logger.info(f'BPE Operation: {i}')
        # get stats
        stats, indices = get_stats(vocab)

        # sort stats
        stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        pair = stats[0][0]
        logger.info(f'Pair: {pair}')

        # replace string
        replace_string(vocab, pair, indices)

        # add pair to tokens
        tokens.append(''.join(pair))

    tokens.sort(key=len, reverse=True)
    with open('bpe_tokens.txt', 'w') as f:
        f.write('\n'.join(tokens))


def _get_token(tokens: list[str], word: str):
    """
    Get the token for the given word.
    """
    for token in tokens:
        len_token = len(token)
        if word[:len_token] == token:
            return token
    return '[UNK]'


def encode(text: str):
    """
    Encode the given text using the BPE tokens.
    """
    with open('bpe_tokens.txt', 'r') as f:
        tokens = f.read().split('\n')

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # convert to lowercase
    text = text.lower()

    # split the text into words
    words = text.split()

    # add </w> to the end of each word
    words = [word + '</w>' for word in words]

    # encode each word
    encoded_words = []
    for word in words:
        while True:
            token = _get_token(tokens, word)
            encoded_words.append(token)
            word = word[len(token):]

            if token == '[UNK]' or word == '':
                break

    return encoded_words


def decode(encoded_text: list[str]):
    """
    Decode the given encoded text.
    """
    return ''.join(encoded_text).replace('</w>', ' ')


def main(args):
    if args.learn:
        learn_bpe()

    try:
        print('Press Ctrl+C to exit...')
        while True:
            if args.mode == 'encode':
                encoded_text = encode(input('Enter text: '))
                print(encoded_text)

            elif args.mode == 'decode':
                decoded_text = decode(eval(input('Enter encoded text: ')))
                print(decoded_text)
    except KeyboardInterrupt:
        print('\nExiting...')


if __name__ == '__main__':
    # create an argument parser with learn flag
    # and encode and decode modes
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn', action='store_true', help='Learn BPE tokens')
    parser.add_argument('--mode', type=str, default='encode', help='Mode: encode or decode')

    # parse the arguments
    args = parser.parse_args()

    main(args)