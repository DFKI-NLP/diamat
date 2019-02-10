import argparse
import pickle
import ujson
import os
import numpy as np
import spacy
from util import log
from configparser import ConfigParser
from gensim.models import KeyedVectors


class Encoder:
    """Encodes natural language.
    :param vocab_size: The vocabulary size to consider (sorted by frequency).
    :param word_vectors_file: Path to file containing word2vec formatted word vectors.
    :param word_vector_dim: The word vector dimension.
    :param language: Spacy supported language.
    """

    def __init__(self,
                 vocab_size,
                 word_vectors_file,
                 word_vector_dim,
                 language='en'):
        self.vocab_size = vocab_size
        self.word_vectors_file = word_vectors_file
        self.word_vectors = KeyedVectors.load_word2vec_format(self.word_vectors_file,
                                                              limit=self.vocab_size)
        self.word_vector_dim = word_vector_dim,
        self.token2idx, self.idx2vec = self.get_dictionaries()
        self.nlp = spacy.load(language, disable=['tagger', 'parser', 'ner', 'textcat'])

    def get_dictionaries(self):
        """Returns token2idx and idx2vec dictionaries of this encoder."""
        assert (len(self.word_vectors.vocab) > 0)
        token_2_idx = {'<unk>': 0, '<pad>': 1}
        idx_2_vec = {0: np.zeros(self.word_vector_dim), 1: np.zeros(self.word_vector_dim)}
        idx = 2
        for token in self.word_vectors.vocab:
            if idx > self.vocab_size:
                break
            token_2_idx[token] = idx
            idx_2_vec[idx] = self.word_vectors[token]
            idx_2_vec[0] = np.add(idx_2_vec[0], self.word_vectors[token])
            idx = idx + 1
        idx_2_vec[0] = idx_2_vec[0] / (idx - 2)  # unk is average over all vectors
        return token_2_idx, idx_2_vec

    def tokenize(self, text):
        """Tokenizes text, returns list of tokens.
        :param text: String to tokenize.
        """
        doc = self.nlp(text)
        res = [token.text for token in doc]
        return res

    def string_to_indices(self, text, seq_len=-1):
        """Tokenizes text and returns list with indices
        :param text: String to tokenize.
        :param seq_len: Maximum sequence length (cut or padded).
        """
        tokens = self.tokenize(text)

        if seq_len > 0:
            if len(tokens) < seq_len:
                pads = ['<pad>'] * (seq_len - len(tokens))
                tokens = tokens + pads
            else:
                tokens = tokens[:seq_len]

        indices = []
        for token in tokens:
            try:
                indices.append(self.token2idx[token])
            except KeyError:
                indices.append(0)
        return indices, tokens

    def indices_to_vec(self, indices):
        """Concatenates word vectors into embedding matrix.
        :param indices: Indices to vectorize.
        :returns vectors: Vectorized indices.
        """
        vectors = None
        for idx in indices:
            if vectors is None:
                vectors = self.idx2vec[idx]
            else:
                vectors = np.concatenate(self.idx2vec[idx])
        return vectors


def bundle(doc_path_in_machine,
           doc_path_in_human,
           doc_path_in_source,
           encoder_source,
           encoder_target,
           doc_path_out,
           seq_len=150,
           prefix=-1,
           print_progress_every=10000):
    """
    Bundles human, machine and source lines alongside word indices in json lines.
    :param doc_path_in_machine: File containing line separated machine translations.
    :param doc_path_in_human: File containing line separated human translations.
    :param doc_path_in_source: File containing line separated source.
    :param encoder_source: Encoder which holds token-to-index dictionary of the source language.
    :param encoder_target: Encoder which holds token-to-index dictionary of the target language.
    :param doc_path_out: File to which json lines are written.
    :param seq_len: The sequence length to which each line is cut or padded.
    :param prefix:  Process <prefix> number of lines. Ignore if prefix less than zero.
    :param print_progress_every: Log progress every <print_progress_every> lines.
    :return line: The number of lines written.
    """
    machine_fin = open(doc_path_in_machine)
    human_fin = open(doc_path_in_human)
    source_fin = open(doc_path_in_source)
    bundle_fout = open(doc_path_out, 'a')

    machine_line = machine_fin.readline()
    human_line = human_fin.readline()
    source_line = source_fin.readline()

    line = 0
    while machine_line and human_line and source_line:
        if 1 <= prefix <= line:
            break
        else:
            line = line + 1

        if (line + 1) % print_progress_every == 0:
            bundle_fout.flush()
            log('Processed line {}'.format((line + 1)))

        jsonl = {'idx': line, 'machine': {}, 'human': {}, 'source': {}}

        machine_idxs, machine_tokens = encoder_target.string_to_indices(machine_line.strip(), seq_len=seq_len)
        jsonl['machine']['indices'] = machine_idxs
        jsonl['machine']['tokens'] = machine_tokens

        human_idxs, human_tokens = encoder_target.string_to_indices(human_line.strip(), seq_len=seq_len)
        jsonl['human']['indices'] = human_idxs
        jsonl['human']['tokens'] = human_tokens

        source_idxs, source_tokens = encoder_source.string_to_indices(source_line.strip(), seq_len=seq_len)
        jsonl['source']['indices'] = source_idxs
        jsonl['source']['tokens'] = source_tokens

        bundle_fout.write(ujson.dumps(jsonl) + os.linesep)

        machine_line = machine_fin.readline()
        human_line = human_fin.readline()
        source_line = source_fin.readline()
    return line


if __name__ == '__main__':
    log("Preprocessing...")
    config = ConfigParser()
    config.read('./data/input/config.INI')
    parser = argparse.ArgumentParser(description='Bundle line separated corpora.')

    parser.add_argument('--vocab_size_source', type=int, default=config.get('PREPROCESSING', 'vocab_size_source'),
                        help='Vocabulary size of the source language.')
    parser.add_argument('--word_vectors_source', type=str, default=config.get('PREPROCESSING', 'word_vectors_source'),
                        help='word2vec formatted file w/ source language word vectors.')
    parser.add_argument('--embedding_dim_source', type=int, default=config.get('PREPROCESSING', 'embedding_dim_source'),
                        help='Word vector dimension of source language word vectors.')
    parser.add_argument('--token2idx_source', type=str, default=config.get('PREPROCESSING', 'token2idx_source'),
                        help='Save token2idx dictionary of source language here.')
    parser.add_argument('--idx2vec_source', type=str, default=config.get('PREPROCESSING', 'idx2vec_source'),
                        help='Save idx2vec dictionary of source language here.')
    parser.add_argument('--vocab_size_target', type=int, default=config.get('PREPROCESSING', 'vocab_size_target'),
                        help='Vocabulary size of the target language.')
    parser.add_argument('--word_vectors_target', type=str, default=config.get('PREPROCESSING', 'word_vectors_target'),
                        help='word2vec formatted file w/ target language word vectors.')
    parser.add_argument('--embedding_dim_target', type=int, default=config.get('PREPROCESSING', 'embedding_dim_target'),
                        help='Word vector dimension of target language word vectors.')
    parser.add_argument('--token2idx_target', type=str, default=config.get('PREPROCESSING', 'token2idx_target'),
                        help='Save token2idx dictionary of target language here.')
    parser.add_argument('--idx2vec_target', type=str, default=config.get('PREPROCESSING', 'idx2vec_target'),
                        help='Save idx2vec dictionary of target language here.')
    parser.add_argument('--machine', type=str, default=config.get('PREPROCESSING', 'machine'),
                        help='Machine translations.')
    parser.add_argument('--human', type=str, default=config.get('PREPROCESSING', 'human'), help='Human translations.')
    parser.add_argument('--source', type=str, default=config.get('PREPROCESSING', 'source'), help='The source file.')
    parser.add_argument('--bundle', type=str, default=config.get('PREPROCESSING', 'bundle'),
                        help='Output path.')
    parser.add_argument('--prefix', type=int, default=config.getint('PREPROCESSING', 'prefix'),
                        help='Only preprocess prefix lines from each corpus.')
    parser.add_argument('--sequence_length', type=int, default=config.getint('PREPROCESSING', 'sequence_length'),
                        help='Cut or pad each input to seq_length.')

    args = parser.parse_args()

    enc_source = Encoder(args.vocab_size_source,
                         args.word_vectors_source,
                         args.embedding_dim_source,
                         language='de')

    pickle.dump(enc_source.token2idx, open(args.token2idx_source, 'wb'))
    pickle.dump(enc_source.idx2vec, open(args.idx2vec_source, 'wb'))

    enc_target = Encoder(args.vocab_size_target,
                         args.word_vectors_target,
                         args.embedding_dim_target)

    pickle.dump(enc_target.token2idx, open(args.token2idx_target, 'wb'))
    pickle.dump(enc_target.idx2vec, open(args.idx2vec_target, 'wb'))

    no_of_lines = bundle(doc_path_in_machine=args.machine,
                         doc_path_in_human=args.human,
                         doc_path_in_source=args.source,
                         encoder_target=enc_target,
                         encoder_source=enc_source,
                         doc_path_out=args.bundle,
                         seq_len=args.sequence_length,
                         prefix=args.prefix)

    log("...done preprocessing.")
