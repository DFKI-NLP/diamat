import argparse
import os
import ujson
import innvestigate
import numpy as np
import train
import util
from configparser import ConfigParser
from shutil import move
from tempfile import mkstemp


def get_model_and_analyser(sequence_length,
                           embedding_dim_target,
                           embedding_dim_source,
                           num_filters,
                           filter_sizes,
                           drop,
                           model_params,
                           analyser_name,
                           have_activation):
    """Constructs a model and initializes an analyser with it.
    :param sequence_length: The sequence length of an input text.
    :param embedding_dim_target: The word vector dimension of the target language.
    :param embedding_dim_source: The word vector dimension of the source language.
    :param num_filters: The number of convolution filters per n-gram.
    :param filter_sizes: The n-gram filter sizes.
    :param drop: The drop out probability.
    :param have_activation: Whether or not there should be a final non-linear activation.
    :param model_params: Location of the model weights.
    :param analyser_name: The name of the analyser (explainability method).
    :returns model, analyser: A model and its analyser.
    """
    model = train.construct_model(sequence_length=sequence_length,
                                  embedding_dim_target=embedding_dim_target,
                                  embedding_dim_source=embedding_dim_source,
                                  num_filters=num_filters,
                                  filter_sizes=filter_sizes,
                                  drop=drop,
                                  have_activation=have_activation)
    model.load_weights(model_params)
    analyser = innvestigate.create_analyzer(analyser_name, model, neuron_selection_mode="index")

    return model, analyser


class XLoader(train.Loader):
    """
    Loads batches of data (w/o the target).
    :param json_lines: JSON lines containing the training data.
    :param idx2vec: Index-to-word-vector dictionary.
    :param batch_size: Batch size __get_item__ returns.
    """

    def __init__(self, json_lines, idx2vec_target, idx2vec_source, batch_size=64):
        super().__init__(json_lines,
                         idx2vec_target=idx2vec_target,
                         idx2vec_source=idx2vec_source,
                         batch_size=batch_size,
                         random_order=False)  # human left (output idx 0), machine right (output idx 1)

    def __getitem__(self, index):
        batch = super().__getitem__(index)[0]  # only return X, not y
        return batch


def get_contributions(explanation):
    """Computes per word contributions.
    :param explanation: Numpy array containing contributions (column-wise, per word vector).
    :returns contributions: Contributions ot a classification decision, summarized and normalized.
    """
    contributions = np.sum(explanation, axis=2)[:, :, 0]
    max_val = np.abs(contributions).max()
    return contributions, max_val


if __name__ == '__main__':
    util.log("Explaining...")
    config = ConfigParser()
    config.read('./data/input/config.INI')

    parser = argparse.ArgumentParser(description='Bundle line separated corpora.')

    parser.add_argument('--explain_doc', type=str, default=config.get('EXPLANATION', 'explain_doc'),
                        help='Json lines containing data which should be labeled and explained.')
    parser.add_argument('--num_filters', type=int, default=config.getint('EXPLANATION', 'num_filters'),
                        help='The number of convolution filters per n-gram.')
    parser.add_argument('--filter_sizes', type=str, default=config.get('EXPLANATION', 'filter_sizes'),
                        help='The sizes of the convolution filters.')
    parser.add_argument('--drop', type=float, default=config.get('EXPLANATION', 'drop'),
                        help='The dropout probability.')
    parser.add_argument('--analyser_name', type=str, default=config.get('EXPLANATION', 'analyser_name'))
    parser.add_argument('--model_params', type=str, default=config.get('EXPLANATION', 'model_params'),
                        help='Pickle file to which model parameters will be saved.')
    parser.add_argument('--sequence_length', type=int, default=config.getint('EXPLANATION', 'sequence_length'),
                        help='The (maximum) sequence length of one input text (padded).')
    parser.add_argument('--embedding_dim_target', type=int, default=config.getint('EXPLANATION', 'embedding_dim_target'),
                        help='Word vector dimensions of the target language.')
    parser.add_argument('--embedding_dim_source', type=int, default=config.getint('EXPLANATION', 'embedding_dim_source'),
                        help='Word vector dimensions of the source language.')
    parser.add_argument('--batch_size', type=int, default=config.getint('EXPLANATION', 'batch_size'),
                        help='The batch size.')
    parser.add_argument('--idx2vec_target', type=str, default=config.get('EXPLANATION', 'idx2vec_target'),
                        help='Index to vec lookup in the target language.')
    parser.add_argument('--idx2vec_source', type=str, default=config.get('EXPLANATION', 'idx2vec_source'),
                        help='Index to vec lookup in the source language.')
    parser.add_argument('--train_doc', type=str, default=config.get('EXPLANATION', 'train_doc'),
                        help='Json lines containing the train split.')
    parser.add_argument('--have_activation', type=bool, default=config.getboolean('EXPLANATION', 'have_activation'),
                        help='Whether the model shall have a non-linear final activation.')
    parser.add_argument('--out_file', type=str, default=config.get('EXPLANATION', 'out_file'),
                        help='The output file')
    args = parser.parse_args()
    filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

    model, analyser = get_model_and_analyser(sequence_length=args.sequence_length,
                                             embedding_dim_target=args.embedding_dim_target,
                                             embedding_dim_source=args.embedding_dim_source,
                                             num_filters=args.num_filters,
                                             filter_sizes=filter_sizes,
                                             drop=args.drop, model_params=args.model_params,
                                             analyser_name=args.analyser_name,
                                             have_activation=args.have_activation)

    train_lines = util.load_lines(doc_path_in=args.train_doc)
    train_loader = XLoader(json_lines=train_lines,
                           idx2vec_target=args.idx2vec_target,
                           idx2vec_source=args.idx2vec_source,
                           batch_size=args.batch_size)

    analyser.fit_generator(train_loader)

    explain_doc = util.load_lines(doc_path_in=args.explain_doc)
    explain_loader = XLoader(json_lines=explain_doc,
                             idx2vec_target=args.idx2vec_target,
                             idx2vec_source=args.idx2vec_source,
                             batch_size=args.batch_size)

    max_abs_contribution = float("-inf")
    fout = open(args.out_file, 'w')
    for i in range(len(explain_loader)):
        batch = explain_loader.__getitem__(i)
        pred = model.predict(batch)  # ndarray (batch, 2)
        prediction = []
        for idx in np.arange(pred.shape[0]):
            prediction.append(pred[idx])
        analysis = analyser.analyze(batch, neuron_selection=1)
        contributions_human, max_val_human = get_contributions(analysis[0])
        # positive evidence that the machine is the input on the right in the human input on the left,
        # is in fact a class signal for the human class
        contributions_human = list(map(lambda cont: -1. * cont, contributions_human))
        contributions_machine, max_val_machine = get_contributions(analysis[1])
        contributions_source, max_val_source = get_contributions(analysis[2])
        max_abs_contribution = max_val_human if max_val_human > max_abs_contribution else max_abs_contribution
        max_abs_contribution = max_val_machine if max_val_machine > max_abs_contribution else max_abs_contribution
        max_abs_contribution = max_val_source if max_val_source > max_abs_contribution else max_abs_contribution
        line_numbers = explain_loader.line_numbers
        indices = np.arange(len(line_numbers))
        for idx, line_number in zip(indices, line_numbers):
            line = explain_doc[line_number]
            line['human']['prediction'] = float(prediction[idx][0])
            line['machine']['prediction'] = float(prediction[idx][1])
            line['human']['contributions'] = list(map(lambda x: float(x), contributions_human[idx]))
            line['machine']['contributions'] = list(map(lambda x: float(x), contributions_machine[idx]))
            line['source']['contributions'] = list(map(lambda x: float(x), contributions_source[idx]))
            fout.write(ujson.dumps(line) + os.linesep)
    fout.close()

    util.log('Normalizing...')
    fh, abs_path = mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        with open(args.out_file) as old_file:
            for line in old_file:
                jsonl = ujson.loads(line.strip())
                human_contributions = jsonl['human']['contributions']
                machine_contributions = jsonl['machine']['contributions']
                source_contributions = jsonl['source']['contributions']
                human_contributions = list(map(lambda cont: cont/max_abs_contribution,
                                               human_contributions))
                machine_contributions = list(map(lambda cont: cont/max_abs_contribution,
                                                 machine_contributions))
                source_contributions = list(map(lambda cont: cont/max_abs_contribution,
                                                 source_contributions))
                jsonl['human']['contributions'] = human_contributions
                jsonl['machine']['contributions'] = machine_contributions
                jsonl['source']['contributions'] = source_contributions
                new_file.write(ujson.dumps(jsonl) + os.linesep)
    # Remove original file
    os.remove(args.out_file)
    # Move new file
    move(abs_path, args.out_file)
    util.log('...done normalizing.')

    util.log("...done explaining. Maximum absolute contribution was {}.".format(max_abs_contribution))
