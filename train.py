import argparse
import pickle
import random
import util
import keras
import numpy as np
from keras import Input, Model
from configparser import ConfigParser
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.optimizers import Adam


# credit: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class Loader(keras.utils.Sequence):
    """
    Loads batches of training data.
    :param json_lines: JSON lines containing the training data.
    :param idx2vec_target: Index-to-word-vector dictionary of the target language.
    :param idx2vec_source: Index-to-word-vector dictionary of the source language.
    :param batch_size: Batch size __get_item__ returns.
    :param shuffle: Whether or not to shuffle after each epoch.
    :param random_order: Whether or not to randomize the input layer that the machine input is passed to.
    """

    def __init__(self, json_lines, idx2vec_target, idx2vec_source, batch_size=64, shuffle=True, random_order=True):
        assert (batch_size % 2 == 0)  # two inputs (human and machine) per line
        self.json_lines = json_lines
        self.idx2vec_target = pickle.load(open(idx2vec_target, 'rb'))
        self.idx2vec_source = pickle.load(open(idx2vec_source, 'rb'))
        self.size = len(self.json_lines)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.__HUMAN = 0.
        self.__MACHINE = 1.
        self.random_order = random_order

    def __data_generation(self, line_numbers):
        lines = [self.json_lines[i] for i in line_numbers]

        indices_source = [line['source']['indices'] for line in lines]
        X_source = np.array([self.__embed(idxs_source, self.idx2vec_source) for idxs_source in indices_source])
        X_source = np.expand_dims(X_source, axis=3)

        indices_human = [line['human']['indices'] for line in lines]
        indices_machine = [line['machine']['indices'] for line in lines]

        indices_left = []
        indices_right = []
        y = []
        for (human, machine) in zip(indices_human, indices_machine):
            if bool(random.getrandbits(1)) and self.random_order:  # randomly assign left, right
                indices_left.append(machine)
                indices_right.append(human)
                y.append(np.array([self.__MACHINE, self.__HUMAN]))
            else:
                indices_left.append(human)
                indices_right.append(machine)
                y.append(np.array([self.__HUMAN, self.__MACHINE]))

        X_left = np.array([self.__embed(idxs_left, self.idx2vec_target) for idxs_left in indices_left])
        X_left = np.expand_dims(X_left, axis=3)
        X_right = np.array([self.__embed(idxs_right, self.idx2vec_target) for idxs_right in indices_right])
        X_right = np.expand_dims(X_right, axis=3)
        y = np.array(y)

        return [X_left, X_right, X_source], y

    @staticmethod
    def __embed(indices, idx2vec):
        res = None
        for idx in indices:
            vec = idx2vec[idx]
            vec = np.reshape(vec, (1, -1))
            if res is None:
                res = vec
            else:
                res = np.concatenate((res, vec))
        return res

    def __getitem__(self, index):
        self.line_numbers = self.indices[(index * self.batch_size):((index + 1) * self.batch_size)]
        X, y = self.__data_generation(line_numbers=self.line_numbers)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        res = int(np.floor(self.size / self.batch_size))
        return res


# credit: https://raw.githubusercontent.com/bhaveshoswal/CNN-text-classification-keras/master/model.py
def construct_model(sequence_length,
                    embedding_dim_target,
                    embedding_dim_source,
                    num_filters=150,
                    filter_sizes=[3, 4, 5],
                    drop=.5,
                    have_activation=True):
    """
    Constructs a CNN text classifier.
    :param sequence_length: The sequence length of an input text.
    :param embedding_dim_target: The word vector dimension of the target langauge.
    :param embedding_dim_source: The word vector dimension of the source langauge.
    :param num_filters: The number of convolution filters per n-gram.
    :param filter_sizes: The n-gram filter sizes.
    :param drop: The drop out probability.
    :param have_activation: Whether or not there should be a final non-linear activation.
    :return: A CNN text classifier.
    """
    right = Input(shape=(sequence_length, embedding_dim_target, 1), dtype='float32')
    left = Input(shape=(sequence_length, embedding_dim_target, 1), dtype='float32')
    source = Input(shape=(sequence_length, embedding_dim_source, 1), dtype='float32')

    conv_0_right = Conv2D(num_filters,  # TODO next lines are repeated below
                          kernel_size=(filter_sizes[0], embedding_dim_target),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          input_shape=(sequence_length, embedding_dim_target, 1))(right)
    conv_1_right = Conv2D(num_filters,
                          kernel_size=(filter_sizes[1], embedding_dim_target),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          input_shape=(sequence_length, embedding_dim_target, 1))(right)
    conv_2_right = Conv2D(num_filters,
                          kernel_size=(filter_sizes[2], embedding_dim_target),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          input_shape=(sequence_length, embedding_dim_target, 1))(right)

    maxpool_0_right = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                                strides=(1, 1),
                                padding='valid')(conv_0_right)
    maxpool_1_right = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                                strides=(1, 1),
                                padding='valid')(conv_1_right)
    maxpool_2_right = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                                strides=(1, 1),
                                padding='valid')(conv_2_right)

    concatenated_tensor_right = Concatenate(axis=1)([maxpool_0_right, maxpool_1_right, maxpool_2_right])
    flatten_right = Flatten()(concatenated_tensor_right)

    conv_0_left = Conv2D(num_filters,  # TODO this is repetitive
                         kernel_size=(filter_sizes[0], embedding_dim_target),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu',
                         input_shape=(sequence_length, embedding_dim_target, 1))(left)
    conv_1_left = Conv2D(num_filters,
                         kernel_size=(filter_sizes[1], embedding_dim_target),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu',
                         input_shape=(sequence_length, embedding_dim_target, 1))(left)
    conv_2_left = Conv2D(num_filters,
                         kernel_size=(filter_sizes[2], embedding_dim_target),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu',
                         input_shape=(sequence_length, embedding_dim_target, 1))(left)

    maxpool_0_left = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                               strides=(1, 1),
                               padding='valid')(conv_0_left)
    maxpool_1_left = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                               strides=(1, 1),
                               padding='valid')(conv_1_left)
    maxpool_2_left = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                               strides=(1, 1),
                               padding='valid')(conv_2_left)

    concatenated_tensor_left = Concatenate(axis=1)([maxpool_0_left, maxpool_1_left, maxpool_2_left])
    flatten_left = Flatten()(concatenated_tensor_left)

    conv_0_source = Conv2D(num_filters,  # TODO this is repetitive
                           kernel_size=(filter_sizes[0], embedding_dim_source),
                           padding='valid',
                           kernel_initializer='normal',
                           activation='relu',
                           input_shape=(sequence_length, embedding_dim_source, 1))(source)
    conv_1_source = Conv2D(num_filters,
                           kernel_size=(filter_sizes[1], embedding_dim_source),
                           padding='valid',
                           kernel_initializer='normal',
                           activation='relu',
                           input_shape=(sequence_length, embedding_dim_source, 1))(source)
    conv_2_source = Conv2D(num_filters,
                           kernel_size=(filter_sizes[2], embedding_dim_source),
                           padding='valid',
                           kernel_initializer='normal',
                           activation='relu',
                           input_shape=(sequence_length, embedding_dim_source, 1))(source)

    maxpool_0_source = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                                 strides=(1, 1),
                                 padding='valid')(conv_0_source)
    maxpool_1_source = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                                 strides=(1, 1),
                                 padding='valid')(conv_1_source)
    maxpool_2_source = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                                 strides=(1, 1),
                                 padding='valid')(conv_2_source)

    concatenated_tensor_source = Concatenate(axis=1)([maxpool_0_source, maxpool_1_source, maxpool_2_source])
    flatten_source = Flatten()(concatenated_tensor_source)

    concatenated = Concatenate(axis=1)([flatten_left, flatten_right, flatten_source])
    dropout = Dropout(drop)(concatenated)

    output = Dense(units=2, activation='softmax' if have_activation else None)(dropout)
    model = Model(inputs=[left, right, source], outputs=output)

    return model


def train(sequence_length,
          embedding_dim_target,
          embedding_dim_source,
          num_filters,
          filter_sizes,
          drop,
          train_doc_path_in,
          val_doc_path_in,
          idx2vec_target,
          idx2vec_source,
          model_params,
          batch_size,
          epochs=5,
          max_queue_size = 10,
          use_multiprocessing = False,
          workers = 1):
    """
    Trains a CNN text classifier.
    :param sequence_length: The sequence length of a text input.
    :param embedding_dim_target: The dimension of a word vector in the target language.
    :param embedding_dim_source: The dimension of a word vector in the source language.
    :param num_filters: The number of convolution filters per n-gram.
    :param filter_sizes: The n-gram filter sizes.
    :param drop: The drop out probability.
    :param train_doc_path_in: File containing json lines to train with.
    :param val_doc_path_in: File containing json lines to validate with.
    :param idx2vec_target: A dictionary mapping word indices onto word vectors in the target language.
    :param idx2vec_source: A dictionary mapping word indices onto word vectors in the source language.
    :param model_params: File in which the model weights are saved.
    :param batch_size: The batch size.
    :param epochs: The number of epochs.
    """

    json_lines_train = util.load_lines(doc_path_in=train_doc_path_in)

    train_loader = Loader(json_lines=json_lines_train,
                          idx2vec_target=idx2vec_target,
                          idx2vec_source=idx2vec_source,
                          batch_size=batch_size)

    json_lines_test = util.load_lines(doc_path_in=val_doc_path_in)

    val_loader = Loader(json_lines=json_lines_test,
                        idx2vec_target=idx2vec_target,
                        idx2vec_source=idx2vec_source,
                        batch_size=batch_size)

    model = construct_model(sequence_length=sequence_length,
                            embedding_dim_target=embedding_dim_target,
                            embedding_dim_source=embedding_dim_source,
                            num_filters=num_filters,
                            filter_sizes=filter_sizes,
                            drop=drop)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # TODO consume arguments from config
    checkpoint = ModelCheckpoint(model_params, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])  # TODO consume arguments from config

    print(model.summary())

    earlystopping = EarlyStopping(monitor='val_acc', patience=2, mode='auto')  # TODO consume arguments from config

    #callbacks = [checkpoint, earlystopping]
    # no checkpoints due to issue https://github.com/keras-team/keras/issues/11101
    callbacks = [earlystopping]

    model.fit_generator(
        generator=train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        use_multiprocessing=use_multiprocessing,
        workers=workers)

    # only save final model after earlystopping, due to issue https://github.com/keras-team/keras/issues/11101
    model.save(model_params)


if __name__ == '__main__':
    util.log("Training...")
    config = ConfigParser()
    config.read('./data/input/config.INI')

    parser = argparse.ArgumentParser(description='Bundle line separated corpora.')

    parser.add_argument('--sequence_length', type=int, default=config.getint('TRAINING', 'sequence_length'),
                        help='The (maximum) sequence length of one input text (padded).')
    parser.add_argument('--embedding_dim_target', type=int, default=config.get('TRAINING', 'embedding_dim_target'),
                        help='Word vector dimension of the target language.')
    parser.add_argument('--embedding_dim_source', type=int, default=config.get('TRAINING', 'embedding_dim_source'),
                        help='Word vector dimension of the source language.')
    parser.add_argument('--num_filters', type=int, default=config.getint('TRAINING', 'num_filters'),
                        help='The number of convolution filters.')
    parser.add_argument('--filter_sizes', type=str, default=config.get('TRAINING', 'filter_sizes'),
                        help='The sizes of the convolution filters.')
    parser.add_argument('--drop', type=float, default=config.get('TRAINING', 'drop'), help='The dropout probability.')
    parser.add_argument('--training_doc', type=str, default=config.get('TRAINING', 'training_doc'),
                        help='Training json lines.')
    parser.add_argument('--val_doc', type=str, default=config.get('TRAINING', 'val_doc'),
                        help='Validation json lines.')
    parser.add_argument('--idx2vec_target', type=str, default=config.get('TRAINING', 'idx2vec_target'),
                        help='Load index-to-word-vector dictionary of the target language from here.')
    parser.add_argument('--idx2vec_source', type=str, default=config.get('TRAINING', 'idx2vec_source'),
                        help='Load index-to-word-vector dictionary of the source language from here.')
    parser.add_argument('--batch_size', type=int, default=config.getint('TRAINING', 'batch_size'),
                        help='The batch size.')
    parser.add_argument('--epochs', type=int, default=config.getint('TRAINING', 'epochs'),
                        help='The maximum number of training epochs.')
    parser.add_argument('--model_params', type=str, default=config.get('TRAINING', 'model_params'),
                        help='Pickle file to which model parameters will be saved.')
    parser.add_argument('--max_queue_size', type=int, default=config.get('TRAINING', 'max_queue_size'),
                        help='Maximum number of batches to load into queue.')
    parser.add_argument('--use_multiprocessing', type=bool,
                        default=config.getboolean('TRAINING', 'use_multiprocessing'),
                        help='Whether or not to use multiprocessing during training.')
    parser.add_argument('--workers', type=bool, default=config.getint('TRAINING', 'workers'),
                        help='The number of workers to use during training.')

    args = parser.parse_args()
    filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

    train(sequence_length=args.sequence_length,
          embedding_dim_target=args.embedding_dim_target,
          embedding_dim_source=args.embedding_dim_source,
          num_filters=args.num_filters,
          filter_sizes=filter_sizes,
          drop=args.drop,
          train_doc_path_in=args.training_doc,
          val_doc_path_in=args.val_doc,
          idx2vec_target=args.idx2vec_target,
          idx2vec_source=args.idx2vec_source,
          batch_size=args.batch_size,
          epochs=args.epochs,
          model_params=args.model_params,
          max_queue_size=args.max_queue_size,
          use_multiprocessing=args.use_multiprocessing,
          workers=args.workers)

    util.log("...done training.")
