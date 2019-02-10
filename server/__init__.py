import os
import re
import ujson

import numpy as np
import math
from flask import Flask, render_template, request
from werkzeug.contrib.cache import SimpleCache

cache = SimpleCache()


class RGB:
    def __init__(self, red, green, blue, score):
        self.red = red
        self.green = green
        self.blue = blue
        self.score = round(score, ndigits=3) if score is not None else score

    def __str__(self):
        return 'rgb({},{},{})'.format(self.red, self.green, self.blue)


class Sequence:
    def __init__(self, words, scores):
        assert (len(words) == len(scores))
        self.words = words
        self.scores = scores
        self.size = len(words)

    def words_rgb(self, gamma):
        rgbs = list(map(lambda tup: self.rgb(word=tup[0], score=tup[1], gamma=gamma), zip(self.words, self.scores)))
        return zip(self.words, rgbs)

    @staticmethod
    def gamma_correction(score, gamma):
        return np.sign(score) * np.power(np.abs(score), gamma)

    def rgb(self, word, score, gamma, reserved=[], threshold=0):
        assert not math.isnan(score), 'Score of word {} is NaN'.format(word)
        score = self.gamma_correction(score, gamma)
        if word in reserved:
            return RGB(0, 0, 0, None)
        if word == '<pad>':
            return RGB(0, 0, 0, None)
        if score >= threshold:
            r = str(int(255))
            g = str(int(255 * (1 - score)))
            b = str(int(255 * (1 - score)))
        else:
            b = str(int(255))
            r = str(int(255 * (1 + score)))
            g = str(int(255 * (1 + score)))
        return RGB(r, g, b, score)


class Explanation:  # TODO write doc

    @staticmethod
    def sm(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def predictions_to_softmax(predictions):
        arr = np.array([predictions['human'], predictions['machine']])
        softmax = Explanation.sm(arr)
        return {'human': softmax[0], 'machine': softmax[1]}

    def __init__(self, source, human, machine, predictions, index):  # predictions should be named logits
        assert (isinstance(source, Sequence))
        assert (isinstance(human, Sequence))
        assert (isinstance(machine, Sequence))
        assert (isinstance(predictions, dict))
        self.source = source
        self.source_text_wo_pad_token = ' '.join([word for word in self.source.words if word != '<pad>'])
        self.human = human
        self.human_text_wo_pad_token = ' '.join([word for word in self.human.words if word != '<pad>'])
        self.machine = machine
        self.machine_text_wo_pad_token = ' '.join([word for word in self.machine.words if word != '<pad>'])
        self.predictions = predictions
        self.softmax = Explanation.predictions_to_softmax(self.predictions)
        self.index = index

    @staticmethod
    def round(f, decimals=3):
        return np.round(f, decimals=decimals)


class Form:
    def __init__(self,
                 no_of_explanations=50,
                 indices=None,
                 regex_source=".*",
                 regex_machine=".*",
                 regex_human=".*",
                 range_machine_softmax=(0.0, 1.0),
                 range_human_softmax=(0.0, 1.0),
                 range_machine_logits=(-100.0, 100.0),
                 range_human_logits=(-100.0, 100.0),
                 sort_by='machine',
                 sort_key='softmax',
                 order='descending',
                 gamma=.5):
        assert (no_of_explanations >= 0)
        assert (range_machine_softmax[0] < range_machine_softmax[1])
        assert (range_human_softmax[0] < range_human_softmax[1])
        assert (range_machine_logits[0] < range_machine_logits[1])
        assert (range_human_logits[0] < range_human_logits[1])
        assert (sort_by in ["machine", "human", "random"])
        assert (order in ["ascending", "descending"])
        self.no_of_explanations = no_of_explanations
        self.indices = indices
        self.regex_source = regex_source
        self.regex_machine = regex_machine
        self.regex_human = regex_human
        self.range_machine_softmax = range_machine_softmax
        self.range_human_softmax = range_human_softmax
        self.range_machine_logits = range_machine_logits
        self.range_human_logits = range_human_logits
        self.sort_by = sort_by
        self.sort_key = sort_key
        self.order = order
        self.gamma = gamma

    def indices_to_string(self):
        if self.indices is None:
            return ""
        else:
            ', '.join(self.indices)


def read_explanations():  # TODO write doc
    print('Reading explanations upon server start...')
    site_root = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(site_root, "static/input", "explain.jsonl")
    explanations = []
    with open(json_url, 'r') as fin:
        for line in fin:
            explanations.append(jsonl_to_explanation(line))
    cache.set('explanations', explanations, timeout=60 * 24 * 365)
    print('... done reading explanations upon server start.')
    return explanations


def jsonl_to_explanation(jsonl):
    dictionary = ujson.loads(jsonl)

    index = dictionary['idx']
    predictions = {'human': dictionary['human']['prediction'],
                   'machine': dictionary['machine']['prediction']}

    human_tokens = dictionary['human']['tokens']
    human_contributions = dictionary['human']['contributions']
    human_seq = Sequence(human_tokens, human_contributions)

    machine_tokens = dictionary['machine']['tokens']
    machine_contributions = dictionary['machine']['contributions']
    machine_seq = Sequence(machine_tokens, machine_contributions)

    source_tokens = dictionary['source']['tokens']
    source_contributions = dictionary['source']['contributions']
    source_seq = Sequence(source_tokens, source_contributions)

    explanation = Explanation(source=source_seq,
                              human=human_seq,
                              machine=machine_seq,
                              predictions=predictions,
                              index=index)

    return explanation


def sort_explanations(form, explanations):  # TODO write docs
    total_len = len(explanations)

    if form.indices is not None:
        explanations = filter(lambda explanation: explanation.index in form.indices, explanations)

    regex_source = form.regex_source
    pattern_source = re.compile(regex_source)

    regex_machine = form.regex_machine
    pattern_machine = re.compile(regex_machine)

    regex_human = form.regex_human
    pattern_human = re.compile(regex_human)

    explanations = filter(lambda explanation: pattern_source.match(explanation.source_text_wo_pad_token), explanations)
    explanations = filter(lambda explanation: pattern_machine.match(explanation.machine_text_wo_pad_token), explanations)
    explanations = filter(lambda explanation: pattern_human.match(explanation.human_text_wo_pad_token), explanations)

    # filter ranges according to softmax ranges
    explanations = filter(
        lambda explanation: form.range_machine_softmax[0] <= explanation.softmax['machine'] <=
                            form.range_machine_softmax[1], explanations)

    explanations = filter(
        lambda explanation: form.range_human_softmax[0] <= explanation.softmax['human'] <=
                            form.range_human_softmax[1], explanations)

    # filter according to logit ranges
    explanations = filter(
        lambda explanation: form.range_machine_logits[0] <= explanation.predictions['machine'] <=
                            form.range_machine_logits[1], explanations)
    explanations = filter(
        lambda explanation: form.range_human_logits[0] <= explanation.predictions['human'] <=
                            form.range_human_logits[1], explanations)

    # sort by machine or human, logits or softmax
    if form.sort_by == "machine":
        if form.sort_key == "logits":
            explanations = list(sorted(explanations, key=lambda explanation: explanation.predictions['machine'],
                                       reverse=form.order == 'descending'))
        else:
            explanations = list(sorted(explanations, key=lambda explanation: explanation.softmax['machine'],
                                       reverse=form.order == 'descending'))

    elif form.sort_by == "human":
        if form.sort_key == "logits":
            explanations = list(sorted(explanations, key=lambda explanation: explanation.predictions['human'],
                                       reverse=form.order == 'descending'))
        else:
            explanations = list(sorted(explanations, key=lambda explanation: explanation.softmax['human'],
                                       reverse=form.order == 'descending'))
    else:
        explanations = list(explanations)

    number_of_hits = len(explanations)

    explanations = explanations[:int(form.no_of_explanations)]
    return explanations, total_len, number_of_hits


# on server startup read and cache explanations
read_explanations()


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/', methods=['GET', 'POST'])
    def items():
        if request.method == 'GET':
            explanations = cache.get('explanations')
            form = Form()
            explanations, total_len, number_of_hits = sort_explanations(form, explanations)
            return render_template('items.html',
                                   form=Form(),
                                   explanations=explanations,
                                   number_of_hits=number_of_hits,
                                   total_len=total_len)

        if request.method == 'POST':
            if request.form['submit_button'] == "Update":
                no_of_explanations = int(float(request.form['noofexplanations']))
                no_of_explanations = no_of_explanations if no_of_explanations >= 0 else 0
                indices = request.form['filter_indices']
                if bool(re.match('(\d+$|[\d+,\s*]+\d+$)', indices)):
                    indices = list(map(lambda s: int(s), indices.split(',')))
                else:
                    indices = None
                regex_source = request.form['regex_source']
                regex_machine = request.form['regex_machine']
                regex_human = request.form['regex_human']
                range_machine_bottom_softmax = float(request.form['range_machine_bottom_softmax'])
                range_machine_top_softmax = float(request.form['range_machine_top_softmax'])
                range_human_bottom_softmax = float(request.form['range_human_bottom_softmax'])
                range_human_top_softmax = float(request.form['range_human_top_softmax'])
                range_machine_bottom_logits = float(request.form['range_machine_bottom_logits'])
                range_machine_top_logits = float(request.form['range_machine_top_logits'])
                range_human_bottom_logits = float(request.form['range_human_bottom_logits'])
                range_human_top_logits = float(request.form['range_human_top_logits'])
                sort_by = request.form['sortby']
                sort_key = request.form['sortkey']
                order = request.form['order']
                gamma = float(request.form['gamma'])
                form = Form(
                    no_of_explanations=no_of_explanations,
                    indices=indices,
                    regex_source=regex_source,
                    regex_machine=regex_machine,
                    regex_human=regex_human,
                    range_machine_softmax=(range_machine_bottom_softmax, range_machine_top_softmax),
                    range_human_softmax=(range_human_bottom_softmax, range_human_top_softmax),
                    range_machine_logits=(range_machine_bottom_logits, range_machine_top_logits),
                    range_human_logits=(range_human_bottom_logits, range_human_top_logits),
                    sort_by=sort_by,
                    sort_key=sort_key,
                    order=order,
                    gamma=gamma
                )
                explanations = cache.get('explanations')
                explanations, total_len, number_of_hits = sort_explanations(form, explanations)
                return render_template('items.html',
                                       form=form,
                                       explanations=explanations,
                                       number_of_hits=number_of_hits,
                                       total_len=total_len)

            else:
                explanations = cache.get('explanations')
                form = Form()
                explanations, total_len, number_of_hits = sort_explanations(form, explanations)
                return render_template('items.html',
                                       form=Form(),
                                       explanations=explanations,
                                       number_of_hits=number_of_hits,
                                       total_len=total_len)  ##explanations[0].source

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, use_reloader=False)
