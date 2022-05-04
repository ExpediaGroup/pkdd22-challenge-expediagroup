"""
    Copyright 2022 Expedia, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from scipy.sparse import coo_matrix


class MarkovBaseline:

    def __init__(self, window, vocabulary):
        self.window = window
        self.vocabulary = vocabulary
        self._transition_matrix = None
        self.hotel2index = None

    def build_hotel2index(self):
        self.hotel2index = {hotel_id: index for index, hotel_id in enumerate(self.vocabulary, 1)}

    def calc_co_occurrence_matrix(self, clicks):
        """
        :param clicks: iterable which keeps the clicked items.
        :return:
        """
        row, col = [], []
        for session in clicks:
            session_indices = []
            for h in session:
                if h in self.hotel2index:
                    session_indices.append(self.hotel2index[h])
            for i, h1 in enumerate(session_indices[:-1]):
                window = session_indices[i + 1: i + self.window + 1]
                row.extend([h1, ] * len(window))
                col.extend(window)

        self._transition_matrix = coo_matrix(([1, ] * len(row), (row, col)),
                                             shape=(len(self.hotel2index) + 1, len(self.hotel2index) + 1)).tocsr()
        self._transition_matrix += self._transition_matrix.transpose()  # make symmetric

    def fit(self, X):
        """
        :param X:
        :return:
        """
        self.calc_co_occurrence_matrix(X)

    def predict_scores(self, X):
        """
        @:param X: the input tensor, these are the ids of the last clicked hotel. Shape is [batch_size].

        :return: scores of shape [batch_size, vocabulary_size]. It will just return the values of the Markov baseline
        scores. These will be the logits to be used in the evaluation of the metric.
        """
        return self._transition_matrix[X].todense()

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass
