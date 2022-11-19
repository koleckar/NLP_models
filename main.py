import os.path
import wget

from abc import ABC, abstractmethod

import numpy as np
import plotly.express as px
from sklearn.decomposition import TruncatedSVD

import torch
import scipy.sparse

rng = np.random.default_rng()


# todo: tokenize with word_tokenizer !?!
#   from nltk import word_tokenize
#   import nltk
#   nltk.download('punkt')


class Model(ABC):

    @abstractmethod
    def create_model(self, corpus: list):
        pass


class TriGramModel(Model):
    """
       2st order Markov (tri-gram) language model.
       """

    def __init__(self, vocabulary: set, corpus: list, smooth: bool):
        self.vocabulary = vocabulary

        self.model = scipy.sparse.lil_matrix((len(vocabulary) * len(vocabulary), len(vocabulary)), dtype=np.int8)
        self.smooth = True

        self.token_to_idx = {}
        self.idx_to_token = {}
        for idx, token in enumerate(vocabulary):
            if token == "<UNK>":
                continue
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        self.create_model(corpus, smooth)

    def create_model(self, corpus: list, smooth: bool):
        max_idx = -1
        for tokens in corpus:
            for i in range(2, len(tokens)):
                idx1 = self.token_to_idx[tokens[i - 2]]
                idx2 = self.token_to_idx[tokens[i - 1]]
                idx3 = self.token_to_idx[tokens[i]]
                m = np.amax([idx1, idx2, idx3])
                max_idx = m if m > max_idx else max_idx
                self.model[idx1 * len(self.vocabulary) + idx2, idx3] += 1

        self.model_by_maximum_likelihood(corpus)
        # self.model[np.isnan(self.model)] = 0
        # assert np.sum(np.isnan(self.model)) == 0

    def model_by_maximum_likelihood(self, corpus: list):
        cols_sums = self.model.sum(axis=1)
        x = np.where(cols_sums > 0)
        self.model[x] = self.model[x] / cols_sums[x]

    def get_prob(self, i, j, k):
        if self.smooth:
            cols_sum = self.model[i * len(self.vocabulary) + j, :].sum()
            return (self.model[i * len(self.vocabulary) + j, k] + 1) / (cols_sum + len(self.vocabulary))
        else:
            return self.model[i * len(self.vocabulary) + j, k]

    def get_token_idx(self, token):
        if self.token_to_idx.get(token) is None:
            return self.token_to_idx["<UNK>"]
        else:
            return self.token_to_idx[token]


def get_vocabulary_and_corpus_trigram(path: str):
    vocabulary = set()
    corpus = []

    for line in open(path, encoding='utf-8'):
        tokens = line.rstrip().lower().split()
        if not tokens:
            continue
        tokens = ['<ss>'] + ['<s>'] + tokens + ['</s>']
        vocabulary.update(tokens)
        corpus.append(tokens)

    return vocabulary, corpus


# todo remove tokenizing from models.
# todo remove unsupported libraries !
class TokenIndexer:
    def __init__(self, vocabulary, include_unk):
        self.token_to_idx = {}
        self.idx_to_token = {}

        unk = "<UNK>" if include_unk else ""
        for idx, token in enumerate(vocabulary.union(unk)):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def get_token_idx(self, token):
        if self.token_to_idx.get(token) is None:
            return self.token_to_idx["<UNK>"]
        else:
            return self.token_to_idx[token]

    def get_idx_token(self, idx):
        if self.idx_to_token.get(idx) is None:
            return self.idx_to_token["<UNK>"]
        else:
            return self.idx_to_token[idx]


class BiGramModel(Model):
    """
    1st order Markov (bi-gram) language model.
    """

    def __init__(self, vocabulary: set, corpus: list, smooth: bool):
        self.vocabulary = vocabulary

        self.model = np.zeros([len(vocabulary) + 1, len(vocabulary) + 1])  # +1 for <UNK>

        self.token_to_idx = {}
        self.idx_to_token = {}
        for idx, token in enumerate(vocabulary.union(["<UNK>"])):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        self.create_model(corpus, smooth)

    def create_model(self, corpus: list, smooth: bool):
        for tokens in corpus:
            for i in range(1, len(tokens)):
                self.model[self.token_to_idx[tokens[i - 1]], self.token_to_idx[tokens[i]]] += 1

        if smooth:
            self.model_by_laplace_estimate(corpus)
        else:
            self.model_by_maximum_likelihood(corpus)
            self.model[np.isnan(self.model)] = 0
        assert np.sum(np.isnan(self.model)) == 0

    def model_by_maximum_likelihood(self, corpus: list):
        self.model /= np.sum(self.model, axis=1).reshape(-1, 1)

    def model_by_laplace_estimate(self, corpus: list):
        self.model += 1
        self.model /= np.sum(self.model, axis=1).reshape(-1, 1)

    def get_token_idx(self, token):
        if self.token_to_idx.get(token) is None:
            return self.token_to_idx["<UNK>"]
        else:
            return self.token_to_idx[token]


def get_log_probability(sentence: list, model: BiGramModel) -> float:
    prob_log = 0
    n = len(sentence)
    for i in range(1, n):
        idx1 = model.get_token_idx(sentence[i - 1])
        idx2 = model.get_token_idx(sentence[i])
        p = np.log(model.model[idx1, idx2])
        prob_log += p
    return prob_log


def get_probability(sentence: list, model: BiGramModel) -> float:
    prob = 1
    n = len(sentence)
    for i in range(1, n):
        idx1 = model.get_token_idx(sentence[i - 1])
        idx2 = model.get_token_idx(sentence[i])
        p = model.model[idx1, idx2]
        prob *= p
    return prob


def get_perplexity(sentence: list, model: Model) -> float:
    n = len(sentence)
    perp = 1
    for i in range(1, n):
        idx1 = model.get_token_idx(sentence[i - 1])
        idx2 = model.get_token_idx(sentence[i])
        perp *= 1 / model.model[idx1, idx2]
    return np.power(perp, 1 / n)


def get_vocabulary_and_corpus(path: str):
    vocabulary = set()
    corpus = []

    for line in open(path, encoding='utf-8'):
        tokens = line.rstrip().lower().split()
        if not tokens:
            continue
        tokens = ['<s>'] + tokens + ['</s>']
        vocabulary.update(tokens)
        corpus.append(tokens)

    return vocabulary, corpus


class MarkovClassifier:
    """
    Naive Bayes classifier using Markov models.
    """

    def __init__(self, prior_1: int, prior_2: int, model_1: Model, model_2: Model,
                 name_class_1=None, name_class_2=None):
        self.prior_1 = prior_1
        self.prior_2 = prior_2
        self.model_1 = model_1
        self.model_2 = model_2
        self.name_class_1 = name_class_1
        self.name_class_2 = name_class_2

    def classify(self, sentence: list):
        p_1 = np.log(self.prior_1) + get_log_probability(sentence, self.model_1)
        p_2 = np.log(self.prior_2) + get_log_probability(sentence, self.model_2)

        if self.name_class_1 is not None:
            return self.name_class_1 if p_1 > p_2 else self.name_class_2
        else:
            return 1 if p_1 > p_2 else 2


def get_train_test_indices(length, train_fraction=0.8):
    indices = rng.permutation(np.arange(length))
    delim = np.round(length * train_fraction).astype(np.int32)
    return indices[0: delim], indices[delim:]


class PPMI:
    def __init__(self, corpus: list, vocabulary: set, context_size: int, token_indexer: TokenIndexer):
        self.text = []
        self.get_text(corpus)

        self.vocabulary = vocabulary
        self.token_indexer = token_indexer
        self.context_size = context_size
        self.half_window_size = (context_size - 1) // 2

        self.co_occurrence_matrix = np.zeros([len(vocabulary), len(vocabulary)], dtype=np.int32)
        self.populate_co_occurrence_matrix()

        self.ppmi_matrix = np.zeros([len(vocabulary), len(vocabulary)])
        self.create_ppmi_matrix()

    def get_text(self, corpus):
        for line in corpus:
            self.text += line[1: -1]  # <s> and </s>

    def populate_co_occurrence_matrix(self):
        for i in range(self.half_window_size, len(self.text) - self.half_window_size):
            context_tokens = self.text[i - self.half_window_size: i] + self.text[i + 1: i + 1 + self.half_window_size]
            idx1 = self.token_indexer.get_token_idx(self.text[i])
            for token in context_tokens:
                idx2 = self.token_indexer.get_token_idx(token)
                self.co_occurrence_matrix[idx1, idx2] += 1
                self.co_occurrence_matrix[idx2, idx1] += 1

    def create_ppmi_matrix(self):
        rows_sums = np.sum(self.co_occurrence_matrix, axis=0)
        cols_sums = np.sum(self.co_occurrence_matrix, axis=1)
        total_sum = np.sum(self.co_occurrence_matrix)
        expected = np.outer(rows_sums, cols_sums) / total_sum
        self.ppmi_matrix = np.log(self.co_occurrence_matrix / expected)
        self.ppmi_matrix[np.isnan(self.ppmi_matrix)] = 0
        self.ppmi_matrix[self.ppmi_matrix < 0] = 0


def sample_word(probabilities: np.ndarray):
    p_threshold = np.sum(probabilities) * rng.random()  # Unif[a,b): (b - a) * random() + a,
    p_cumulative = 0
    for idx in range(len(probabilities)):
        p_cumulative += probabilities[idx]
        if p_threshold < p_cumulative:
            return idx
    assert False, f"p_cumulative={p_cumulative}, p_thr={p_threshold}"


def sample_from_model(model: Model, number_of_sentences=4):
    for _ in range(number_of_sentences):
        sentence = []
        idx = model.get_token_idx("<s>")
        while True:
            next_idx = sample_word(model.model[idx, :])
            next_word = model.idx_to_token[next_idx]
            if next_word == "</s>":
                print(' '.join(sentence))
                break
            sentence.append(next_word)
            idx = next_idx


def sample_from_model_top_p(model: Model, number_of_sentences=4, top_p=5):
    for _ in range(number_of_sentences):
        sentence = []
        idx = model.get_token_idx("<s>")
        while True:
            probs = model.model[idx, :]
            idcs = np.argpartition(probs, -top_p)[-top_p:]
            x = probs[idcs]
            idx_to_partition = sample_word(probs[idcs])
            next_idx = idcs[idx_to_partition]
            next_word = model.idx_to_token[next_idx]
            if next_word == "</s>":
                print(' '.join(sentence))
                break
            sentence.append(next_word)
            idx = next_idx


def main():
    # todo: can transform corpus into ids
    if not os.path.exists("edgar_allan_poe.txt"):
        wget.download("https://raw.githubusercontent.com/GustikS/smu-nlp/master/edgar_allan_poe.txt")
    if not os.path.exists("robert_frost.txt"):
        wget.download("https://raw.githubusercontent.com/GustikS/smu-nlp/master/robert_frost.txt")

    vocabulary_frost, corpus_frost = get_vocabulary_and_corpus("robert_frost.txt")
    vocabulary_poe, corpus_poe = get_vocabulary_and_corpus("edgar_allan_poe.txt")

    v, c = get_vocabulary_and_corpus_trigram("robert_frost.txt")
    v.union("<UNK>")
    model_frost = TriGramModel(v, c, smooth=False)
    idx1 = model_frost.get_token_idx('well')
    idx2 = model_frost.get_token_idx('may')
    idx3 = model_frost.get_token_idx('prove')
    print(model_frost.get_prob(idx1, idx2, idx3))

    # Exercise 1. Create bi-gram model.

    smooth = True
    model_frost = BiGramModel(vocabulary_frost, corpus_frost, smooth=smooth)

    for i in range(model_frost.model.shape[0]):
        if smooth is False:
            if model_frost.idx_to_token[i] == "<UNK>" or model_frost.idx_to_token[i] == "</s>":
                continue
        prob = np.sum(model_frost.model[i, :])
        assert np.abs(prob - 1) < 10 ** -9, f"prob={prob} at token: \"{model_frost.idx_to_token[i]}\""

    # Exercise 2. Calculate sentence probability under bi-gram model.
    sentence = corpus_frost[rng.integers(0, len(corpus_frost))]
    print("==== Exercise 2. ====")
    print(f"Probability of \"{' '.join(sentence)}\" is {get_probability(sentence, model_frost)}.")

    # Exercise 3. Find sentence from corpus with maximum and minimum perplexity.
    max_perplexity = 0
    min_perplexity = np.inf
    min_idx = np.nan
    max_idx = np.nan
    for i, sentence in enumerate(corpus_frost):
        perp = get_perplexity(sentence, model_frost)
        if perp > max_perplexity:
            max_perplexity = perp
            max_idx = i
        if perp < min_perplexity:
            min_perplexity = perp
            min_idx = i

    print("\n==== Exercise 3. ====")
    print(f"Sentence with minimum perplexity = {np.round(min_perplexity, 2)}: \"{' '.join(corpus_frost[min_idx])}\"")
    print(f"Sentence with maximum perplexity = {np.round(max_perplexity, 2)}: \"{' '.join(corpus_frost[max_idx])}\"")

    # Exercise 4. Markov classifier
    vocabulary = vocabulary_frost.union(vocabulary_poe)

    frost_idcs_train, frost_idcs_test = get_train_test_indices(len(corpus_frost))
    poe_idcs_train, poe_idcs_test = get_train_test_indices(len(corpus_poe))
    corpus_frost_train = [corpus_frost[idx] for idx in frost_idcs_train]
    corpus_frost_test = [corpus_frost[idx] for idx in frost_idcs_test]
    corpus_poe_train = [corpus_poe[idx] for idx in poe_idcs_train]
    corpus_poe_test = [corpus_poe[idx] for idx in poe_idcs_test]

    # # The two vocabularies are presumably not identical. The smoothing is done for each vocabulary independently.
    # # The probability of <UNK> is 1/V differs. So I will use V as union of both vocabularies.
    # print(np.sum(model_poe.model[model_poe.token_to_idx["<UNK>"], :]))
    # print(np.sum(model_frost.model[model_frost.token_to_idx["<UNK>"], :]))
    # Training on the different corpora which can differ in size,
    #   but on union of vocabularies, thus smoothing treat classes equally.
    model_frost = BiGramModel(vocabulary, corpus_frost_train, smooth=True)
    model_poe = BiGramModel(vocabulary, corpus_poe_train, smooth=True)

    for i in range(model_frost.model.shape[0]):
        prob = np.sum(model_frost.model[i, :])
        assert np.abs(prob - 1) < 10 ** -9, f"prob={prob} at token: \"{model_frost.idx_to_token[i]}\""

    number_of_tokens_frost = 0
    number_of_tokens_poe = 0
    for sentence in corpus_frost:
        number_of_tokens_frost += len(sentence) - 2
    for sentence in corpus_poe:
        number_of_tokens_poe += len(sentence) - 2
    number_of_tokens = (number_of_tokens_poe + number_of_tokens_frost)
    prior_poe = number_of_tokens_poe / number_of_tokens
    prior_frost = number_of_tokens_frost / number_of_tokens

    print("\n==== Exercise 4. ====")
    print(f"prior_frost={np.round(prior_frost, 2)}, prior_poe={np.round(prior_poe, 2)}")

    markov_classifier = MarkovClassifier(prior_1=prior_poe, prior_2=prior_frost,
                                         model_1=model_poe, model_2=model_frost,
                                         name_class_1="poe", name_class_2="frost")

    correctly_classified_poe = 0
    for sentence in corpus_poe_test:
        if markov_classifier.classify(sentence) == "poe":
            correctly_classified_poe += 1

    correctly_classified_frost = 0
    for sentence in corpus_frost_test:
        if markov_classifier.classify(sentence) == "frost":
            correctly_classified_frost += 1

    test_accuracy_poe = correctly_classified_poe / len(corpus_poe_test)
    print(f"Test accuracy Poe: {np.round(test_accuracy_poe * 100, 2)}%.")

    test_accuracy_frost = correctly_classified_frost / len(corpus_frost_test)
    print(f"Test accuracy Frost: {np.round(test_accuracy_frost * 100, 2)}%.")

    # Exercise 5. PPMI word-word occurrences.
    token_indexer = TokenIndexer(vocabulary, include_unk=False)
    ppmi = PPMI(corpus=corpus_frost + corpus_poe, vocabulary=vocabulary, context_size=5, token_indexer=token_indexer)

    # Exercise 6. Word embeddings.
    components = TruncatedSVD(n_components=8).fit_transform(ppmi.ppmi_matrix)
    fig = px.scatter(x=components[:, 0], y=components[:, 1], text=list(token_indexer.token_to_idx.keys()),
                     size_max=60, title="SVD components of PPMI matrix.")
    fig.update_traces(textposition='top center')
    fig.show()

    # Exercise 7. LSTM Language Model.
    class PoetrySequential(torch.utils.data.Dataset):
        def __init__(self, lines, word2idx, sequence_length=10):
            self.sequence_length = sequence_length
            self.words = [item for sublist in lines for item in sublist]

            # self.index_to_word = idx2word
            # self.word_to_index = word2idx

            self.words_indices = [word2idx[w] for w in self.words]

        def __len__(self):
            return len(self.words_indices) - self.sequence_length

        def __getitem__(self, index):
            return (torch.tensor(self.words_indices[index:index + self.sequence_length]),
                    torch.tensor(self.words_indices[index + 1:index + self.sequence_length + 1]))
            # the same sequence shifted by 1 step

    class RNNLM(torch.nn.Module):
        def __init__(self, n_vocab, pretrained_vectors: np.ndarray):
            super(RNNLM, self).__init__()
            self.embedding_dim = pretrained_vectors.shape[1]
            self.hidden_dim = 16
            self.num_layers = 2

            # self.embedding = torch.nn.Embedding(n_vocab, self.embedding_dim)
            self.embedding = torch.nn.EmbeddingBag.from_pretrained(
                torch.from_numpy(pretrained_vectors).float(), freeze=False)

            self.rnn = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)

            self.fc = torch.nn.Linear(self.hidden_dim, n_vocab)

        def forward(self, x):
            embed = self.embedding(x)
            output, state = self.rnn(embed)
            logits = self.fc(output)
            return logits, state

    # sequence_length = 10
    # poetry_data = PoetrySequential(lines, word2idx)
    # dataloader = DataLoader(poetry_data, batch_size=2048)

    # Exercise 8. Sampling.
    print("\n==== Exercise 8. ====")

    print("*Sample from not-smoothed bi-gram model trained on Frost corpus:")
    sample_from_model(BiGramModel(vocabulary_frost, corpus_frost, smooth=False))

    top_p = 50
    print(f"\n*Sample from not-smoothed bi-gram model trained on Frost corpus, best {top_p} words:")
    sample_from_model_top_p(BiGramModel(vocabulary_frost, corpus_frost, smooth=False), top_p=top_p)

    debug_buffer = 1


if __name__ == '__main__':
    main()

# notes:
# https://huggingface.co/blog/how-to-generate
