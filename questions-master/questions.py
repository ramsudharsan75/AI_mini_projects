import nltk
import sys
import os
import string
import functools
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    file_dict = {}

    for i in files:
        f = open(os.path.join(directory, i), encoding="utf8")
        file_dict[i] = f.read()
        f.close()

    return file_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document.lower())
    refined_tokens = [
        word for word in tokens if not (word in nltk.corpus.stopwords.words("english") or word in string.punctuation)
    ]

    return refined_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set(functools.reduce(lambda x, y: x + y, documents.values(), []))
    idfs = {}

    for word in words:
        s = sum(word in documents[f] for f in documents)
        idf = math.log(len(documents) / s)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    score = {}

    for word in query:
        for f in files.keys():
            if word in files[f]:
                score[f] = score.get(f, 0) + files[f].count(word) * idfs[word]

    sorted_scores = sorted(score.items(), key=lambda i: i[1], reverse=True)
    result = [sorted_scores[i][0] for i in range(n)]
    return result


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    score = {}

    for s in sentences:
        score[s] = {
            'idf': 0,
            'q_den': 0
        }

        for word in query:
            if word in sentences[s]:
                score[s]['idf'] += idfs[word]
                score[s]['q_den'] += (s.count(word) / len(s))

    sorted_scores = sorted(score.items(),
                           key=lambda i: (i[1]['idf'], i[1]['q_den']),
                           reverse=True
                           )
    # print(sorted_scores)
    result = [sorted_scores[i][0] for i in range(n)]

    return result


if __name__ == "__main__":
    main()
