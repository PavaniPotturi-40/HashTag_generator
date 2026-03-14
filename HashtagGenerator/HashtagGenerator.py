#!/usr/bin/env python
##########################################
#
# Improved HashtagGenerator.py
# Hashtag extraction using LDA
# Evaluation: Coherence, Precision, Recall, F1, BLEU
#
#########################################

from __future__ import print_function

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import string
import argparse
import re

regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

FLAGS = None


def main():

    stop = set(stopwords.words(FLAGS.language))

    custom_stopwords = {
        "may","also","like","one","two","many","much",
        "article","good","bad","ugly","side","help",
        "using","used","use","system","technology"
    }

    stop.update(custom_stopwords)

    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    path = FLAGS.document
    text = ""

    # -------- URL CASE --------
    if re.match(regex, path) != None:

        import bs4
        import requests

        print("[Url found]")

        html = bs4.BeautifulSoup(requests.get(path).text, "html.parser")

        paragraphs = html.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

    # -------- FILE CASE --------
    else:
        text = open(path).read()

    # -------- PHRASE NORMALIZATION --------
    text = text.lower()

    text = text.replace("machine learning", "machinelearning")
    text = text.replace("deep learning", "deeplearning")
    text = text.replace("artificial intelligence", "ai")
    text = text.replace("natural language processing", "nlp")

    doc_complete = text.split("\n")

    # -------- CLEANING FUNCTION --------
    def clean(doc):

        stop_free = " ".join(
            [i for i in doc.lower().split() if i not in stop]
        )

        punc_free = ''.join(
            ch for ch in stop_free if ch not in exclude
        )

        words = []

        important_words = {"ai","ml","nlp"}

        for word in punc_free.split():

            word = lemma.lemmatize(word)

            if (len(word) > 3 or word in important_words) and word.isalpha():
                words.append(word)

        return " ".join(words)

    # -------- CLEAN DOCUMENT --------
    doc_clean = [
        clean(doc).split() for doc in doc_complete if clean(doc) != ""
    ]

    if len(doc_clean) == 0:
        print("No valid text found for topic modeling.")
        return

    # -------- CREATE DICTIONARY --------
    dictionary = corpora.Dictionary(doc_clean)

    # -------- BAG OF WORDS --------
    doc_term_matrix = [
        dictionary.doc2bow(doc) for doc in doc_clean
    ]

    # -------- LDA MODEL --------
    Lda = gensim.models.ldamodel.LdaModel

    ldamodel = Lda(
        doc_term_matrix,
        num_topics=FLAGS.hashtags,
        id2word=dictionary,
        passes=FLAGS.passes
    )

    # -------- EXTRACT TOPICS --------
    topics = ldamodel.show_topics(
        num_topics=FLAGS.hashtags,
        num_words=10,
        formatted=False
    )

    hashtags = []

    for topic in topics:
        for word, prob in topic[1]:

            if len(word) > 3:
                hashtags.append(word)

    hashtags = list(dict.fromkeys(hashtags))
    hashtags = hashtags[:FLAGS.hashtags]

    print("\nGenerated HashTags:\n")

    for ht in hashtags:
        print("#"+ht, end=" ")

    print("\n")

    # ==========================
    # Evaluation Metrics
    # ==========================

    # -------- Topic Coherence --------
    coherence_model = CoherenceModel(
        model=ldamodel,
        texts=doc_clean,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    print("Topic Coherence Score:", round(coherence_score,3))

    # -------- Reference Hashtags (for demo evaluation) --------
    reference_tags = {"ai","machinelearning","data","algorithm","healthcare","technology"}

    generated_tags = set(hashtags)

    # -------- Precision --------
    tp = len(reference_tags & generated_tags)

    precision = tp / len(generated_tags) if len(generated_tags) > 0 else 0
    recall = tp / len(reference_tags)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print("Precision:", round(precision,3))
    print("Recall:", round(recall,3))
    print("F1 Score:", round(f1,3))

    # -------- BLEU Score --------
    reference = [list(reference_tags)]
    candidate = list(generated_tags)

    bleu_score = sentence_bleu(reference, candidate)

    print("BLEU Score:", round(bleu_score,3))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract hashtags from URL or long text using LDA"
    )

    parser.add_argument(
        "--document",
        type=str,
        required=True,
        help="Path or URL to document"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language of text"
    )

    parser.add_argument(
        "--hashtags",
        type=int,
        default=6,
        help="Number of hashtags"
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=50,
        help="LDA iterations"
    )

    FLAGS = parser.parse_args()

    main()
