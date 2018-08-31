from __future__ import print_function, unicode_literals

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")


import click

from tabulate import tabulate
import emoji
from pyfiglet import Figlet

import gensim
import spacy
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

import pandas as pd

tabulate.PRESERVE_WHITESPACE = True

from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint

nlp = spacy.load('en')


style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})


nlp = spacy.load('en', parser=False)
model = gensim.models.Word2Vec.load('sentence_doc2vec_model.doc2vec')


class Document(object):
    def __init__(self, home=None, debug=False):
        self.home = os.path.abspath(home or '.')
        self.debug = debug


@click.group()
@click.option('--document')
@click.pass_context
def cli(ctx, document):
    ctx.obj = Document(document)

pass_document = click.make_pass_decorator(Document)


def open_data(directory):
    # Folder containing all papers.
    data_dir = directory
    files = os.listdir(data_dir)
    for filen in files:
        if not filen.startswith('.'):
            with open(data_dir + '/' + filen, errors='ignore') as fid:
                txt = fid.read().replace('\n', '')
            docs.append(txt)
            titles.append(filen)
    return titles, docs

def create_dataframes(directory):
    open_data(directory)
    doc_text = pd.DataFrame({"text": docs})
    doc_titles = pd.DataFrame({"titles": titles})
    dataframe = pd.concat([doc_text,doc_titles], axis=1)
    return dataframe

def read_corpus(documents):
    for i, text in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [i])

def create_clusters(dv_model, num_clusters):
    df = pd.DataFrame()
    doc_vectors = model.docvecs.doctag_syn0
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( doc_vectors )
    word_centroid_list = list(zip(model.docvecs.offset2doctag, idx))
    word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
    
    for word_centroid in word_centroid_list_sort:
        line = word_centroid[0] + '\t' + str(word_centroid[1]) + '\n'
        d.append({'author': word_centroid[0], 'category': word_centroid[1]})


def read_corpus(documents):
    for i, plot in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(plot, max_len=30), [i])


click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")

f = Figlet(font='slant')
click.echo(f.renderText('WIRED! CLI'))

click.echo('Wired! CLI Is A Command Line Interface for Image Analysis. This CLI allows users to develop image categorization and object detection interfaces from a folder of images. It was developed throught the Wired! Lab at Duke University {}'.format(emoji.emojize(':zap:', use_aliases=True)))


click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")

@cli.command()
@click.option('--document', prompt='What article would you like to use?', help='ex. document_1.txt -- the document that you would like to query')
def train(document):
    """
    Train a new model
    """
    
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')
    doc = nlp(document_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    
    articledf = pd.DataFrame({"sentence": sentences})
    articledf['name'] = 'article1'

    train_corpus = list(read_corpus(articledf.sentence))

    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)

    model.build_vocab(train_corpus)
    
    click.echo('....Training Model {}'.format(emoji.emojize(':muscle:', use_aliases=True)))
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    
    click.echo('....Saving Model {}'.format(emoji.emojize(':pencil:', use_aliases=True)))
    model.save('sentence_doc2vec_model.doc2vec')
    click.echo('....Performing Inference {}'.format(emoji.emojize(':boom:', use_aliases=True)))


@cli.command()
@click.option('--document', prompt='Input Document', help='ex. document_1.txt -- the document that you would like to query')
@click.option('--input_sentence', prompt='Quote', help='ex. document_1.txt -- the document that you would like to query')
def quote(document, input_sentence):
    """
    Find a quote in the document
    """
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')
    doc = nlp(document_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    
    articledf = pd.DataFrame({"sentence": sentences})
    articledf['name'] = 'article1'
    click.echo('....Performing Inference {}'.format(emoji.emojize(':boom:', use_aliases=True)))
    tokens = gensim.utils.simple_preprocess(input_sentence)
    vec = model.infer_vector(tokens)
    sim = model.docvecs.most_similar(positive=[vec], topn=1)
    quotes = articledf['sentence'][sim[0][0]]
    click.echo(quotes)


@cli.command()
@click.option('--document', prompt='Input Document', help='ex. document_1.txt -- the document that you would like to query')
def summary(document):
    """
    Get a summary of the document 
    """
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')

    click.echo('....Performing Inference {}'.format(emoji.emojize(':boom:', use_aliases=True)))
    summary_text = summarize(str(document_text))
    click.echo("")
    click.echo("")
    click.echo(summary_text)
    click.echo("")
    click.echo("")

@cli.command()
@click.option('--document', prompt='Input Document', help='ex. document_1.txt -- the document that you would like to query')
def words(document):
    """
    List the keywords in the document
    """
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')

    click.echo('....Performing Inference {}'.format(emoji.emojize(':boom:', use_aliases=True)))
    document_keywords = keywords(document_text).split('\n')
    keywords_df = pd.DataFrame({"keywords": document_keywords})
    keywords_table = tabulate(keywords_df, headers=['keyword'], tablefmt="fancy_grid") 
    click.echo(keywords_table)

@cli.command()
@click.option('--document', prompt='Input Document', help='ex. document_1.txt -- the document that you would like to query')
def info(document):
    """
    List the facts in the document
    """
    click.secho(".........loading {}".format(emoji.emojize(':clock1030:', use_aliases=True)), fg='red')
    info_nlp = spacy.load('en_core_web_lg')
    click.secho('.........ready! {}'.format(emoji.emojize(':rocket:', use_aliases=True)), fg='green')
    
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')
    doc = nlp(document_text)

    statements = textacy.extract.semistructured_statements(doc, "{}".format(query))

    click.echo("Here are the things I know about {}:".format(query))

    for statement in statements:
        subject, verb, fact = statement
        click.secho(" {} -- {}".format(emoji.emojize(':heavy_check_mark:', use_aliases=True),fact), fg='green')

@cli.command()
@click.option('--document', prompt='What url would you like to use?', help='ex. document_1.txt -- the document that you would like to query')
def ner(document):
    """
    List people in the document
    """
    document_text = click.open_file(document, errors='ignore').read().replace('\n', '').replace('\t', '')


    questions = [
        {
        'type': 'list',
        'name': 'ner',
        'message': 'What size do you need?',
        'choices': ['PERSON', 'GPE', 'NORP', 'FAC', 'ORG', 'LOC'],
        'filter': lambda val: val
        }
    ]

    answers = prompt(questions, style=style)

    click.echo('....Performing Inference {}'.format(emoji.emojize(':boom:', use_aliases=True)))

    doc = nlp(article_text)
    people = [ent for ent in doc.ents if ent.label_ == answers['ner']]
    people_df = pd.DataFrame({"people": people})
    people_table = tabulate(people_df, headers=[answers['ner']], tablefmt="fancy_grid") 
    click.echo(people_table)

@cli.command()
@click.option('--path', prompt='Your Folder', help='ex. document_1.txt -- the document that you would like to query')
def doc_compare(path):
    dataframe = create_dataframes(path)
    train_corpus = list(read_corpus(dataframe.text))
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save_word2vec_format('doc_tensor_wedland_documents.w2v', doctag_vec=True, word_vec=False)
    click.echo(model.docvecs.most_similar(positive=[2], topn=10))
    create_cluset
    
if __name__ == '__main__':
    cli()