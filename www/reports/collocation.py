#!/usr/bin/env python

import sys
sys.path.append('..')
import functions as f
try:
    import ujson as json
except ImportError:
    import json
import timeit
from collections import defaultdict
from philologic.DB import DB
from wsgiref.handlers import CGIHandler
from functions.wsgi_handler import WSGIHandler
from philologic.Query import get_expanded_query


def collocation(environ,start_response):
    config = f.WebConfig()
    db = DB(config.db_path + '/data/')
    request = WSGIHandler(db, environ)
    headers = [('Content-type', 'application/json; charset=UTF-8'),("Access-Control-Allow-Origin","*")]
    start_response('200 OK',headers)
    hits = db.query(request["q"],"cooc",request["arg"],**request.metadata)
    hits.finish()
    collocation_object = fetch_collocation(hits, request, db, config)
    yield json.dumps(collocation_object)

def fetch_collocation(hits, q, db, config):
    collocation_object = {"query": dict([i for i in q])}

    length = config['concordance_length']
    try:
        within_x_words = int(q['word_num'])
    except ValueError: ## Getting an empty string since the keyword is not specificed in the URL
        within_x_words = None

    if q.colloc_filter_choice == "nofilter":
        filter_list = []
    else:
        filter_list = build_filter_list(q, config)
    collocation_object['filter_list'] = filter_list
    filter_list = set(filter_list)

    # Build list of search terms to filter out
    query_words = []
    for group in get_expanded_query(hits):
        for word in group:
            word = word.replace('"', '')
            query_words.append(word)
    query_words = set(query_words)
    filter_list = filter_list.union(query_words)

    stored_sentence_id = None
    stored_sentence_counts = defaultdict(int)
    sentence_hit_count = 1
    hits_done = q.start or 0
    max_time = q.max_time or 10
    all_collocates = {}
    c = db.dbh.cursor()
    start_time = timeit.default_timer()
    try:
        for hit in hits[hits_done:]:
            word_id = ' '.join([str(i) for i in hit.philo_id])
            query = """select philo_name, parent, rowid from words where philo_id='%s'""" % word_id
            c.execute(query)
            result = c.fetchone()
            parent = result['parent']
            rowid = int(result['rowid'])
            if parent != stored_sentence_id:
                sentence_hit_count = 1
                stored_sentence_id = parent
                stored_sentence_counts = defaultdict(int)
                if within_x_words:
                    begin_rowid = rowid - within_x_words
                    if begin_rowid < 0:
                        begin_rowid = 0
                    end_rowid = rowid + within_x_words
                    row_query = """select philo_name from words where parent='%s' and rowid between %d and %d"""  % (parent, begin_rowid, rowid)
                else:
                    row_query = """select philo_name from words where parent='%s'"""  % (parent,)
                c.execute(row_query)
                for i in c.fetchall():
                    stored_sentence_counts[i['philo_name']] += 1
            else:
                sentence_hit_count += 1
            for word in stored_sentence_counts:
                if word in filter_list or stored_sentence_counts[word] < sentence_hit_count:
                    continue
                if word in all_collocates:
                    all_collocates[word]['count'] += 1
                else:
                    # Generating links takes too long. This can be done one the client.
                    # all_link = f.link.make_absolute_query_link(config, q, report="concordance", q=query_string, method='cooc', start='0', end='0')
                    all_collocates[word] = {"count": 1}
            hits_done += 1
            elapsed = timeit.default_timer() - start_time
            if elapsed > int(max_time): # avoid timeouts by splitting the query if more than q.max_time (in seconds) has been spent in the loop
                break
    except IndexError:
        collocation['hits_done'] = len(hits)

    collocation_object['collocates'] = all_collocates

    collocation_object["results_length"] = len(hits)
    if hits_done < collocation_object["results_length"]:
        collocation_object['more_results'] = True
        collocation_object['hits_done'] = hits_done
    else:
        collocation_object['more_results'] = False
        collocation_object['hits_done'] = collocation_object["results_length"]

    return collocation_object

def build_filter_list(q, config):
    ## set up filtering with stopwords or most frequent terms ##
    if config.stopwords and q.colloc_filter_choice == "stopwords":
        filter_file = open(config.db_path + '/data/' + config.stopwords)
        filter_num = float("inf")
    else:
        filter_file = open(config.db_path + '/data/frequencies/word_frequencies')
        if q.filter_frequency:
            filter_num = int(q.filter_frequency)
        else:
            filter_num = 100 ## default value in case it's not defined
    filter_list = [q['q']]
    for line_count, line in enumerate(filter_file):
        if line_count == filter_num:
            break
        try:
            word = line.split()[0]
        except IndexError:
            continue
        filter_list.append(word)
    return filter_list

if __name__ == "__main__":
    CGIHandler().run(collocation)
