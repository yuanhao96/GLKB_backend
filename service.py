import os
import flask
from flask import Flask, request, jsonify, url_for, Blueprint
from flask_restplus import Resource, Api, fields
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from neo4j import GraphDatabase
import pandas as pd
import re
# from bs4 import BeautifulSoup
# import requests
import socket
import sys
# from nlp import comm
import json
from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from Bio import Entrez
from semanticscholar import SemanticScholar
import networkx as nx
import networkx.algorithms.community as nx_comm
import config
import glkb_schema
import logging
from graphdatascience import GraphDataScience
from scipy import stats
import random
import string
import textwrap
from fuzzysearch import find_near_matches
import itertools
import pickle

# monkey patch courtesy of
# https://github.com/noirbizarre/flask-restplus/issues/54
# so that /swagger.json is served over https
# if os.environ.get('HTTPS'):
@property
def specs_url(self):
    """Monkey patch for HTTPS"""
    return url_for(self.endpoint('specs'), _external=False)
Api.specs_url = specs_url

# @property
# def base_url(self):
#     '''
#     The API base absolute url

#     :rtype: str
#     '''
#     print(url_for(self.endpoint('root'))+'api')
#     return url_for(self.endpoint('root'), _external=True)
# Api.base_url = base_url

app = Flask(__name__)
# app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(flask.Flask(__name__))
app.logger.setLevel(logging.INFO)
blueprint = Blueprint('api', __name__)
api = Api(blueprint, version=u'1.0', title='GLKB REST API', description='The RESTful APIs of GLKB (Genomic Literature KnowledgeBase)', contact='authors', contact_email='hyhao@umich.edu',prefix='/api' ,doc='/docs')
# api = Api(blueprint, version=u'1.0', title='GLKB REST API', description='The RESTful APIs of GLKB (Genomic Literature KnowledgeBase)', contact='authors', contact_email='hyhao@umich.edu',doc='/docs')
app.register_blueprint(blueprint)

frontend = api.namespace('frontend', description='GLKB web interface api')
nodes = api.namespace('nodes', description='Retrieving GLKB nodes')
graphs = api.namespace('graphs', description='Retrieving GLKB subgraphs')
search = api.namespace('search', description='Searching GLKB entities using a query')
emb = api.namespace('emb', description='Retrieving GLKB semantic embeddings')
others = api.namespace('others', description='other APIs')

article = graphs.model('article', {'pmid':fields.String(required=True, description='PubMed ID of the article')})
biomedical_term = graphs.model('biomedical term', {'element id':fields.String(required=True, description='Element ID of the biomedical term')})

enriched_sets = {
    'gene': json.load(open(config.gene_set)),
    'chemical': json.load(open(config.chemical_set)),
    'go': json.load(open(config.go_set)),
    'anatomy': json.load(open(config.anatomy_set)),
    'disease': json.load(open(config.disease_set)),
    'organism': json.load(open(config.organism_set)),
    'pathway': json.load(open(config.pathway_set)),
    'high_freq': json.load(open(config.high_freq_set)),
}
enriched_set_feats = {
    'high_freq': json.load(open(config.high_freq_set_feats))
}
# pmid2dates = json.load(open(config.pmid2date))
pmid2ncit = json.load(open(config.pmid2ncit))
semantic_embeddings = pickle.load(open(config.semantic_embedding, 'rb'))

@api.route('/hello')
@api.hide
class HelloWorld(Resource):
    def get(self):
        """
        Test
        """
        app.logger.info('test logging... Success!')
        return jsonify({'hello': 'world'})

@nodes.route('/article/<string:pmid>')
@nodes.doc(params={'pmid': 'PubMed ID of the article'})
class Article(Resource):
    def get(self, pmid):
        """
        Fetch a PubMed article based on its pmid
        """
        info = _Article_helper(pmid)
        return jsonify(info)

@nodes.route('/entity/<string:nid>')
@nodes.doc(params={'nid': 'entity id of the biomedical term'})
class Entity(Resource):
    def get(self, nid):
        """
        Fetch detailed biomedical entity information based on its element id
        """
        app.logger.info(
            f'Getting detailed information of vocabulary: {nid}...')
        info = _get_ent_detail(nid)
        return jsonify(info)

@nodes.route('/unmapped_entity/<string:nid>')
@nodes.doc(params={'nid': 'entity id of the unmapped entity (biomedical events / unmapped biomedical entities)'})
class Unmapped_entity(Resource):
    def get(self, nid):
        """
        Fetch detailed unmapped entity information based on its element id
        """
        app.logger.info(
            f'Getting detailed information of vocabulary: {nid}...')
        info = _get_unmapped_ent_detail(nid)
        return jsonify(info)

@nodes.route('/rel/<rel_id>')
@nodes.doc(params={'rel_id': 'relationship id'})
class rel_info(Resource):
    def get(self, rel_id):
        """
        Fetch detailed relationship information based on their ids
        """
        info = _get_rel_detail(rel_id)
        return jsonify(info[0])

@nodes.route('/rel_mult/<string:nid>/<string:nid2>')
@nodes.doc(params={'nid': 'database id of node1', 'nid2': 'database id of node2'})
class rel_info2(Resource):
    def get(self, nid, nid2):
        """
        Fetch detailed information of all relationships between two nodes 
        """
        info = _get_rel_detail2(nid, nid2)
        info = [i[0] for i in info]
        return jsonify(info)

@search.route('/lexical_search')
@search.doc(params={'query': 'a list of keywords about the article content', 'limit': 'maximum number of returned articles. default:200'})
class Query_article_lexical(Resource):
    def get(self):
        """
        Fetch related PubMed article list based on a query using lexical match
        """
        query = request.args.getlist('query')
        limit = request.args.get('limit', 200)
        # toks, ents, rels = _process_query(query)
        query = ' OR '.join(query)
        candidate_list = _lexical_search(query, toks=[], ents=[], limit=int(limit))
        return jsonify([a[0] for a in candidate_list])

@search.route('/semantic_search')
@search.doc(params={'query': 'a list of keywords about the article content', 'limit': 'maximum number of returned articles. default:200'})
class Query_article_semantic(Resource):
    def get(self):
        """
        Fetch related PubMed article list based on a query using semantic match
        """
        query = request.args.getlist('query')
        limit = request.args.get('limit', 200)
        # toks, ents, rels = _process_query(query)
        terms = _search_entities(query)
        terms = [t[1] for t in terms]
        app.logger.info(
            f'Searching terms: {terms}...')
        candidate_list = _semantic_search(terms, limit=int(limit))
        return jsonify([a[0] for a in candidate_list])

# @search.route('/search')
# @search.doc(params={'query': 'a query about the article content'})
# class Query_article(Resource):
#     def get(self):
#         """
#         Fetch related PubMed article list based on a query
#         """
#         query = request.args.get('query')
#         app.logger.info(
#             f'Getting searching articles: {query}...')
#         toks, ents, rels = _process_query(query)
#         app.logger.info(
#             f'Detected tokens: {toks}...')
#         app.logger.info(
#             f'Detected entities: {ents}...')
#         app.logger.info(
#             f'Detected rels: {rels}...')
#         candidate_list = _Article_search(query, toks, ents)
#         return jsonify([a[0] for a in candidate_list])

# @search.route('/external_search')
# @search.doc(params={'query': 'a query about the article content'})
# class Query_article_external(Resource):
#     def get(self):
#         """
#         Fetch related PubMed article list based on a query in semantic scholar
#         """
#         query = request.args.get('query')
#         candidate_list = _external_search(query)
#         return jsonify(candidate_list)

@search.route('/entity_search')
@search.doc(params={'query': 'names of biomedical terms'})
class Query_entity(Resource):
    def get(self):
        """
        Fetch related entities based on a query
        """
        query = request.args.getlist('query')
        app.logger.info(f'Searching vocabulary: {query}...')
        candidate_list = _search_entities(query)
        return jsonify(candidate_list)

@search.route('/rel_text')
@search.doc(params={'ent1': 'name of the 1st biomedical term',
'ent2': 'name of the 2nd biomedical term',
'level': 'level of the returned context, choose from [sentence, abstract]. default: sentence',
'semantic': 'return semantic relationships from PubMed abstracts, choose from [True, False]. default: True',
'curated': 'return curated relationships from biomedical databases, choose from [True, False]. default: True'
})
class get_text_rel(Resource):
    def get(self):
        """
        get descriptions of relationships between two biomedical terms from PubMed abstracts
        """
        ent1 = request.args.get('ent1')
        ent2 = request.args.get('ent2')
        level = request.args.get('level', 'sentence')
        semantic = request.args.get('semantic', 'True')
        curated = request.args.get('curated', 'True')
        heads = _search_entities([ent1])
        tails = _search_entities([ent2])
        info = _get_rel_text(heads, tails, level=level, semantic=eval(semantic), curated=eval(curated))
        return jsonify(info)

@search.route('/event_text')
@search.doc(params={'ent': 'name of a biomedical term',
'event_type': 'type of events, choose from ["Regulation", "Positive_regulation", "Negative_regulation", "Binding", "Phosphorylation", "Gene_expression", "Localization", "Transcription", "Protein_catabolism", None]. default: None',
'level': 'level of the returned context, choose from [sentence, abstract]. default: sentence'})
class get_text_event(Resource):
    def get(self):
        """
        get descriptions of events
        """
        ent = request.args.get('ent')
        typ = request.args.get('event_type', None)
        level = request.args.get('level', 'sentence')
        terms = _search_entities([ent])
        info = _get_event_text(terms, event_type=typ, level=level)
        return jsonify(info)

@graphs.route('/entity_graph')
@graphs.doc(params={'eid': 'entity ids of the biomedical terms'})
class Entity_graph(Resource):
    def get(self):
        """
        Fetch subgraph of input biomedical terms
        """
        _id = request.args.getlist('eid')
        app.logger.info(f'Searching for terms: {_id}...')
        nids = [int(_get_ent_detail(nid)['database_id']) for nid in _id]
        g = _get_vocab_graph0(nids)
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/article_graph')
class Article_graph(Resource):
    @graphs.doc(params={'pmid': 'PubMed IDs of the articles', 'vv_rel': 'Return relationships between biomedical terms. default: True'})
    def get(self):
        """
        Fetch subgraph of one or multiple articles and their content
        """
        _id = request.args.getlist('pmid')
        vv_rel = request.args.get('vv_rel', 'True')
        app.logger.info(f'Getting article graph of pmids: {_id}...')
        g = _get_articles_graph(_id, events=False, merge_edge=False, vv_rel=eval(vv_rel))
        info = _create_cytoscape_graph(g)
        return jsonify(info)
    
    @graphs.doc(params={'vv_rel': 'Return relationships between biomedical terms. default: True'})
    @graphs.expect([article])
    def post(self):
        """
        Fetch subgraph of one or multiple articles and their content
        """
        pmids = request.get_json()
        _id = [i['pmid'] for i in pmids]
        vv_rel = request.args.get('vv_rel', 'True')
        app.logger.info(f'Getting article graph of pmids: {pmids}...')
        g = _get_articles_graph(_id, events=False, merge_edge=False, vv_rel=eval(vv_rel))
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/citation_graph')
class Citation_graph(Resource):
    @graphs.doc(params={'pmid': 'PubMed IDs of the articles', 'inner': 'Return only citing relationships among input articles, otherwise articles that cite the input articles are also returned. default: True'})
    def get(self):
        """
        Fetch citation network of multiple articles
        """
        _id = request.args.getlist('pmid')
        inner = request.args.get('inner', 'True')
        app.logger.info(f'Getting citation graph of pmids: {_id}...')
        g = _get_cite_graph(_id, inner=eval(inner))
        info = _create_cytoscape_graph(g)
        return jsonify(info)

    @graphs.doc(params={'inner': 'Return only citing relationships among input articles, otherwise articles that cite the input articles are also returned. default: False'})
    @graphs.expect([article])
    def post(self):
        """
        Fetch citation network of multiple articles
        """
        pmids = request.get_json()
        _id = [i['pmid'] for i in pmids]
        inner = request.args.get('inner', 'True')
        app.logger.info(f'Getting citation graph of pmids: {_id}...')
        g = _get_cite_graph(_id, inner=eval(inner))
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/ontology_graph')
@graphs.doc(params={'eid': 'entity ids of the biomedical terms'})
class Ontology_graph(Resource):
    def get(self):
        """
        Fetch descending ontology network of multiple entities
        """
        _id = request.args.getlist('eid')
        app.logger.info(f'Getting ontology graph of entities: {_id}...')
        nids = [int(_get_ent_detail(nid)['database_id']) for nid in _id]
        g = _get_children_ontology(nids)
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/entity_graph_louvain')
@graphs.doc(params={'eid': 'entity ids of the biomedical terms'})
class Entity_graph(Resource):
    def get(self):
        """
        Fetch subgraph of one or multiple entities using Louvain community detection
        """
        _id = request.args.getlist('eid')
        app.logger.info(f'Searching for terms: {_id}...')
        _id = [int(_get_ent_detail(nid)['database_id']) for nid in _id]
        g, _ = _get_vocab_graph(_id, k=4, merge_edge=True)
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/entity_graph_enrichment')
@graphs.doc(params={'eid': 'entity ids of the biomedical terms'})
class Entity_graph_enrichment(Resource):
    def get(self):
        """
        Fetch subgraph of one or multiple entities using hyperGeometric enrichment test
        """
        _id = request.args.getlist('eid')
        app.logger.info(f'Searching for terms: {_id}...')
        nids = [int(_get_ent_detail(nid)['database_id']) for nid in _id]
        g, _ = _get_vocab_graph2(nids)
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@graphs.route('/entity_graph_k_shortest')
@graphs.doc(params={'eid': 'entity ids of the biomedical terms', 'k': 'top k shortest paths between all biomedical term pairs. default: 3'})
class Entity_graph_kshortest(Resource):
    def get(self):
        """
        Fetch subgraph of one or multiple entities using hyperGeometric enrichment test
        """
        _id = request.args.getlist('eid')
        app.logger.info(f'Searching for terms: {_id}...')
        k = request.args.get('k', 3)
        nids = [int(_get_ent_detail(nid)['database_id']) for nid in _id]
        g = _get_vocab_graph3(nids, k=int(k))
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@emb.route('/entity_embedding/<string:eid>')
@emb.doc(params={'eid': 'entity ids of the biomedical terms'})
class Entity_emb(Resource):
    def get(self, eid):
        """
        Get the semantic embedding of an entity
        """
        emb = semantic_embeddings.get(eid)
        if emb is None:
            emb = 'No semantic embeddings available for this term'
        
        return jsonify(emb.tolist())

@emb.route('/batch_entity_embedding')
class Entity_emb_mult(Resource):
    @emb.doc(params={'eid': 'entity ids of the biomedical terms'})
    def get(self):
        """
        Get the semantic embedding of a one or multiple entities
        """
        eids = request.args.getlist('eid')
        embs = dict()
        for eid in eids:
            if semantic_embeddings.get(eid) is not None:
                embs[eid] = semantic_embeddings.get(eid).tolist()
        return jsonify(embs)
    
    @emb.expect([biomedical_term])
    def post(self):
        """
        Get the semantic embedding of a one or multiple entities
        """
        eids = request.get_json()
        eids = [i['element id'] for i in eids]
        embs = dict()
        for eid in eids:
            if semantic_embeddings.get(eid) is not None:
                embs[eid] = semantic_embeddings.get(eid).tolist()
        return jsonify(embs)

@frontend.route('/frontend_node_detail/<string:nid>')
@frontend.hide
class Frontend_node_info(Resource):
    def get(self, nid):
        """
        Fetch detailed node information and related articles based on its id
        """
        app.logger.info(f'Getting detailed information of node: {nid}...')
        info = _get_node_detail(int(nid))
        return jsonify(info)

@frontend.route('/frontend_query2graph')
@frontend.hide
class Frontend_q2g(Resource):
    def get(self):
        """
        Fetch related PubMed articles into article graphs based on a query
        """
        query = request.args.get('query')
        toks, ents, rels = _process_query(query)
        candidate_list = _Article_search(query, toks, ents, max_length=10)
        _id = [a[0] for a in candidate_list]
        # app.logger.info(
        #     f'Retrieved {len(_id)} articles...')
        g = _get_articles_graph(_id, events=False, merge_edge=False)
        # app.logger.info(
        #     f'Created graph of size {len(g["nodes"])}...')
        info = _create_cytoscape_graph(g)
        return jsonify(info)

@frontend.route('/frontend_rel_detail_mult/<string:nid>/<string:nid2>')
@frontend.hide
class Frontend_rel_info_mult(Resource):
    def get(self, nid, nid2):
        """
        Fetch detailed relationship information based on their ids
        """
        info = _get_rel_detail2(nid, nid2)
        return jsonify(info)

@frontend.route('/frontend_rel_detail/<rel_id>')
@frontend.hide
class Frontend_rel_info(Resource):
    def get(self, rel_id):
        """
        Fetch detailed relationship information based on their ids
        """
        info = _get_rel_detail(rel_id)
        return jsonify(info)

# @V1.route('/ranking')

@frontend.route('/frontend_term2graph')
@frontend.hide
class Frontend_t2g(Resource):
    def get(self):
        """
        Fetch related terms into a graph based on a list of terms
        """
        query = request.args.get('query')
        app.logger.info(f'getting query: {query}...')
        query = query.split('|')
        candidate_list = _search_entities(query)
        app.logger.info(f'matched terms: {candidate_list}...')
        if len(candidate_list)>0:
            g, term_list = _get_vocab_graph([t[0] for t in candidate_list], k=4, merge_edge=True)
        else:
            g = {'nodes':[], 'links':[]}
            term_list = dict()
        app.logger.info(f'retrieved graph')
        info = _create_cytoscape_graph(g)
        all_terms = [i[1] for i in term_list]
        # for t in term_list:
        #     nid, pval = t
        #     term_info = _get_node_detail(int(nid), rel_article=False)
        #     all_terms.append({'name': term_info['name'], 'n_citations': term_info['n_citations'], 'pval': pval, 'type': '; '.join(term_info['type']), 'id':term_info['database_id']})
        # app.logger.info(f'get all related terms')
        return jsonify([info, all_terms])

@frontend.route('/frontend_add_nodes')
@frontend.hide
class Frontend_add_nodes(Resource):
    def get(self):
        """
        add new nodes to an existing graph
        """
        existing = request.args.get('existing')
        existing = [int(i) for i in existing.split('|')]
        new = request.args.get('new')
        new = [int(i) for i in new.split('|')]

        triplets = _get_triplets(
            existing+new, new, event_types=False)
        if len(triplets) > 0:
            graph = _generate_graph(triplets, self_loop=False)
            graph = _merge_graph_edge(graph)
        else:
            graph = _single_node_graph(new)

        return jsonify(graph)

@others.route('/_cypher_query')
@others.hide
class run_cypher(Resource):
    def get(self):
        """
        run cypher
        """
        cypher = request.args.get('cypher')

        with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
            with driver.session() as session:
                result = session.run(cypher)
                result = result.values().copy()  # candidate article list

        return jsonify(result)

# @V1.route('/filter_node')
# class Filter_graph_node(Resource):
#     """
#         Modify knowledge graph based on node features
#     """

#     def post(self):
#         content_type = request.headers.get('Content-Type')
#         if (content_type == 'application/json'):
#             info = request.get_json()  # [{field:value}, cytoscape graph]
#             method = request.args.get('method')
#             if not method:
#                 method = 'equal'
#             try:
#                 mod, graph = info
#                 graph = _create_json_graph2(graph)
#                 graph = _filter_graph_nodes(graph, method=method, **mod)
#                 graph = _create_cytoscape_graph(graph)
#                 return graph
#             except:
#                 return 'Format error! Expect data format [{field:value}, cytoscape graph]'
#         else:
#             return 'Content-Type not supported!'


# @V1.route('/filter_relationship')
# class Filter_graph_relationship(Resource):
#     """
#         Modify knowledge graph based on relationship features
#     """

#     def post(self):
#         content_type = request.headers.get('Content-Type')
#         if (content_type == 'application/json'):
#             info = request.get_json()  # [{field:value}, cytoscape graph]
#             method = request.args.get('method')
#             if not method:
#                 method = 'equal'
#             try:
#                 mod, graph = info
#                 graph = _create_json_graph2(graph)
#                 graph = _filter_graph_edges(graph, method=method, **mod)
#                 graph = _create_cytoscape_graph(graph)
#                 return graph
#             except:
#                 return 'Format error! Expect data format [{field:value}, cytoscape graph]'
#         else:
#             return 'Content-Type not supported!'


# @V1.route('/graph_view')
# class Graph_view(Resource):
#     """
#         Modify knowledge graph based on predifined views (article, article-term, term)
#     """

#     def post(self):
#         content_type = request.headers.get('Content-Type')
#         if (content_type == 'application/json'):
#             view = request.args.get('view')
#             # cytoscape graph # view = 'article', 'article-term', 'term'
#             graph = request.get_json()
#             try:
#                 graph = _create_json_graph2(graph)
#                 if view == 'article':
#                     graph = _filter_graph_nodes(graph, label="Article")
#                 elif view == 'article-term':
#                     graph = _filter_graph_edges(graph, label="Contain_vocab")
#                 elif view == 'term':
#                     graph = _filter_graph_nodes(graph, is_vocab="true ")
#                 else:
#                     return "Graph view not supported!"
#                 graph = _create_cytoscape_graph(graph)
#                 return graph
#             except:
#                 return 'Format error! Expect data format in cytoscape graph'
#         else:
#             return 'Content-Type not supported!'


# @V1.route('/nx_graph')
# class Nx_graph(Resource):
#     def post(self):
#         content_type = request.headers.get('Content-Type')
#         if (content_type == 'application/json'):
#             # [{field:threshold}, method, cytoscape graph]
#             info = request.get_json()
#             try:
#                 graph = _create_nx_graph(info)
#                 return graph
#             except:
#                 return 'Format error! Expect data format in cytoscape graph'
#         else:
#             return 'Content-Type not supported!'

# @V1.route('/community')

# @V1.route('/node_similarity')

# @V1.route('/hits')

# @V1.route('/pgrank')


def tokenize_query(q):
    tks = word_tokenize(q)
    all_tks = tks.copy()
    for i in range(len(tks)-1):
        all_tks.append(' '.join(tks[i:i+2]))
        if i < len(tks)-2:
            all_tks.append(' '.join(tks[i:i+3]))
        if i < len(tks)-3:
            all_tks.append(' '.join(tks[i:i+4]))
        if i < len(tks)-4:
            all_tks.append(' '.join(tks[i:i+5]))
    return all_tks


def _process_query(query):
    toks = tokenize_query(query)
    # ents, rels = _query_nlp(query)
    # print("finish process query")
    # return toks, ents, rels
    return toks, [], []


# def _query_nlp(data):
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
#     sock.connect((comm.server_host, comm.ner_server_port))
#     comm.send_data(data, sock)
#     result = comm.receive_data(sock)
#     sock.close()
#     # response = flask.jsonify({'request':data,'response': result})
#     return result

# article search


def _search_pubmed(query, retmax=100):
    Entrez.email = 'hyhao@umich.edu'
    handle = Entrez.esearch(db='pubmed', sort='relevance',
                            retmax=retmax, retmode='xml', term=query)
    results = Entrez.read(handle)
    return results


# def _semantic_search(query, toks=None, ents=[]):
#     if toks is None:
#         toks = tokenize_query(query)
#     if len(toks) >= 15:  # trim query
#         toks = toks[:15]
#     with open(config.cypher_file) as f:
#         content = json.load(f)
#         articles_cypher_seman = content["search"]["/ent-article"]
#     with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
#         with driver.session() as session:
#             terms = _search_entities(toks)
#             terms = [t[1] for t in terms]
#             result = session.run(articles_cypher_seman.format(terms))
#             candidate_list = result.values().copy()  # candidate article list
#     return candidate_list

def _semantic_search(terms, limit=200):
    with open(config.cypher_file) as f:
        content = json.load(f)
        articles_cypher_seman = content["search"]["/ent-article"]
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(articles_cypher_seman.format(terms, limit))
            candidate_list = result.values().copy()  # candidate article list
    return candidate_list


def _lexical_search(query, toks=None, ents=[], limit=200):
    if toks is None:
        toks = tokenize_query(query)
    if len(toks) >= 15:  # trim query
        toks = toks[:15]
    with open(config.cypher_file) as f:
        content = json.load(f)
        articles_cypher_lexi = content["search"]["/article-result"]
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            if len(toks) <= 5:
                result = session.run(articles_cypher_lexi.format(query), limit)
                candidate_list = result.values().copy()  # candidate article list
                candidate_list = sorted(
                    candidate_list, key=lambda x: x[1], reverse=True)
                candidate_list = candidate_list[:min(
                    limit, len(candidate_list))]
            # elif len(ents) > 0:
            #     terms = _search_entities(ents)
            #     result = session.run(articles_cypher_seman.format(terms))
            #     candidate_list = result.values().copy()  # candidate article list
            else:
                candidate_list = defaultdict(int)
                for tok in toks:
                    result = session.run(
                        articles_cypher_lexi.format(tok), limit).values()
                    for a in result:
                        candidate_list[a[0]] += a[1]
                # ranking scores
                candidate_list = sorted(
                    candidate_list.items(), key=lambda x: x[1], reverse=True)
                candidate_list = candidate_list[:min(
                    limit, len(candidate_list))]
    return candidate_list


def _external_search(query, max_len=500):
    sch = SemanticScholar()
    res = sch.search_paper(query, limit=100, fields_of_study=['Biology'])
    res = [item['externalIds'].get('PubMed') for item in res[:max_len]]
    res = [i for i in res if i is not None]
    return res


def _Article_search(query, toks, ents, external=False, max_length=500):
    # get cypher
    with open(config.cypher_file) as f:
        content = json.load(f)
        articles_cypher_pmid = content["search"]["/article-result2"]

    if not external:
        candidate_list = _lexical_search(
            query, toks, ents, max_length=max_length)
    else:
        sch = SemanticScholar()
        res = sch.search_paper(query, limit=100, fields_of_study=['Biology'])
        res = [item['externalIds'].get('PubMed') for item in res[:200]]
        res += _search_pubmed(query)['IdList']
        res = list(set(res).difference({None}))
        # query database
        with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
            with driver.session() as session:
                if len(toks) <= 5:
                    result = session.run(
                        articles_cypher_pmid.format(query, res))
                    # ranking scores
                    candidate_list = result.values().copy()  # candidate article list
                elif len(ents) > 1:
                    result = session.run(articles_cypher_pmid.format(
                        ' AND '.join([ent[0] for ent in ents]), res))
                    candidate_list = result.values().copy()  # candidate article list
                else:
                    result = session.run(articles_cypher_pmid.format(
                        ' AND '.join([tok for tok in toks]), res))
                    candidate_list = result.values().copy()  # candidate article list
    # print(f"finish search {len(candidate_list)} articles")
    return candidate_list


def _get_article_node_id(pmid):
    with open(config.cypher_file) as f:
        content = json.load(f)
        if type(pmid) is list:
            cypher = content["pmid2id"]["articles"].format(pmid)
        elif type(pmid) is str:
            cypher = content["pmid2id"]["article"].format(pmid)
        else:
            return None

    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return result.value().copy()


def _get_citation_list(pmid):
    with open(config.cypher_file) as f:
        content = json.load(f)
        if type(pmid) is str:
            cypher = content["article_info"]["citation"].format(pmid)
        else:
            return None
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return result.value().copy()


def _get_vocab_node_id(pmid, unmapped_ents=False):
    with open(config.cypher_file) as f:
        content = json.load(f)
        if type(pmid) is list:
            cypher = content["pmid2id"]["vocabs"].format(pmid)
        elif type(pmid) is str:
            if unmapped_ents:
                cypher = content["pmid2id"]["vocab_and_unmapped"].format(pmid)
            else:
                cypher = content["pmid2id"]["vocab"].format(pmid)
        else:
            return None

    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return result.values().copy()


def _get_event_nodes_id(pmid):
    with open(config.cypher_file) as f:
        content = json.load(f)
        if type(pmid) is list:
            cypher = content["pmid2id"]["events"].format(pmid)
        elif type(pmid) is str:
            cypher = content["pmid2id"]["event"].format(pmid)

    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return result.value().copy()


def _search_entities(ents, ent_thres=5, max_ent=7):
    # ent_ids = dict()
    # ent_cits = dict()
    with open(config.cypher_file) as f:
        content = json.load(f)
        ents_cypher = content["search"]['/ent-result']

    # query database
    results = []
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            # get entity ids in query
            for ent in ents:
                ent_ids = dict()
                ent_cits = dict()
                result = session.run(ents_cypher.format(ent)).values().copy()
                for k, _id, name, ncit, v in result:
                    if ncit:
                        ent_cits[(k, _id, name)] = ncit
                    if v >= ent_thres:
                        # if len(ent_ids) < max_ent:
                        if ent_ids.get((k, _id, name)):
                            ent_ids[(k, _id, name)] = max(v, ent_ids[(k, _id, name)])
                        else:
                            ent_ids[(k, _id, name)] = v
                if len(ent_ids) > 0:
                    # results += list(set([max(ent_ids, key=ent_ids.get)] + sorted(ent_cits, key=lambda x: ent_cits.get(x) if ent_cits.get(x) else 0, reverse=True)[:1]))
                    results += list(set([max(ent_ids, key=ent_ids.get)]))
                else:
                    # results += list(set(sorted(ent_cits, key=lambda x: ent_cits.get(x) if ent_cits.get(x) else 0, reverse=True)[:1]))
                    results += []
    # print("finish search entities")
    # return list(set(list(ent_ids.keys()) + sorted(ent_cits, key=lambda x: ent_cits.get(x) if ent_cits.get(x) else 0, reverse=True)[:3]))
    return list(set(results))

def _get_ent_detail(eid):
    with open(config.cypher_file) as f:
        content = json.load(f)
        ent_cypher = content["search"]['/ent']
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(ent_cypher.format(eid)).values().copy()
    try:
        node, alias = result[0]
        external_sources = dict()
        for i in glkb_schema.vocab_source:
            if node.get(i):
                external_sources[i] = node.get(i)
        n_cit = 0
        if node.get('n_citation'):
            n_cit = node.get('n_citation')
        res = {
            'element_id': node['Element ID'],
            'name': node['Name'],
            'aliases': [n.get('Name') for n in alias],
            'type': list(set(node.labels).difference({"Vocabulary"})),
            'external_sources': external_sources,
            'n_citations': n_cit,
            'database_id': node.element_id
        }
        if node['Definition'] is not None:
            res['description'] = node['Definition']
        else:
            res['description'] = ''
    except:
        res = 'Invalid Element ID'
    return res


def _get_unmapped_ent_detail(eid):
    with open(config.cypher_file) as f:
        content = json.load(f)
        ent_cypher = content["search"]['/unmapped_ent']
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(ent_cypher.format(eid)).value().copy()
    try:
        node = result[0]
        res = {
            'entity id': node['Entity ID'],
            'source article': node['Entity ID'].split('-')[0],
            'text': node['Text'],
            'description': "This is an entity indicating genomic concepts or events in the article.",
            'type': node['Type']
        }
    except:
        res = 'Invalid Entity ID'
    return res


def _get_node_detail(_id, rel_article=True):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["search"]['/node']
        ent_article = content['search']["/ent-article"]
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(_id)).value().copy()
            try:
                node = result[0]
                related_articles = []
                if "Article" in node.labels:
                    res = _Article_helper(node['PubmedID'])
                    if rel_article:
                        if not node['n_citation']:
                            n_citation = 0
                        else:
                            n_citation = node['n_citation']
                        related_articles = [(f"{node['Title']} ({node['Pubdate']})", f"https://pubmed.ncbi.nlm.nih.gov/{node['PubmedID']}/", n_citation, node['Pubdate'])]
                    related_articles.sort(key=lambda a: a[2])
                    related_articles = related_articles[:20]
                elif "Vocabulary" in node.labels:
                    res = _get_ent_detail(node['Element ID'])
                    if rel_article:
                        rel_articles = session.run(ent_article.format([node['Element ID']], 100)).values().copy()
                        if rel_articles:
                            for n in rel_articles:
                                pmid, _, n_citation, date, title = n
                                if not n_citation:
                                    n_citation = 0
                                if pmid and date and title:
                                    related_articles.append((f"{title} ({date})", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", n_citation, date))
                    related_articles.sort(key=lambda a: a[2])
                    related_articles = related_articles[:20]
                elif "Unmapped_entity" in node.labels:
                    res = {
                        'entity id': node['Entity ID'],
                        'source article': node['Entity ID'].split('-')[0],
                        'text': node['Text'],
                        'description': "This is an entity indicating genomic concepts or events in the article.",
                        'type': node['Type']
                    }
                else:
                    res = 'Unsupported node ID'
            except:
                res = 'Invalid node ID'
    if rel_article:
        return [res, related_articles]
    else:
        return res


def _get_rel_detail2(_id, _id2):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["node2graph"]['general_single_node']
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(_id, _id2)).values().copy()
    try:
        res = []
        for _, r, _ in result:
            d = {}
            d['node1'] = r.nodes[0].get('Name')
            d['node2'] = r.nodes[1].get('Name')
            d['relationship label'] = r.type
            d['relationship type'] = r.get('Type')
            d['number of citations'] = r.get('Frequency')
            
            article_list = r.get('Source')
            related_articles = []
            if article_list:
                related_articles = _pmids2articles(sorted(article_list, key=lambda x: pmid2ncit.get(x) if pmid2ncit.get(x) else 0, reverse=True)[:15])
            res.append([d, related_articles])
    except:
        res = [['Invalid node ID(s)', []]]
    return res

def _get_rel_detail(rel_id):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["node2graph"]['general_single_edge']
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(rel_id)).values().copy()
    try:
        res = []
        for _, r, _ in result:
            d = {}
            d['node1'] = r.nodes[0].get('Name')
            d['node2'] = r.nodes[1].get('Name')
            d['relationship label'] = r.type
            d['relationship type'] = r.get('Type')
            d['number of citations'] = r.get('Frequency')

            article_list = r.get('Source')
            related_articles = []
            if article_list:
                # related_articles = _pmids2articles(article_list[:10])
                related_articles = _pmids2articles(sorted(article_list, key=lambda x: pmid2ncit.get(x) if pmid2ncit.get(x) else 0, reverse=True)[:15])
            res.append([d, related_articles])
    except:
        res = [['Invalid edge ID(s)', []]]
    return res

def _generate_graph(triplets, self_loop=False, key_nodes=[]):
    graph = {
        "nodes": [],
        "links": []
    }
    nodes = {}
    rels = {}
    events = dict()
    c = Counter()
    for h, r, t in triplets:
        if self_loop:
            # if h and r and t:
            nodes[h.element_id] = h
            nodes[t.element_id] = t
            rels[r.element_id] = r
            c.update([h.element_id, t.element_id])
        else:
            if h.element_id != t.element_id:
                nodes[h.element_id] = h
                nodes[t.element_id] = t
                rels[r.element_id] = r
                c.update([h.element_id, t.element_id])
    app.logger.info(f'find {len(nodes)} nodes, {len(rels)} relationships')
    for n in nodes.values():
        node_info = {
            "id": n.element_id,
            # "name": n['Name'],
            "label": set(n.labels).difference({"Vocabulary"}).pop(),
            # "display": n['Name'],
            "frequency": c.get(n.element_id),
            "is_vocab": "false"
        }
        if n.element_id in key_nodes:
            node_info['key_nodes'] = 'true'
        else:
            node_info['key_nodes'] = 'false'
        if "Article" in n.labels:
            if n['Authors']:
                first_author = n['Authors'][0]
            else:
                first_author = 'Unknown'
            node_info["display"] = f"{first_author} et al ({str(n['Pubdate'])})"
            # node_info["frequency"] = None
            node_info["name"] = n["Title"]
            node_info["source"] = "PubMed"
            node_info["date"] = n["Pubdate"]
            node_info["pubmed_id"] = n["PubmedID"]
            if n["n_citation"]:
                node_info["n_citation"] = n["n_citation"]
            else:
                node_info["n_citation"] = 0
        elif "Unmapped_entity" in n.labels:
            if len(n["Text"]) <= 20:
                node_info["display"] = n["Text"]
            else:
                node_info["display"] = f'{n["Text"][:5]}...{n["Text"][-5:]}'
            node_info["name"] = n["Text"]
            node_info["source"] = n["Source"]
            if n["is_event"] == "False":  # unmapped ents
                node_info["label"] = "Entity"
            else:  # events
                node_info["label"] = "Event"
                node_info["type"] = n["Type"]
                node_info['roles'] = []  # list of ents in events
                events[n.element_id] = node_info
                continue
        elif "Vocabulary" in n.labels:
            if len(n["Name"]) <= 20:
                node_info["display"] = n["Name"]
            else:
                node_info["display"] = textwrap.shorten(n["Name"], width=20, placeholder="...")
                # node_info["display"] = f'{n["Name"][:5]}...{n["Name"][-5:]}'
            node_info["name"] = n["Name"]
            node_info["source"] = n["Element ID"].split("_")[0]
            node_info["is_vocab"] = "true"
            node_info["element_id"] = n["Element ID"]
            if n["n_citation"]:
                node_info["n_citation"] = n["n_citation"]
            else:
                node_info["n_citation"] = 0
        graph["nodes"].append(node_info)

    for r in rels.values():
        rel_info = {
            "source": r.nodes[0].element_id,
            "target": r.nodes[1].element_id,
            "label": r.type,
            "eid": r.element_id,
            "article_source": r['Source'],
            "dates": []
        }
        if r.get('Frequency'):
            rel_info['weight'] = r.get('Frequency')
        else:
            rel_info['weight'] = 10
        if r.type == "Semantic_relationship":
            if events.get(rel_info['source']):
                events[rel_info['source']]['roles'].append(
                    (rel_info['target'], r["Type"]))
                continue
            elif events.get(rel_info['target']):
                events[rel_info['target']]['roles'].append(
                    (rel_info['source'], r["Type"]))
                continue
            else:
                rel_info["type"] = r["Type"]
            # rel_info['dates'] = _pmids2pubdates(r['Source'])
            # rel_info['dates'] = [pmid2dates.get(i) for i in r['Source'] if pmid2dates.get(i)]
        elif r.type == "Curated_relationship":
            rel_info["type"] = r["Type"]
        graph["links"].append(rel_info)

    for ev in events.values():
        if len(ev['roles']) > 1:
            for i in range(len(ev['roles'])):
                for j in range(i+1, len(ev['roles'])):
                    graph['links'].append({
                        "source": ev['roles'][i][0],
                        "target": ev['roles'][j][0],
                        "label": "Semantic_relationship",
                        "type": ev['type'],
                        "weight": 1  # fix this later
                    })

    return graph

def _merge_graph_edge(g):
    ls = g['links']
    merged_ls = defaultdict(dict)
    for l in ls:
        d = merged_ls[(l.get('source'), l.get('target'), l.get('label'), l.get('type'))]
        d['label'] = l.get('label')
        d['source'] = l.get('source')
        d['target'] = l.get('target')
        d['type'] = l.get('type')
        if not d.get('article_source'):
            d['article_source'] = []
        if not d.get('dates'):
            d['dates'] = []
        if not d.get('eid'):
            d['eid'] = []
        d['eid'].append(l['eid'])
        # if not d.get('type'):
        #     d['type'] = set()
        # if l.get('type'):
        #     d['type'].add(l.get('type'))
        if not d.get('weight'):
            d['weight'] = 0
        d['weight'] += l.get('weight')
        if l.get('article_source'):
            if isinstance(l.get('article_source'), list):
                d['article_source'] += l.get('article_source')
            elif isinstance(l.get('article_source'), str):
                d['article_source'].append(l.get('article_source'))
        if l.get('dates'):
            d['dates'] += l['dates']

    merged_ls = list(merged_ls.values())
    for l in merged_ls:
        # l['type'] = ';'.join(l['type'])
        l['article_source'] = l['article_source'][:15]
        l['dates'] = Counter(l['dates'])
        l['dates'] = [{'year':k, 'weight':v} for k, v in l['dates'].items()]

    return {
        'links': merged_ls,
        'nodes': g['nodes']
    }


def _get_triplets(source_ids, target_ids, event_types=False, co_occur_pmid=False):
    with open(config.cypher_file) as f:
        content = json.load(f)
        if co_occur_pmid:
            if type(co_occur_pmid) is str:
                cypher = content["node2graph"]["co_occur"].format(
                    source_ids, target_ids, co_occur_pmid)
            elif type(co_occur_pmid) is list:
                cypher = content["node2graph"]["co_occur_list"].format(
                    source_ids, target_ids, co_occur_pmid)
            else:
                return None
        elif not event_types:
            if source_ids == target_ids:
                cypher = content["node2graph"]["general_directed"].format(
                    source_ids, target_ids)
            else:
                cypher = content["node2graph"]["general"].format(
                    source_ids, target_ids)
        elif type(event_types) is str:
            if source_ids == target_ids:
                cypher = content["node2graph"]["rel_type_directed"].format(
                    event_types, source_ids, target_ids)
            else:
                cypher = content["node2graph"]["rel_type"].format(
                    event_types, source_ids, target_ids)
        else:
            return []

    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return result.values().copy()

# def _filter_graph(graph, target_node_labels=None, target_graph_labels=None):
#     target_graph = {
#         "nodes": [],
#         "links": []
#     }
#     target_node_ids = []
#     if target_node_labels:
#         for n in graph["nodes"]:
#             if n["label"] in target_node_labels:
#                 target_graph["nodes"].append(n)
#                 target_node_ids.append(n["id"])
#         for r in graph["links"]:
#             if r["source"] in target_node_ids and r["target"] in target_node_ids:
#                 if target_graph_labels and r["label"] in target_graph_labels:
#                     target_graph["links"].append(r)
#                 elif not target_graph_labels:
#                     target_graph["links"].append(r)
#     else:
#         target_graph = graph
#     return target_graph


def _filter_graph_nodes(graph, method='equal', **kw):
    target_graph = {
        "nodes": [],
        "links": []
    }
    target_node_ids = set()
    for n in graph["nodes"]:
        add_n = False
        for feat in kw:
            if type(kw[feat]) in [int, str]:
                add_n = _filter_feat(n, feat, kw[feat], method)
            elif type(kw[feat]) is list:
                add_n = any([_filter_feat(n, feat, k, method)
                            for k in kw[feat]])
        if add_n:
            target_graph["nodes"].append(n)
            target_node_ids.add(n["id"])

    for r in graph["links"]:
        if r["source"] in target_node_ids and r["target"] in target_node_ids:
            target_graph["links"].append(r)
    return target_graph


def _filter_graph_edges(graph, method='equal', **kw):
    """
    method:
        'equal', 'greater' or 'less'
    filter properties: 
        INT or STR: 
        LST: calculate independently, select if any item pass the filter
    """
    target_graph = {
        "nodes": [],
        "links": []
    }
    target_node_ids = set()
    for r in graph["links"]:
        add_r = False
        for feat in kw:
            if type(kw[feat]) in [int, str]:
                add_r = _filter_feat(r, feat, kw[feat], method)
            elif type(kw[feat]) is list:
                add_r = any([_filter_feat(r, feat, k, method)
                            for k in kw[feat]])
        if add_r:
            target_graph["links"].append(r)
            target_node_ids.add(r["source"])
            target_node_ids.add(r["target"])

    n_dict = dict(zip([n["id"] for n in graph["nodes"]], graph["nodes"]))
    for n in target_node_ids:
        target_graph["nodes"].append(n_dict[n])

    return target_graph


def _filter_feat(data, key, value, method='equal'):
    if data.get(key):
        try:
            if method == 'equal':
                if data.get(key) == value:
                    return True
            elif method == 'greater':
                if data.get(key) >= value:
                    return True
            elif method == 'less':
                if data.get(key) <= value:
                    return True
            else:
                return False
        except:
            return False


def _create_cytoscape_graph(graph):
    csg = {
        "nodes": [{"data": n} for n in graph["nodes"]],
        "edges": [{"data": l} for l in graph["links"]]
    }
    return csg


def _create_nx_graph(graph):
    df = pd.DataFrame(graph['links'])
    nxg = nx.from_pandas_edgelist(df, 'source', 'target', ['label', 'type'])
    nx.set_node_attributes(
        nxg, dict(zip((i['id'] for i in graph['nodes']), graph['nodes'])))
    return nxg


def _create_json_graph(nxg):
    g = {
        "nodes": list(nxg.nodes.values()),
        "links": []
    }
    for k, v in nxg.edges.items():
        v['source'] = k[0]
        v['target'] = k[1]
        g["links"].append(v)
    return g


def _create_json_graph2(csg):
    g = {
        "nodes": [n["data"] for n in csg["nodes"]],
        "links": [n["data"] for n in csg["edges"]]
    }
    return g


def louvain_cluster(nxg, res=0.5, weight=None, thres=0.0000001):
    comms = nx_comm.louvain_communities(
        nxg, resolution=res, weight=weight, threshold=thres)
    comm_dict = {}
    for c, comm in enumerate(comms):
        for n in comm:
            comm_dict[n] = {'lovain_comm': c}
    nx.set_node_attributes(nxg, comm_dict)
    return nxg


def hits_rank(nxg, max_iter=100, tol=1e-08, nstart=None, normalized=True):
    hubs, auths = nx.hits(nxg, max_iter, tol, nstart, normalized)
    dic = {}
    for n in nxg.nodes:
        dic[n] = {'hub': hubs.get(n), 'auth': auths.get(n)}
    nx.set_node_attributes(nxg, dic)
    return nxg


def harmonic_centrality(nxg, nbunch=None, distance=None, sources=None):
    cents = nx.harmonic_centrality(nxg, nbunch, distance, sources)
    dic = {}
    for n in nxg.nodes:
        dic[n] = {'harmonic_centrality': cents[n]}
    nx.set_node_attributes(nxg, dic)
    return nxg


def _extract_entities_from_abstract(abstract, entities):
    new_entities = {}
    old_abstract_list = [[abstract, 'none']]
    new_abstract_list = []
    for ent_type in entities.keys():
        for ent in entities[ent_type]:
            count = 0
            for abstr, type in old_abstract_list:
                if type == 'none':
                    pos_list = [a.start() for a in re.finditer(ent, abstr)]
                    start = 0
                    end = 0
                    count += len(pos_list)
                    for pos in pos_list:
                        end = pos
                        new_abstract_list.append(
                            [abstr[start:end], 'none'])
                        tmp_end = end+len(ent)
                        new_abstract_list.append(
                            [abstr[end:tmp_end], ent_type])
                        start = tmp_end
                    if (start < len(abstr)):
                        new_abstract_list.append(
                            [abstr[start:len(abstract)], 'none'])
                else:
                    new_abstract_list.append([abstr, type])
            if count > 0:
                if ent_type in new_entities.keys():
                    new_entities[ent_type].append([ent, count])
                else:
                    new_entities[ent_type] = [[ent, count]]
            old_abstract_list = new_abstract_list.copy()
            new_abstract_list = []
    # return old_abstract_list, list(new_entities)
    # return jsonify({'abstract_list': old_abstract_list, 'entities': list(new_entities)})
    return old_abstract_list, list(new_entities)


def _Article_helper(pmid):
    # init
    # ent_labels = ['Genes and Gene Products', 'Anatomy', 'Chemicals and Drugs', 'GO', 'Diseases', 'Organisms', 'Unmapped_entity']
    node_info = {}
    node_info['title'] = ''
    node_info['pmid'] = ''
    node_info['date'] = ''
    node_info['authors'] = ''
    node_info['abstract'] = ''
    node_info['journal'] = 'Unknown Journal'
    node_info['abstract_list'] = []
    node_info['citation_list'] = []
    entities = defaultdict(set)

    # query neo4j
    try:
        article_node = _get_article_node_id(pmid)[0]
        vocab_nodes = _get_vocab_node_id(pmid, unmapped_ents=True)
        citation_list = _get_citation_list(pmid)

        # process info
        node_info['title'] = article_node['Title']
        node_info['pmid'] = article_node['PubmedID']
        node_info['date'] = str(article_node['Pubdate'])
        node_info['authors'] = article_node['Authors']
        node_info['abstract'] = str(article_node['Abstract'])
        node_info['journal'] = article_node['Journal']
        for article in citation_list:
            art_dic = dict(article)
            art_dic['Pubdate'] = str(art_dic['Pubdate'])
            node_info['citation_list'].append(
                [art_dic["PubmedID"], art_dic["Pubdate"]])
        for r, v in vocab_nodes:
            if r['Text']:
                label = set(v.labels).difference({"Vocabulary"}).pop()
                texts = r['Text'].split('|')
                for t in texts:
                    entities[label].add(t)
        abstract_list, entities = _extract_entities_from_abstract(
            node_info['abstract'], entities)
        # abstract_list: [[text1, type1],[text2, type2]...]
        node_info['abstract_list'] = abstract_list
        node_info['entities'] = entities  # entities: {type: [text1, text2...]}
        node_info['n_citation'] = article_node.get('n_citation')
        node_info['author_ids'] = article_node.get('Author_IDs')
        return node_info

    except:
        return 'Invalid PubmedID'


def _get_articles_graph(pmids, unmapped_ents=False, events=True, merge_edge=False, self_loop=False, vv_rel=True):
    article_nodes = _get_article_node_id(pmids)
    app.logger.info(f'Get {len(article_nodes)} article nodes')
    vocab_nodes = _get_vocab_node_id(
        pmids, unmapped_ents=unmapped_ents)  # [[r1, v1], [r2, v2], ...]
    vocab_nodes = set([v[1] for v in vocab_nodes])
    app.logger.info(f'Get {len(vocab_nodes)} vocab nodes')
    
    if events:
        event_nodes = _get_event_nodes_id(pmids)
        # node_ids = list(set([a.element_id for a in article_nodes] +
        #                 [a.element_id for a in vocab_nodes]+[a.element_id for a in event_nodes]))
        vocab_nodes = list(set([int(a.element_id) for a in vocab_nodes])) + list(set([int(a.element_id) for a in event_nodes]))
    else:
        # node_ids = list(
        #     set([a.element_id for a in article_nodes]+[a.element_id for a in vocab_nodes]))
        vocab_nodes = list(set([int(a.element_id) for a in vocab_nodes]))
    # triplets = _get_triplets(node_ids, node_ids, event_types=':Contain_vocab|cite|Curated_relationship|Hierarchical_structure|Semantic_relationship|co_occur')

    article_nodes = list(set([int(a.element_id) for a in article_nodes]))
    
    triplets = _get_triplets(article_nodes, vocab_nodes,
                             event_types=':Contain_vocab')
    if vv_rel:
        triplets += _get_triplets(article_nodes,
                                article_nodes, event_types=':cite')
        triplets += _get_triplets(vocab_nodes, vocab_nodes,
                                event_types=':Curated_relationship|Hierarchical_structure|Semantic_relationship')
    app.logger.info(f'Get {len(triplets)} triplets')
    graph = _generate_graph(triplets, self_loop=self_loop)
    # graph = _filter_graph(graph)
    if merge_edge:
        graph = _merge_graph_edge(graph)
    return graph

def _get_cite_graph(pmids, inner=True):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["article_info"]['citations']
    # query database
    if not inner:
        with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
            with driver.session() as session:
                result = session.run(cypher.format(pmids)).value().copy()
        article_nodes = _get_article_node_id(pmids) + result
        app.logger.info(f'Retrieved {len(article_nodes)} articles...')
    else:
        article_nodes = _get_article_node_id(pmids)
        app.logger.info(f'Retrieved {len(article_nodes)} articles...')
    _ids = list(set([int(a.element_id) for a in article_nodes]))
    triplets = _get_triplets(_ids, _ids, event_types=':cite')
    graph = _generate_graph(triplets)
    return graph

def _get_children_ontology(_ids):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["node2graph"]['ontology']
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(_ids)).value().copy()
    _ids = _ids + list(set([int(a.element_id) for a in result]))
    triplets = _get_triplets(_ids, _ids, event_types=':Hierarchical_structure')
    graph = _generate_graph(triplets)
    return graph

# def _get_vocab_graph(eid_lst, unmapped_ents=False, events=True, co_occur=False):
#     with open(config.cypher_file) as f:
#         content = json.load(f)
#         cypher = content["search"]['/ent-article']
#         cypher2 = content["search"]['/ent-article2']
#     # query database
#     with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
#         with driver.session() as session:
#             pmids = session.run(cypher.format(eid_lst)).value().copy()
#             pmids += session.run(cypher2.format(eid_lst)).value().copy()
#             pmids = pmids[:5]

#     # graph = _get_articles_graph(pmids, unmapped_ents, events, co_occur)
#     graph = _get_articles_graph(pmids, unmapped_ents, events, co_occur)
#     return graph


def _get_vocab_path(nids, k=1):
    with open(config.cypher_file) as f:
        content = json.load(f)
        # top 5 important neighbors
        cypher = content["search"]['/expand_single_vocab']
    gds = GraphDataScience(config.glkb_uri, auth=("neo4j", "password"))
    G = gds.graph.get(config.vocab_graph_projection)
    paths = []
    if len(nids) > 1:
        for i in range(len(nids)-1):
            for j in range(i, len(nids)):
                res = gds.shortestPath.yens.stream(
                    G, sourceNode=nids[i], targetNode=nids[j], k=k, relationshipWeightProperty='neg_log_importance',
                    # nodeLabels=['Chemicals and Drugs', 'Diseases', 'Genes and Gene Products', 'Pathway'], 
                    # relationshipTypes=['Curated_relationship', 'Hierarchical_structure', 'Semantic_relationship']
                    )
                if len(res) > 0:  # have path
                    for l in res['nodeIds'].to_list():
                        paths += l
    elif len(nids) == 1:
        with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
            with driver.session() as session:
                paths += nids
                paths += session.run(cypher.format(nids[0])).value().copy()
    return list(set(paths))

def _get_vocab_graph0(nids, self_loop=True, merge_edge=True, event_types=':Contain_vocab|cite|Curated_relationship|Hierarchical_structure|Semantic_relationship'):  # |co_occur
    list_of_nodes = nids
    # all_nodes = {nid:0 for nid in nids}
    triplets = _get_triplets(list_of_nodes, list_of_nodes, event_types=False)
    graph = _generate_graph(triplets, self_loop=self_loop)
    if merge_edge:
        graph = _merge_graph_edge(graph)
    # article_list = _get_articles_from_rels(triplets, max_article=3)
    # return graph, article_list
    return graph

def _get_vocab_graph3(nids, k=3, self_loop=True, merge_edge=True, event_types=':Contain_vocab|cite|Curated_relationship|Hierarchical_structure|Semantic_relationship'):  # |co_occur
    list_of_nodes = _get_vocab_path(nids, k=k)
    # all_nodes = {nid:0 for nid in nids}
    triplets = _get_triplets(list_of_nodes, list_of_nodes, event_types=False)
    graph = _generate_graph(triplets, self_loop=self_loop)
    if merge_edge:
        graph = _merge_graph_edge(graph)
    # article_list = _get_articles_from_rels(triplets, max_article=3)
    # return graph, article_list
    return graph

def _get_vocab_graph2(nids, sets=['gene', 'disease', 'go', 'pathway', 'chemical'], k=7, self_loop=True, merge_edge=True, event_types=':Contain_vocab|cite|Curated_relationship|Hierarchical_structure|Semantic_relationship'):  # |co_occur
    # list_of_nodes = _get_vocab_path(nids, k=k)
    list_of_nodes = nids.copy()
    all_nodes = {nid:0 for nid in nids}
    for s in sets:
        # app.logger.info(f'nids: {nids}, set: {s}')
        l, pvals = _enriched_terms(nids, enriched_sets.get(s), top_k=100)
        # app.logger.info(f'set: {s}, enriched ids: {l}')
        list_of_nodes += l[:k]
        # all_nodes += list(zip(l, pvals))
        all_nodes.update(dict(zip(l, pvals)))
    triplets = _get_triplets(list_of_nodes, list_of_nodes, event_types=False)
    graph = _generate_graph(triplets, self_loop=self_loop)
    if merge_edge:
        graph = _merge_graph_edge(graph)
    # article_list = _get_articles_from_rels(triplets, max_article=3)
    # return graph, article_list
    return graph, list(all_nodes.items())

def _get_vocab_graph(nids, k=7, self_loop=False, merge_edge=True):
    list_of_nodes = [str(i) for i in nids]
    all_nodes = dict()

    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["projection"]['louvain']
    
    graph_name = _create_projection(nids)
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(graph_name)).values().copy()

    df = pd.DataFrame(result, columns=['id', 'name', 'type', 'n_citation', 'degree', 'comm'])
    df['type'] = df['type'].apply(lambda x: '; '.join(list(set(x).difference({"Vocabulary"}))))
    df['id'] = df['id'].astype(str)
    # largest comms
    largetst_comms = list(df['comm'].value_counts().nlargest(1).index)
    largetst_comms = df[df['comm'].isin(largetst_comms)]
    largetst_comms['type'] = largetst_comms['type'].astype(str)
    largetst_group = largetst_comms.groupby('type')
    # high citation terms
    enhanced_types = ['Genes and Gene Products', 'Pathway', 'Diseases']
    high_citation_group = df[df['type'].isin(enhanced_types)].groupby('type')
    # target comm
    target_comm = pd.unique(df.loc[df.id.isin(list_of_nodes), 'comm'])
    df = df[df['comm'].isin(target_comm)] # target comms
    # df['type'] = df['type'].astype(str)
    group = df.groupby('type')
    all_nodes.update({v.get('id'):v for v in group.apply(pd.DataFrame.nlargest, n=50, columns=['n_citation', 'degree']).reset_index(drop=True)[['id', 'name', 'type']].to_dict(orient='index').values()})
    all_nodes.update({v.get('id'):v for v in largetst_group.apply(pd.DataFrame.nlargest, n=50, columns=['n_citation', 'degree']).reset_index(drop=True)[['id', 'name', 'type']].to_dict(orient='index').values()})
    all_nodes.update({v.get('id'):v for v in high_citation_group.apply(pd.DataFrame.nlargest, n=10, columns=['n_citation', 'degree']).reset_index(drop=True)[['id', 'name', 'type']].to_dict(orient='index').values()})
    # all_nodes.update([(i, 0) for i in list(group.apply(pd.DataFrame.nlargest, n=50, columns='degree').reset_index(drop=True)['id'])])
    # all_nodes.update([(i, 0) for i in list(largetst_group.apply(pd.DataFrame.nlargest, n=50, columns='degree').reset_index(drop=True)['id'])])
    list_of_nodes += list(group.apply(pd.DataFrame.nlargest, n=k, columns=['n_citation', 'degree']).reset_index(drop=True)['id'])
    list_of_nodes += list(high_citation_group.apply(pd.DataFrame.nlargest, n=5, columns=['n_citation', 'degree']).reset_index(drop=True)['id'])
    list_of_nodes = list(set(list_of_nodes))
    _delete_projection(graph_name)

    # high freqs
    l, pvals = _enriched_terms(list(all_nodes.keys()), enriched_sets.get('high_freq'), top_k=500)
    enriched_high_freq_terms = [int(_id) for _id, pval in zip(l, pvals) if pval<5e-2]
    high_freq_info = []
    for i in enriched_high_freq_terms:
        d = enriched_set_feats.get('high_freq').get(str(i))
        if d:
            high_freq_info.append({
                'id': str(i),
                'name': d[0],
                'type': d[1]
            })
    
    all_nodes.update([(i['id'], i) for i in high_freq_info])
    list_of_nodes += enriched_high_freq_terms

    list_of_nodes = [int(i) for i in list_of_nodes]
    triplets = _get_triplets(list_of_nodes, list_of_nodes, event_types=False)
    graph = _generate_graph(triplets, self_loop=self_loop, key_nodes=[str(i) for i in nids])
    if merge_edge:
        graph = _merge_graph_edge(graph)
    
    # update all nodes with initial nodes
    for n in graph['nodes']:
        if n['id'] in [str(i) for i in nids]:
            all_nodes[n['id']] = {
                'id': n['id'],
                'name': n['name'],
                'type': n['label']
            }

    # article_list = _get_articles_from_rels(triplets, max_article=3)
    # return graph, article_list
    return graph, list(all_nodes.items())

def _hypergeom_test(sample, target_set, total_nodes=108508):
    hit = set(sample).intersection(set(target_set))
    return stats.hypergeom.sf(len(hit)-1, total_nodes, len(target_set), len(sample))

def _enriched_terms(sample, d, top_k=10):
    if not d:
        return [], []
    res = {}
    for k, vs in d.items():
        res[int(k)] = _hypergeom_test(sample, vs)
    ids = sorted(res, key=res.get, reverse=False)

    return ids[:top_k], [res[i] for i in ids[:top_k]]

def _get_articles_from_rels(triplets, max_article=3):
    c = Counter()
    all_articles = {}
    article_list = []
    for h, r, t in triplets:
        source = r.get('Source')
        if source:
            if isinstance(source, list):
                c.update(source)
                all_articles[r.element_id] = source
            elif isinstance(source, str):
                if source.isnumeric():
                    source = [source]
                    c.update(source)
                    all_articles[r.element_id] = source
    
    for k, v in all_articles.items():
        if len(v) <= max_article:
            article_list += v
        else:
            article_list += sorted(v, key=lambda x:c[x], reverse=True)[:max_article]
    return article_list

def _pmids2articles(pmids):
    related_articles = []
    article_list = _get_article_node_id(pmids)
    for n in article_list:
        pmid = n['PubmedID']
        date = n['Pubdate']
        # authors = n['Authors']
        title = n['Title']
        n_citation = n['n_citation']
        if not n_citation:
            n_citation = 0
        # if pmid and date and authors:
        if pmid and date and title:
            related_articles.append((f"{title} ({date})", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", n_citation, date)) # authors[0].split(' ')[-1]
    return related_articles

def _pmids2pubdates(pmids):
    related_articles = []
    article_list = _get_article_node_id(pmids)
    for n in article_list:
        date = n['Pubdate']
        if date:
            related_articles.append(date)
    return related_articles

def _create_projection(nids):
    """
    create neighborhood projection of a list of nodes
    """
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["projection"]["neighborhood"]
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            graph_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
            session.run(cypher.format(nids, nids, graph_name))
            # app.logger.info(f'creating projection: {cypher.format(nids, nids, graph_name)}')
    return graph_name

def _delete_projection(graph_name):
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["projection"]['delete']
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            session.run(cypher.format(graph_name))
    return None

def _single_node_graph(_ids):
    graph = {
        "nodes": [],
        "links": []
    }

    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["search"]['/node']
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            for _id in _ids:
                result = session.run(cypher.format(_id)).value().copy()
                n = result[0]
   
                node_info = {
                    "id": n.element_id,
                    # "name": n['Name'],
                    "label": set(n.labels).difference({"Vocabulary"}).pop(),
                    # "display": n['Name'],
                    "frequency": 1,
                    "is_vocab": "false"
                    }
                if "Article" in n.labels:
                    node_info["display"] = f"{n['Authors'][0]} et al ({str(n['Pubdate'])})"
                    # node_info["frequency"] = None
                    node_info["name"] = n["Title"]
                    node_info["source"] = "PubMed"
                    node_info["date"] = n["Pubdate"]
                    if n["n_citation"]:
                        node_info["n_citation"] = n["n_citation"]
                    else:
                        node_info["n_citation"] = 0
                elif "Unmapped_entity" in n.labels:
                    if len(n["Text"]) <= 20:
                        node_info["display"] = n["Text"]
                    else:
                        node_info["display"] = f'{n["Text"][:5]}...{n["Text"][-5:]}'
                    node_info["name"] = n["Text"]
                    node_info["source"] = n["Source"]
                    if n["is_event"] == "False":  # unmapped ents
                        node_info["label"] = "Entity"
                    else:  # events
                        node_info["label"] = "Event"
                        node_info["type"] = n["Type"]
                        node_info['roles'] = []  # list of ents in events
                elif "Vocabulary" in n.labels:
                    if len(n["Name"]) <= 20:
                        node_info["display"] = n["Name"]
                    else:
                        node_info["display"] = f'{n["Name"][:5]}...{n["Name"][-5:]}'
                    node_info["name"] = n["Name"]
                    node_info["source"] = n["Element ID"].split("_")[0]
                    node_info["is_vocab"] = "true"
                    if n["n_citation"]:
                        node_info["n_citation"] = n["n_citation"]
                    else:
                        node_info["n_citation"] = 0
                graph["nodes"].append(node_info)
    
    return graph

def _get_rel_text_pmid(head_ids, tail_ids, pmid, level='sentence', semantic=True):
    assert level in ['abstract', 'sentence']
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["pmid2id"]['vocab']
        cypher2 = content["pmid2id"]['article']
    
    head_text = defaultdict(list)
    head_sents = defaultdict(list)
    tail_text = defaultdict(list)
    tail_sents = defaultdict(list)
    
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(pmid)).values().copy()
            result2 = session.run(cypher2.format(pmid)).value().copy()

    for r, v in result:
        if r['Text']:
            if v['Element ID'] in head_ids:
                head_text[v['Element ID']] += r['Text'].split('|')
            if v['Element ID'] in tail_ids:
                tail_text[v['Element ID']] += r['Text'].split('|')

    if len(result)>0 and len(result2)>0:
        for r, v in result:
            if r['Text']:
                if v['Element ID'] in head_ids:
                    head_text[v['Element ID']] += r['Text'].split('|')
                if v['Element ID'] in tail_ids:
                    tail_text[v['Element ID']] += r['Text'].split('|')

        sents = sent_tokenize(f"{result2[0]['Title']}. {result2[0]['Abstract']}")

        for term, texts in head_text.items():
            for text in texts:
                for i, sent in enumerate(sents):
                    matches = find_near_matches(text, sent, max_l_dist=int(len(text)*0.3))
                    if len(matches) > 0:
                        head_sents[term].append(i)
        for term, texts in tail_text.items():
            for text in texts:
                for i, sent in enumerate(sents):
                    matches = find_near_matches(text, sent, max_l_dist=int(len(text)*0.3))
                    if len(matches) > 0:
                        tail_sents[term].append(i)
        
        if level=='sentence':
            spans = dict()
            if len(head_sents)>=1 and len(tail_sents)>=1:
                for h in set(itertools.chain.from_iterable(head_sents.values())):
                    for t in set(itertools.chain.from_iterable(tail_sents.values())):
                        if abs(h-t)<2:
                            spans[(min(h,t),max(h,t)+1)] = abs(h-t)
                return [' '.join(sents[span[0]:span[1]]) for span in spans]
        elif level=='abstract':
            if len(head_sents)>=1 and len(tail_sents)>=1:
                return [' '.join(sents)]

def _get_rel_text(heads, tails, level='sentence', semantic=True, curated=True, max_return=100):
    head_eids = [i[1] for i in heads]
    tail_eids = [i[1] for i in tails]
    head_ids = [i[0] for i in heads]
    tail_ids = [i[0] for i in tails]

    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["node2graph"]['general']
    
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(head_ids, tail_ids)).values().copy()


    clues = defaultdict(list)
    pmids = set()
    for _, r, _ in result:
        h = r.nodes[0]
        t = r.nodes[1]
        if semantic:
            if r.type=='Semantic_relationship':
                if r['Source']:
                    pmids.update(r['Source'])
        if curated:
            if r.type=='Curated_relationship' or r.type=='Hierarchical_structure':
                if r['Type']:
                    if r['Source']:
                        clues[r['Source']].append(f"{h['Name']} {r['Type'].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))} {t['Name']}.")
                    else:
                        clues['Others'].append(f"{h['Name']} {r['Type'].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))} {t['Name']}.")

    for pmid in pmids:
        if len(clues)<max_return:
            clue = _get_rel_text_pmid(head_eids, tail_eids, pmid, level=level)
            if isinstance(clue, list):
                clues[f"pmid:{pmid}"] += clue
    
    return clues

def _get_single_event_text(_id, level='sentence'):
    assert level in ['abstract', 'sentence']
    with open(config.cypher_file) as f:
        content = json.load(f)
        cypher = content["node2graph"]["event_descendants"]
        cypher2 = content["pmid2id"]['article']
        cypher3 = content["pmid2id"]['vocab']
    
    term_text = defaultdict(list)
    sent_terms = defaultdict(set)
    pmid = None
    vids = []

    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            result = session.run(cypher.format(_id)).value().copy()[0]
            if len(result)==0:
                return []
            for n in result:
                if n.element_id==str(_id):
                    pmid = n['Source']
                if 'Unmapped_entity' in n.labels:
                    term_text[n.element_id].append(n['Text'])
                elif 'Vocabulary' in n.labels:
                    vids.append(n.element_id)
            
            # entitiy text
            result3 = session.run(cypher3.format(pmid)).values().copy()
            for r, v in result3:
                if r['Text']:
                    if v.element_id in vids:
                        term_text[v.element_id] += r['Text'].split('|')
                
            result2 = session.run(cypher2.format(pmid)).value().copy()
            if len(result2)==0:
                return []
    sents = sent_tokenize(f"{result2[0]['Title']}. {result2[0]['Abstract']}")

    for term, texts in term_text.items():
        for text in texts:
            for i, sent in enumerate(sents):
                matches = find_near_matches(text, sent, max_l_dist=int(len(text)*0.3))
                if len(matches) > 0:
                    sent_terms[i].add(term)
    
    if level=='sentence':
        spans = dict()
        for h, t in itertools.combinations_with_replacement(sent_terms, 2):
            if abs(h-t)<2:
                if sent_terms[h].union(sent_terms[t]) == set(term_text.keys()):
                    spans[(min(h,t),max(h,t)+1)] = abs(h-t)
        # for s in sent_terms:
        #     if sent_terms[s] == set(term_text.keys()):
        #         spans[(s,s+1)] = 1
        if len(spans)>0:
            span = min(spans, key=spans.get)
            return [' '.join(sents[span[0]:span[1]])]
        else:
            return []
    elif level=='abstract':
        return [' '.join(sents)]

def _get_event_text(terms, event_type=None, level="sentence"):    
    term_eids = [i[1] for i in terms]
    term_ids = [i[0] for i in terms]
    assert event_type in ["Regulation", "Positive_regulation", "Negative_regulation", "Binding", "Phosphorylation", "Gene_expression", "Localization", "Transcription", "Protein_catabolism", None]

    with open(config.cypher_file) as f:
        content = json.load(f)
        if not event_type:
            cypher = content["node2graph"]["events"]
        else:
            cypher = content["node2graph"]['events_type']
    
    # query database
    with GraphDatabase.driver(config.glkb_uri, auth=("neo4j", "password"), max_connection_lifetime=1000) as driver:
        with driver.session() as session:
            if not event_type:
                result = session.run(cypher.format(term_ids)).value().copy()
            else:
                result = session.run(cypher.format(term_ids, event_type)).value().copy()
            if len(result)==0:
                return defaultdict(list)
    clues = defaultdict(list)
    for event in result:
        clues[f"pmid:{event['Source']}"] = _get_single_event_text(event.element_id, level=level)
    return clues

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=config.debug_port)
