#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import math
import collections
import itertools
import os
import errno

try:
	from yahmm import *
except ImportError:
	print "yahmm module not installed! '> pip install yahmm'"
	exit(-1)

import networkx as nx  # Usually installed with yahmm


def _make_node_label(_node):
	""" Create node label from assigned distribution parameters.
	:param _node: A nx.DiGraph.Node()
	:return: Node label
	"""

	def pretty_param(_dist, _sep='\n', _short=False):
		p = _dist.parameters
		if isinstance(_dist, NormalDistribution):
			fmt = '%.2f'
			mu = (fmt % p[0]).rstrip('0').rstrip('.')  # Remove trailing zeros
			sigma = (fmt % p[1]).rstrip('0').rstrip('.')
			if _short:
				ret = [mu, _sep, sigma]
			else:
				ret = ['mu= ', mu, _sep, 'sig= ', sigma]
			return "".join(ret)
		else:
			return str(_node.distribution.parameters)

	try:
		p = _node.distribution.parameters
		if isinstance(_node.distribution, MultivariateDistribution):
			dists, weights = p
			return "\n".join([_node.name] + [pretty_param(d, _sep='   ', _short=True) for d in dists])
		else:
			return "\n".join([_node.name, pretty_param(_node.distribution)])
	except AttributeError:
		return _node.name


def _preformat_graph(_g, _graph=None, _grouped=None):
	""" Assign node colors and node layout intention.

	:param _g: A nx.DiGraph()
	:param _graph: Propagate graph attributes if specified.
	:param _grouped: List of grouped nodes; useful for nicer node alignment
	:return: None
	"""

	# Short GraphViz introduction: http://4webmaster.de/wiki/Graphviz-Tutorial
	if _grouped is None:
		_grouped = []

	# Graph formatting
	_g.graph['graph'] = {  # There is no wrapper method; need to modify this directly
		'rankdir': 'LR'    # Draw from left to right
	}
	if _graph:
		_g.graph['graph'].update(_graph)

	# Node formatting
	node_labels = {}
	node_colors = {}  # GraphViz X11 scheme colors: http://www.graphviz.org/doc/info/colors.html
	lut = {
		'-start': 'palegreen',
		'-end': 'lightcoral',
		'-meta': 'lightblue',
		'-tm': 'lightyellow'
	}
	for node in _g.nodes():
		node_labels[node] = _make_node_label(node)
		node_colors[node] = ([lut[key] for key in lut if key in node.name] + ['white'])[0]  # Default to 'white' if no keyword found in node name
	nx.set_node_attributes(_g, 'label', node_labels)
	nx.set_node_attributes(_g, 'style', 'filled')  # Need this, fillcolor has no effect otherwise
	nx.set_node_attributes(_g, 'fillcolor', node_colors)
	# nx.set_node_attributes(_g, 'shape', 'circle')

	# Edge Formatting
	try:  # Check if we have weights -> draw HMM graph with weights as labels; otherwise use existing labels
		edge_labels = {}
		for u, v, data in _g.edges(data=True):
			p = math.exp(data['trans_weight'])
			edge_labels[(u, v)] = '' if round(p, 6) == 1.0 else ("%.2f" % p)[1:]  # 2 Decimals, no leading zero
		nx.set_edge_attributes(_g, 'label', edge_labels)
	except KeyError:
		pass
	# nx.set_edge_attributes(_g, 'splines', 'curved')

	# Add edge weights for grouped nodes
	# id_lut = {n.identity: n for n in _g.nodes()}
	# for n, group in enumerate(_grouped):
	# nx.set_node_attributes(_g, 'group', {id_lut[node.identity]: 'group_' + str(n) for node in group})  # Using 'group' node attribute
	seq_pairs = {}
	edges = _g.edges()
	for group in _grouped:
		seq_pairs.update({(u.identity, v.identity) for u, v in zip(group[:-1], group[1:])})  # a -> b -> c -> d  TO  (a->b), (b->c), (c->d)
	edge_weights = {}
	for u, v in edges:
		if (u.identity, v.identity) in seq_pairs:
			edge_weights[(u, v)] = '100'
	nx.set_edge_attributes(_g, 'weight', edge_weights)


def plot_graph(_g, _show=True, _output_file=None, _output_format=None, _output_dpi=150, _layout_prog='dot'):
	""" Create a rendered representation of a nx.DiGraph() or yahmm model graph and optionally display it.

	:param _g: A nx.DiGraph()
	:param _show: Whether or not to plot the generated graph.
	:param _output_file: Output file path of the rendered graph.
	:param _output_format: Image file format; 'png' on default or derived from _output_file
	:param _output_dpi: DPI of rendered graph
	:param _layout_prog: GraphViz layout tool; 'dot' on default
	:return: A pygraphviz graph.
	"""

	if _output_format is None:
		_output_format = 'png'
	if _output_file is None:
		_output_file = 'tmp.' + _output_format
	elif isinstance(_output_file, str) and '.' not in _output_file:
		_output_file += '.' + _output_format

	# Update graph attributes
	graph_attr = {
		'dpi': _output_dpi  # Vector graphics are converted to raw images on load; annotate with desired resolution
	}
	if 'graph' not in _g.graph:
		_g.graph['graph'] = graph_attr
	else:
		_g.graph['graph'].update(graph_attr)

	import time
	for u, v, d in _g.edges_iter(data=True):
		d['key'] = d.get('key', time.clock())

	a = nx.to_agraph(_g)
	a.layout(prog=_layout_prog)

	# Ensure that intermediate folders exist
	try:
		path = os.path.abspath(os.path.join(_output_file, os.pardir))
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

	a.draw(_output_file, format=_output_format)

	if _show:
		im = plt.imread(_output_file)  # Windows PIL GhostScript fix: https://stackoverflow.com/a/13101970/1507220
		plt.imshow(im, interpolation="quadric", resample=True)
		plt.axis('off')
		plt.tight_layout()

	return a


def from_sequence_make_graph(_indexed_state_seq, _drop_states=None, **_kwargs):
	""" Create a linear graph out of a state sequence.
	:param _indexed_state_seq: The indexed state sequence can be obtained from a decoded sequence.
	:param _drop_states: State instances which should not be shown in the final graph.
	:param _kwargs: Parameters passed to _preformat_graph()
	:return: A nx.DiGraph()
	"""

	state_seq = map(lambda x: x[-1], _indexed_state_seq)  # Extract nodes from ([sample,] idx, node) pair list
	if _drop_states:
		state_seq = [s for i, s in enumerate(state_seq) if i in [0, len(state_seq) - 1] or s not in _drop_states]

	grouped_state_seq = [(k, len(list(g))) for k, g in itertools.groupby(state_seq)]  # Group & count consecutive states
	grouped_state_seq = [(u.tied_copy(), n) for u, n in grouped_state_seq]  # Create unique states
	state_seq = list(itertools.chain.from_iterable([itertools.repeat(u, n) for u, n in grouped_state_seq]))  # Rebuild state sequence
	state_trans = zip(state_seq[:-1], state_seq[1:])  # Generate list of state transitions; this is a list of edges
	trans_cnt = collections.Counter(state_trans).items()  # Get a list of (edge, count) pairs
	annotated_edges = map(lambda (e, cnt): e + ({'label': cnt},), trans_cnt)  # (u,v, {'label': count}) edge tuples

	g = nx.DiGraph()
	g.add_nodes_from(set(state_seq))
	g.add_edges_from(annotated_edges)

	_preformat_graph(g, **_kwargs)

	return g


def from_model_make_graph(_model, **_kwargs):
	""" Creates a graph from dense matrix of a yahmm model.
	:param _model: A yahmm model.
	:param _kwargs: Parameters passed to _preformat_graph()
	:return: A nx.DiGraph().
	"""

	class DotState(collections.namedtuple("state", ['name', 'identity', 'distribution', 'hmm_weight'])):  # Custom named tuple for readable '.dot' files
		__slots__ = ()
		def __str__(self):
			return " ".join([self.name, ' ', self.identity])
	states = [DotState(name=s.name, identity=s.identity, distribution=s.distribution, hmm_weight=s.weight) for s in _model.states]
	mat_trans = _model.dense_transition_matrix()
	sparse_edges = [((states[n], states[m]), p) for n, x in enumerate(mat_trans) for m, p in enumerate(x) if not math.isinf(p)]
	annotated_edges = map(lambda (e, log_p): e + ({'trans_weight': log_p}, ), sparse_edges)  # (u, v, {'weight': probability}) edge tuples

	g = nx.DiGraph()
	g.add_nodes_from(states)
	for u, v, data in annotated_edges:
		g.add_edge(u, v, data)

	_preformat_graph(g, **_kwargs)

	return g


def backannotate_internal_graph(_model):
	""" bake() propagates the graph data to the internal matrix; back-annotation behaves in the opposite way.
	:param _model: The model to update.
	:return: None
	"""

	mat_trans = _model.dense_transition_matrix()
	states = _model.states
	dense_edges = [(states[n], states[m], p) for n, x in enumerate(mat_trans) for m, p in enumerate(x)]
	edges_to_remove = [(u, v) for u, v, p in dense_edges if math.isinf(p)]
	edges_to_update = [(u, v, p) for u, v, p in dense_edges if not math.isinf(p)]

	g = _model.graph
	g.remove_edges_from(edges_to_remove)
	for u, v, p in edges_to_update:
		g.add_edge(u, v, {'weight': p})


def sync_weights(_dest_model, _src_model, _backannotate_before=True, _bake_afterwards=True):
	""" Synchronizes weights from a source to a destination model.
	:param _dest_model: The destination yahmm model.
	:param _src_model:  The source yahmm model.
	:param _backannotate_before: Whether or not to back-annotate the source model first; 'True' on default.
	:param _bake_afterwards: Whether or not to bake the destination model after sync ; 'True' on default.
	:return: None
	"""

	if _backannotate_before:
		backannotate_internal_graph(_src_model)

	src = _src_model.graph
	dest = _dest_model.graph

	edges = [(u.name, v.name, data) for u, v, data in src.edges(data=True)]
	old_edges = [(u.name, v.name) for u, v in dest.edges()]
	lut = {n.name: n for n in dest.nodes()}

	for u, v, data in edges:
		if (u, v) in old_edges:
			dest.add_edge(lut[u], lut[v], {'weight': data['weight']})

	if _bake_afterwards:
		_dest_model.bake(merge=None)


def plot_model(_model, **_kwargs):
	""" Plot a single or multiple yahmm models.
	:param _model: A list of or a single yahmm model.
	:param _kwargs: Parameters to pass to plot_graph()
	:return: None
	"""

	kwargs = dict(_kwargs)  # Force copy
	if '_output_file' not in kwargs:
		if isinstance(_model, list):
			kwargs['_output_file'] = 'dump/'
		else:
			kwargs['_output_file'] = 'dump/' + _model.name

	def plot_generic():
		plt.figure()
		g = from_model_make_graph(_model, _graph={
			'ranksep': 0.5,  # Space between nodes
			'nodesep': 0.5  # This mainly affects loops
		})
		plot_graph(g, **kwargs)

	def plot_group():
		for m in _model:
			plt.figure()
			g = from_model_make_graph(m, _graph={
				'ranksep': 0.5,  # Space between nodes
				'nodesep': 0.5  # This mainly affects loops
			})
			tmp_kwargs = dict(kwargs)
			tmp_kwargs['_output_file'] = kwargs['_output_file'] + m.name
			plot_graph(g, **tmp_kwargs)

	def plot_threshold():
		plt.figure()
		g = from_model_make_graph(_model, _graph={
			'ranksep': 0.5,  # Space between nodes
			'overlap': 'scale',
			'sep': 3,
			'nodesep': 0.5  # This mainly affects loops
		})
		plot_graph(g, _layout_prog='neato', **kwargs)

	if isinstance(_model, list):
		plot_group()
	else:
		plot_generic()


def from_data_make_sequence(_df, _models, _decoder=None, _show=True, _drop_states=None, _output_prefix=None, **_kwargs):
	""" Decode a data set with the specified model(s) and optionally plot the result graph. Can be post-processed with from_sequence_make_segments().
	:param _df: A DataFrame to be decoded.
	:param _models: A list of or a single yahmm model.
	:param _decoder: The algorithm used for decoding; can be yahmm.Model.viterbi (default) or yahmm.Model.maximum_a_posteriori
	:param _show: Whether or not to display the resulting graph; 'True' on default.
	:param _drop_states: A list of states to be excluded in the result graph.
	:param _output_prefix: Output destination prefix.
	:param _kwargs: Parameters passed to plot_graph()
	:return: The decoded sequence for each specified model.
	"""

	if not hasattr(_models, '__iter__'):
		_models = [_models]
	if _decoder is None:
		_decoder = Model.viterbi  # Model.maximum_a_posteriori # WARNING: MAP may produce impossible sequences.
	if _drop_states is None:
		_drop_states = []
	if _output_prefix is None:
		_output_prefix = ''

	test_seq = _df.to_records(index=False).tolist()

	# Decode for all provided models
	decoded = [(m, _decoder(m, test_seq)) for m in _models]

	# Lift with sample index
	def lift(_stream, _idx):
		return [(None if state.is_silent() else next(_idx), n, state) for n, state in _stream]
	decoded = [(m, (log_likeli, lift(idx_state_seq, iter(_df.index)))) for m, (log_likeli, idx_state_seq) in decoded]

	print 'Log-likelihoods of decoded test sequence'
	max_name_len = max(len(m.name) for m, _ in decoded)  # Just for pretty-printing
	for m, (log_likeli, indexed_state_seq) in decoded:
		print ("Model %" + str(max_name_len) + "s: %9.4f") % (m.name, log_likeli)

	if _show:
		for m, (log_likeli, indexed_state_seq) in decoded:
			plt.figure()
			seq = from_sequence_make_graph(indexed_state_seq, _drop_states=_drop_states)
			if '_output_file' not in _kwargs:
				_kwargs['_output_file'] = _output_prefix + ' ' + m.name + ' path.png'
			plot_graph(seq, **_kwargs)

	return decoded


def from_sequence_make_segments(_decoded, _model_to_label):
	""" Create a segments table from a decoded sequences produced by from_data_make_sequence().
	Each table entry corresponds to a sequence which fits entirely into one of the specified models.

	:param _decoded: The decoded sequence.
	:param _model_to_label: A model->label mapping which provides the label ID for a detected model sequence.
	:return: A list of segment tables
	"""
	state_to_model = {}
	for m in _model_to_label.keys():
		for s in m.states:
			state_to_model[s] = m

	dfs = []
	for m, (ll, seq) in _decoded:

		# Add originating model to state sequence
		seq_lifted = [(idx, n_state, state, state_to_model.get(state)) for idx, n_state, state in seq]

		# Split sequences represented by models from '_model_to_id'
		# Based on https://stackoverflow.com/a/4322780
		split_seq = [list(g) for k, g in itertools.groupby(seq_lifted, lambda x: x[3] is None) if not k]

		# Reduce splits to intervals with model ID
		def to_interval(_seq):
			start, _, _, model = itertools.ifilter(lambda (sample, n_state, state, model): sample is not None, _seq).next()
			end, _, _, _ = itertools.ifilter(lambda (sample, n_state, state, model): sample is not None, reversed(_seq)).next()
			return start, end, end-start, _model_to_label[model]
		dfs.append([to_interval(x) for x in split_seq])

	# Transform to nice dataframe; create empty frames if no match detected
	cols = ['Begin', 'End', 'Duration', 'ID']
	dfs = [pd.DataFrame(df or None, columns=cols) for df in dfs]

	return dfs


__author__ = "Martin Freund"
__email__ = "freund@fim.uni-passau.de"
__copyright__ = "Copyright 2015, ACTLab"
