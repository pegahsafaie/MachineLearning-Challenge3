#!/usr/bin/python
# -*- coding: utf-8 -*-

from yahtools import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For legend
import math
import pandas as pd
import dataset


# Increase font sizes in plots
params = {
	'axes.titlesize': 20,  # Plot title
	'axes.labelsize': 20,  # Axis label
	'xtick.labelsize': 15,
	'ytick.labelsize': 15,
	'legend.fontsize': 15,
	'font.size': 15  # Default text
}
plt.rcParams.update(params)


def make_nd_distribution(_n, _type, _params):
	""" Create n-dimensional distributions from 1-dimensional base type.
	:param _n: Dimension of the data stream.
	:param _type: Any distribution from yahmm module or custom ones.
	:param _params: Common parameters provided to the distribution constructor.
	:return: The created distribution.
	"""
	if not isinstance(_params[0], list):
		_params = [_params] * _n
	return MultivariateDistribution([_type(*(_params[dim])) for dim in range(_n)])


def plot_labels(_segments, _label_to_color, _ax=None, _alpha=0.2):
	""" Highlight regions by label in current plot
	:param _segments: Segments DataFrame containing labels
	:param _label_to_color: Dict containing label->color mappings.
	:param _ax: ax for drawing; current ax if None
	:param _alpha: Alpha channel of highlight
	:return: None
	"""
	_ax = _ax or plt
	for idx, interval in _segments.iterrows():
		_ax.axvspan(interval['Begin'], interval['End'], facecolor=_label_to_color[interval['ID']], alpha=_alpha)


def add_labels_to_legend(_label_to_color, _ax=None):
	""" Append legend for label spans; see plot_labels()

	:param _label_to_color: Dict containing label->color mappings.
	:param _ax: Ax containing the legend; current ax if None
	:return: None
	"""
	if _ax is None:
		_ax = plt.gca()

	handles, labels = _ax.get_legend_handles_labels()

	patches = []
	for label, color in _label_to_color.items():
		try:
			name = dataset.class_name(label)
		except KeyError:
			name = 'Label'
		patches.append(mpatches.Patch(color=color, alpha=0.5, label=name + ' (%d)' % label))
	_ax.legend(handles=handles + patches)


def run(_store):
	""" Build. training, test and evaluate HMM.

	:param _store: HDFStore providing data
	:return: None
	"""

	# ######################################################  FRAMEWORK SETTINGS  ######################################################

	# TODO: Play with these settings
	data_column_names = ['RLAaccx', 'RLAaccy', 'RLAaccz']  # Right hand accelerometer; check dataset.py for feature names
	p_TM = 0.6  # p(TM) - the entry probability of the garbage HMM; p(G) = (1 - p(TM)); See Lee & Kim paper
	extension_len = 0  # Additional samples before and after the training label to compensate for cut-offs

	# No need to change these
	output_dir = 'img/'  # Directory is needed for intermediate output when plotting
	data_dimension = len(data_column_names)
	# Available labels:
	# - 17: nested label, separates training data from free-living data, this should NOT be used directly
	# - 18: drinking gesture
	# - 19: drinking gesture
	# - 20: drinking gesture
	# - 21: drinking gesture
	spotter_label = 90  # Virtual label for marking spotted gestures

	# Label color lookup table; 'spotter_label' is for the spotted events
	label_to_color = {17: 'c', 18: 'b', 19: 'g', 20: 'r', 21: 'm', spotter_label: 'y'}

	# ######################################################  TRAINING DATA  ######################################################
	print '\nExtracting training data ...'

	# Fetch the TRAINING dataset, this is a list of data frames
	# TODO: Increase the training pool
	training_participants = [13]  # Can pick from [11, 13, 15]
	training_labels = [18]  # Can pick from [18, 19, 20, 21]; see above
	training_type = 'training'  # Can be initial 'training' data (scripted), 'free-living' (office) data or 'both'
	training_dfs = list(dataset.gen_segments(
		_store, _participants=training_participants,
		_labels=training_labels, columns=data_column_names,
		_extend_front=extension_len, _extend_back=extension_len,
		_type=training_type
		)
	)

	# Plot some (3) training gestures
	# TODO: try this
	if False:
		for i, train_df in zip(range(3), training_dfs):
			train_df.plot()
			plt.title('TRAINING set %d' % i)
			plt.xlabel('Samples')
			plt.ylabel('Raw data')
			plt.tight_layout()
			plt.show()

	# Print some statistics of the training set
	print 'gesture sample statistics:'
	sample_lengths = [df.shape[0] for df in training_dfs]
	print 'training instances:', len(sample_lengths)
	print 'mean length: ', np.average(sample_lengths)
	print 'median len:  ', np.median(sample_lengths)
	print 'min length:  ', min(sample_lengths)
	print 'max length:  ', max(sample_lengths)

	# ######################################################  TESTING DATA  ######################################################
	print '\nExtracting testing data ...'

	# The test data only serves for testing the model and to get a quick look at the data. It is NOT intended for evaluation

	# Fetch TEST data; segments of labels and raw sample data
	test_participant = 11
	test_labels = [18]  # The labels associated with the gesture
	test_type = 'free-living'  # Can be initial 'training' data, 'free-living' data or 'both'
	test_seg = list(dataset.gen_seg_table(_store, _participants=test_participant, _labels=test_labels, _type=test_type))  # For for each subject
	assert test_seg  # Ensure we got data, otherwise the query was invalid
	test_seg = test_seg[0][1]  # Fetch single frame; there is only one
	test_seg = test_seg.iloc[1:2]  # Pick the first 10 segments
	test_min_start, test_max_stop = test_seg.iloc[0][0], test_seg.iloc[-1][1]  # Limit TEST data frame to region of interest
	test_min_start, test_max_stop = test_min_start - 300, test_max_stop + 300  # Add some margin for visibility
	test_df = dataset.get_frame(_store, _participant=test_participant, _start=test_min_start, _stop=test_max_stop, columns=data_column_names)

	# Plot TEST data stream
	# TODO: try this
	if False:
		ax = test_df.plot()
		plot_labels(test_seg, label_to_color)
		lut = {key: col for key, col in label_to_color.iteritems() if key in test_labels}  # Reduce legend to used labels
		add_labels_to_legend(_ax=ax, _label_to_color=lut)
		plt.title('TEST set with ground truth; participant=%d' % test_participant)
		plt.xlabel('Samples')
		plt.ylabel('Raw data')
		plt.tight_layout()
		plt.show()

	# ######################################################  BUILD HMM  ######################################################
	print '\nBuilding the gesture model ...'

	# Example HMM; linear with 3 states; all gaussian
	#
	# model_entry --> state_1  -->  state_2 --> state_3  -->  model_exit
	#                  ^   |         ^   |       ^   |
	#                  \__/          \__/        \__/


	# TODO: Try different models. Models can be nested (i.e., 'model' can have sub-models) as can be seen in the yahmm tutorial.
	model = Model(name='Drink')

	# Gaussian probability distribution function factory
	def make_dist(_mean=0, _std=2):
		return make_nd_distribution(data_dimension, NormalDistribution, [_mean, _std])

	# The name of the State() can be suffixed with color codes:
	# * '-start' (green) and '-end' (red) to mark silent entry and exit points
	# * '-tm' (blue) to mark derived threshold model states
	# * '-meta' (yellow) to mark silent / meta states

	# TODO: Play with the distribution types and the initial parameters, they do not need to be all the same
	model_states = []
	for i in range(3):  # Generate some states
		s = State(make_dist(), name=model.name + "_%d" % (i + 1))
		model_states.append(s)
	model.add_states(model_states)
	model.add_transition(model.start, model_states[0], 1)  # Entry transition
	model.add_transitions(model_states, model_states, [0.8] * len(model_states))  # 80% self loop for each state
	model.add_transitions(model_states, model_states[1:] + [model.end], [0.2] * len(model_states))  # 20% to-next for each state

	# Always bake the model after changes to update internal structures and normalize transition probabilities
	model.bake()

	# TODO: try this
	if False:  # Plot and save visual model
		plot_model(model, _output_file=output_dir + 'drink_initial.png', _output_dpi=200)  # Check plot_graph() for more parameters
		plt.title('Initial gesture HMM')
		plt.show()

	# ######################################################  TRAIN HMM  ######################################################
	print '\nTraining the gesture model ...'

	# Get a list of training instances
	training_stream = [df.to_records(index=False).tolist() for df in training_dfs]

	# Train gesture HMM; If you have sub-models you can train them individually even after combining them
	model.train(training_stream, algorithm='baum-welch', transition_pseudocount=1)

	# Training changes the internal representation, back-annotate changes to internal graph for plotting
	backannotate_internal_graph(model)

	# Plot and save visual model
	# TODO: try this
	if False:
		plot_model(model, _output_file=output_dir + 'drink_trained.png', _output_dpi=200)  # Check plot_graph() for more parameters
		plt.title('Trained gesture HMM')
		plt.show()

	# ######################################################  BUILD THRESHOLD HMM  ######################################################
	print '\nBuilding the threshold model ...'

	# The threshold model contains all emitting states from the gesture model; the threshold model is ergodic
	model_tm = Model(name="threshold")

	# Collect all states and their self-loop transition probabilities
	tm_states = {}
	trans_mat = model.dense_transition_matrix()  # The transition matrix is created during the baking process
	for i, s in enumerate(model.states):
		p = trans_mat[i, i]
		if not math.isinf(p):
			s_ = s.tied_copy()  # New state but same distribution
			s_.name += '-tm'
			tm_states[s_] = math.exp(p)

	# TODO: try to improve the threshold model by adding states
#	# Add a noise state
#	dummy = State(make_nd_distribution(data_dimension, NormalDistribution, [0, 47]), name='1337')
#	tm_states[dummy] = 0.6  # 60% loop probability

	# Create the ergodic graph
	model_tm_meta = State(None, name=model_tm.name + '-meta')  # Virtual node to model ergodic graph
	for s, p in tm_states.items():
		model_tm.add_state(s)
		model_tm.add_transition(s, s, p)  # Loop
		model_tm.add_transition(s, model_tm_meta, 1 - p)  # Return to virtual node
		model_tm.add_transition(model_tm_meta, s, 1)  # Enter state

	# We cannot create a 'start -> meta' as that could lead to a non-emitting 'start -> meta -> end' sequence
	# and consequently, a non-emitting loop in the top level HMM. Thus, force a 'start -> emitting-state' transition
	model_tm.add_transitions(model_tm.start, tm_states.keys(), [1] * len(tm_states))
	model_tm.add_transition(model_tm_meta, model_tm.end, 1)

	# Normalize transition probabilities, do not merge silent states
	model_tm.bake(merge='None')

	# Plot and save visual model
	# TODO: try this
	if False:
		plot_model(model_tm, _output_file=output_dir + 'threshold_model.png', _output_dpi=200)  # Check plot_graph() for more parameters
		plt.title('Threshold HMM')
		plt.show()

	# ######################################################  BUILD TOP LEVEL HMM  ######################################################
	print '\nBuilding the top level model ...'

	#                                (1 - p_TM)    (model)
	#                              /-----------> gesture_hmm -----\
	# TLD_entry  --> meta_entry --O                                O---> meta_exit --> TLD_exit
	#                    ^         \-----------> threshold_hmm  --/          |
	#                    |             (p_TM)     (model_tm)                 |
	#                    \__________________________________________________/

	# Build the TLD model which just combines the gesture and threshold models
	model_tld = Model(name='TLD')
	model_tld.add_model(model)  # Import gesture model as instance
	model_tld.add_model(model_tm)  # Import threshold model as instance

	# Introduce silent states for loops
	model_tld_meta_entry = State(None, name=model_tld.name + ' start-meta')
	model_tld_meta_exit = State(None, name=model_tld.name + ' end-meta')
	model_tld.add_transition(model_tld.start, model_tld_meta_entry, 1)
	model_tld.add_transition(model_tld_meta_exit, model_tld.end, 1)

	# Loopback to start of model; this allows for capturing multiple gestures in one stream
	model_tld.add_transition(model_tld_meta_exit, model_tld_meta_entry, 1)

	# p_enter_tm specifies the probability for choosing the threshold model over the gesture model
	model_tld.add_transition(model_tld_meta_entry, model_tm.start, p_TM)  # Enter threshold model
	model_tld.add_transition(model_tld_meta_entry, model.start, 1 - p_TM)  # Enter gesture model
	model_tld.add_transition(model_tm.end, model_tld_meta_exit, 1)
	model_tld.add_transition(model.end, model_tld_meta_exit, 1)

	# Normalize transition probabilities, do not merge silent states
	model_tld.bake(merge='None')

	# Plot and save visual model
	# TODO: try this
	if False:
		plot_model(model_tld, _output_file=output_dir + 'tld_model.png', _output_dpi=200)  # Check plot_graph() for more parameters
		plt.title('Top level HMM')
		plt.show()

	# ######################################################  TEST HMMs  ######################################################
	print '\nTesting the top level model ...'

	# Hide some of the silent states to increase readability
	drop_states = [model_tld_meta_entry, model_tld_meta_exit, model_tm_meta]

	# Test with limited TRAINING data; only use  sets
	# TODO: try this
	if False:
		for i, train_df in zip(range(3), training_dfs):

			# Decode evaluation set using the top level model; omit meta states for readability when plotting
			path = output_dir + 'TRAINING set %d' % i
			test_decoded = from_data_make_sequence(train_df, _models=model_tld, _drop_states=drop_states, _output_prefix=path, _show=True)
			plt.title('Testing with TRAINING set %d' % i)

			# Convert the sequence to segments (intervals); result is compatible with plot_labels()
			# If we capture a sequence part of the gesture HMM 'model' assign the label ID 'spotter_label'
			test_spotted_seg = from_sequence_make_segments(test_decoded, _model_to_label={model: spotter_label})
			test_spotted_seg = test_spotted_seg[0]  # We only have a single model to evaluate

			# Next plot the data stream
			ax = train_df.plot()

			# The whole set is ground truth, no need for coloring, just color the spotted segment
			plot_labels(test_spotted_seg, _label_to_color=label_to_color, _alpha=0.5)
			lut = {key: col for key, col in label_to_color.iteritems() if key in test_labels + [spotter_label]}  # Reduce legend to used labels
			add_labels_to_legend(_ax=ax, _label_to_color=lut)

			plt.title('Testing with TRAINING set %d' % i)
			plt.xlabel('Samples')
			plt.ylabel('Raw data')

			plt.tight_layout()
			plt.savefig(output_dir + 'testing with TRAINING set %d.png' % i, bbox_inches='tight')
			plt.show()

	# ######################################################  EVALUATE HMM  ######################################################
	print '\nEvaluating the top level model ...'

	# Demonstrate evaluation with test set
	# TODO: Increase the test set to cover more participants.
	eval_labels = test_labels
	eval_dfs = [test_df]  # We can evaluate more than one test DataFrame
	eval_segs = [test_seg]

	# Evaluate each frame individually
	eval_spotted_segs = []
	for i, (eval_df, eval_seg) in enumerate(zip(eval_dfs, eval_segs)):

		# Decode evaluation set using the top level model; omit meta states for readability when plotting
		# TODO: try '_show=True' to plot the decoded sequence as graph
		path = output_dir + 'EVALUATION set %d' % i
		eval_decoded = from_data_make_sequence(eval_df, _models=model_tld, _drop_states=drop_states, _output_prefix=path, _show=True)
		plt.title('EVALUATION set %d' % i)

		# Convert the sequence to segments (intervals); result is compatible with plot_labels()
		# If we capture a sequence part of the gesture HMM 'model' assign the label ID 'spotter_label'
		eval_spotted_seg = from_sequence_make_segments(eval_decoded, _model_to_label={model: spotter_label})
		eval_spotted_seg = eval_spotted_seg[0]  # We only have a single model to evaluate
		eval_spotted_segs.append(eval_spotted_seg)

		# TODO: Implement spotter performance evaluation per frame (based on eval_spotted_seg and eval_seg)
		print 'Evaluation results for set %d ...' % i
		print 'Ground truth'
		print eval_seg
		print 'Spotted segments'
		print eval_spotted_seg
		print ''

		# Plot the evaluation data set with labels
		if True:
			ax = eval_df.plot()
			plt.tight_layout()
			plt.title('EVALUATION set %d, label of spotted events=%d' % (i, spotter_label))
			plt.xlabel('Samples')
			plt.ylabel('Raw data')
			plot_labels(eval_seg, _label_to_color=label_to_color)
			plot_labels(eval_spotted_seg, _label_to_color=label_to_color, _alpha=0.5)

			# Reduce legend to used labels
			lut = {key: col for key, col in label_to_color.iteritems() if key in eval_labels + [spotter_label]}
			add_labels_to_legend(_ax=ax, _label_to_color=lut)

			plt.tight_layout()
			plt.savefig(output_dir + 'evaluation samples %d.png' % i, bbox_inches='tight')

		plt.show()

	# TODO: Implement spotter performance evaluation over all frames (based on eval_spotted_segs and eval_segs)

__author__ = "Martin Freund"
__email__ = "freund@fim.uni-passau.de"
__copyright__ = "Copyright 2015, ACTLab"

if __name__ == "__main__":
	random.seed(0)  # Force determinism to get predictable results

	if False:  # Qt backend; maximized plot
		plt.switch_backend('QT4Agg')
		fig_mgr = plt.get_current_fig_manager()
		fig_mgr.window.showMaximized()

	# Load data sets, build model, train and run HMM
	store = pd.HDFStore('dataset.h5', mode='r')
	try:
		run(store)
	finally:
		store.close()
