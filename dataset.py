#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def class_name(_id):
	""" Translate a label / class ID to a name (string).
	:param _id: Label ID
	:return: Label name
	"""

	return {
		17: 'Training',  # This is not a gesture but marks the training region
		18: 'Glass Sip',  # Type 1
		19: 'Cup Sip',  # Type 2
		20: 'Mug Sip',  # Type 3
		21: 'Bottle Sip',  # Type 4
	}[_id]

# Feature list of the right hand sensor
# RLAaccx, RLAaccy, RLAaccz  # 3-axis Accelerometer
# RLAaccn                    # Accelerometer-derived euclidean norm
# RLAgyrx, RLAgyry, RLAgyrz  # 3-axis Gyroscope
# RLAroll, RLApitch, RLAyaw  # Gyro-derived euler angles; clipping problem at +- 180°, do not use
# RLAmagx, RLAmagy, RLAmagz  # Magnetometer

# RLAr11, RLAr12, RLAr13  # Not Relevant
# RLAr21, RLAr22, RLAr23  # ""
# RLAr31, RLAr32, RLAr33  # ""

# Feature list of the left hand sensor; see above
# LHSaccx, LHSaccy, LHSaccz
# LHSaccn
# LHSgyrx, LHSgyry, LHSgyrz
# LHSroll, LHSpitch, LHSyaw  # Clipping problem at +- 180°, do not use
# LHSmagx, LHSmagy, LHSmagz

# LHSr11, LHSr12, LHSr13
# LHSr21, LHSr22, LHSr23
# LHSr31, LHSr32, LHSr33


def _extract_participant(_key):
	""" Extract the participant ID from a HDFStore key, e.g. '/participant_11/segments' -> 11.
	:param _key: The HDFStore key.
	:return: The participant ID as integer.
	"""

	main_table = _key.split('/', 2)[1]  # Extract '/<this sequence>/abc/def'
	return int(main_table.rsplit('_', 1)[1])  # Extract '.*_<this sequence>'


def gen_seg_table(_store, _participants=None, _labels=None, _type=None):
	""" Generator for data set segment tables.
	:param _store: The HDFStore containing the data set.
	:param _participants: List of, a single participant, or all if None.
	:param _labels: List of, a single label, or all if None.
	:param _type: The segment type; one out of 'training', 'free-living', 'both'.
	:return: A generator.
	"""

	if _participants is None:
		_participants = frozenset()
	elif hasattr(_participants, '__iter__'):
		_participants = frozenset(_participants)
	elif not isinstance(_participants, set):
		_participants = frozenset([_participants])

	if _labels and not hasattr(_labels, '__iter__'):
		_labels = frozenset([_labels])

	if _type is None:
		_type = 'both'
	if _type not in ['training', 'free-living', 'both']:
		raise KeyError("Invalid '_type'! Only 'training', 'free-living' and 'both' / None are supported.")

	seg_table_names = (k for k in _store.keys() if k.endswith('/segments'))  # Get all segment keys

	if _participants:
		segs_by_participants = [k for k in seg_table_names for p in _participants if _extract_participant(k) == p]
	else:
		segs_by_participants = seg_table_names

	for k in segs_by_participants:
		t = _store.select(k).sort('Begin')
		if _type == 'training':
			t = t[t['End'] < t[t['ID'] == 17]['End'][0]]
		elif _type == 'free-living':
			t = t[t['Begin'] > t[t['ID'] == 17]['End'][0]]

		j = k.rsplit('/', 1)[0]  # Link segment to participant name
		if _labels is None:
			yield j, t
		else:
			yield j, t[np.any(map(lambda s: t['ID'] == s, _labels), axis=0)]


def gen_segments_from_table(_store, _seg_table, _extend_front=0, _extend_back=0, **_kwargs):
	""" Generator for data set sample segments based on a given segments table.
	:param _store: The HDFStore containing the data set.
	:param _seg_table: A segments table; can be obtained from gen_seg_table().
	:param _extend_front: Additional samples fetched before the beginning of the segment.
	:param _extend_back: Additional samples fetched after the end of the segment.
	:param _kwargs: Parameters passed down to HDFStore.select() on '/samples'
	:return: A generator.
	"""

	for k, seg in _seg_table:
		for start, end, length, label in seg.values:
			yield _store.select(k + '/samples', start=start - _extend_front, stop=start+length + _extend_back, **_kwargs)


def gen_segments(_store, _participants=None, _labels=None, _extend_front=0, _extend_back=0, **_kwargs):
	""" Generator for data set sample segments.
	:param _store: The HDFStore containing the data set.
	:param _participants: List of, a single participant, or all if None.
	:param _labels: List of, a single label, or all if None.
	:param _extend_front: Additional samples fetched before the beginning of the segment.
	:param _extend_back: Additional samples fetched after the end of the segment.
	:param _kwargs: Parameters passed down to HDFStore.select() on '/samples'
	:return: A generator.
	"""

	for k, seg in gen_seg_table(_store, _participants, _labels):
		for start, end, length, label in seg.values:
			yield _store.select(k + '/samples', start=start - _extend_front, stop=start+length + _extend_back, **_kwargs)


def get_frame(_store, _participant, _start=None, _stop=None, **_kwargs):
	""" Query participant data set.
	:param _store: The HDFStore containing the data set.
	:param _participant: Participant ID.
	:param _start: Start index for query or None for first possible.
	:param _stop: Stop index for query or None for last possible.
	:param _kwargs: Parameters passed down to HDFStore.select() on '/samples'
	:return: A DataFrame.
	"""

	table_name = [k for k in _store.keys() if k.endswith('/samples') and _extract_participant(k) == _participant][0]
	return _store.select(table_name, start=_start, stop=_stop, **_kwargs)


def gen_windows(_store, _wnd_size=10, _step=1, _participants=None, _labels=None, **_kwargs):
	""" Generator for windowing segment table entries.
	:param _store: The HDFStore containing the data set.
	:param _wnd_size: Size of the segment window; 10 on default.
	:param _step: Step size of the segment windowing; 1 on default.
	:param _participants: List of, a single participant, or all if None.
	:param _labels: List of, a single label, or all if None.
	:param _kwargs: Parameters passed down to HDFStore.select() on '/segments'
	:return: A generator.
	"""

	for df in gen_segments(_store, _participants, _labels, **_kwargs):
		for x in xrange(0, df.shape[0] - _wnd_size + 1, _step):
			yield df.iloc[x:x+_wnd_size]


def test():
	""" Implementation test function.
	:return:None
	"""

	store = pd.HDFStore('dataset.h5', mode='r', format='table')
	try:
		features = ['RLAaccx', 'RLAaccy', 'RLAaccz']
		for x in gen_windows(store, _wnd_size=50, _step=10, _participants=11, _labels=[18], columns=features):
			print x

	finally:
		store.close()

__author__ = "Martin Freund"
__email__ = "freund@fim.uni-passau.de"
__copyright__ = "Copyright 2015, ACTLab"

if __name__ == "__main__":
	test()
