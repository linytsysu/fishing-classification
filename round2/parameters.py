fc_parameters_v1 = {'lat': {'quantile': [{'q': 0.9},
   {'q': 0.8},
   {'q': 0.6},
   {'q': 0.3},
   {'q': 0.2},
   {'q': 0.1}],
  'maximum': None,
  'minimum': None,
  'number_cwt_peaks': [{'n': 1}],
  'fft_coefficient': [{'coeff': 5, 'attr': 'real'},
   {'coeff': 5, 'attr': 'angle'},
   {'coeff': 74, 'attr': 'angle'}],
  'approximate_entropy': [{'m': 2, 'r': 0.1}],
  'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 2}],
  'median': None},
 'lon': {'quantile': [{'q': 0.1},
   {'q': 0.2},
   {'q': 0.4},
   {'q': 0.9},
   {'q': 0.3},
   {'q': 0.8},
   {'q': 0.6}],
  'minimum': None,
  'maximum': None,
  'approximate_entropy': [{'m': 2, 'r': 0.1}],
  'fft_coefficient': [{'coeff': 64, 'attr': 'angle'},
   {'coeff': 99, 'attr': 'angle'},
   {'coeff': 58, 'attr': 'angle'},
   {'coeff': 2, 'attr': 'angle'}],
  'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.8}],
  'percentage_of_reoccurring_datapoints_to_all_datapoints': None},
 '速度': {'number_crossing_m': [{'m': 1}],
  'ar_coefficient': [{'k': 10, 'coeff': 1}],
  'ratio_beyond_r_sigma': [{'r': 0.5}],
  'quantile': [{'q': 0.7}, {'q': 0.8}],
  'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}],
  'partial_autocorrelation': [{'lag': 6}, {'lag': 4}],
  'fft_coefficient': [{'coeff': 9, 'attr': 'angle'},
   {'coeff': 5, 'attr': 'angle'}],
  'change_quantiles': [{'f_agg': 'mean', 'isabs': False, 'qh': 0.8, 'ql': 0.0},
   {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.4}],
  'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 1}]},
 '方向': {'fft_coefficient': [{'coeff': 5, 'attr': 'real'},
   {'coeff': 91, 'attr': 'abs'}],
  'quantile': [{'q': 0.9}],
  'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2}],
  'standard_deviation': None,
  'ratio_beyond_r_sigma': [{'r': 0.5}]}}

fc_parameters_v2 = {'lon': {'quantile': [{'q': 0.1},
   {'q': 0.2},
   {'q': 0.3},
   {'q': 0.9},
   {'q': 0.7},
   {'q': 0.8}],
  'minimum': None,
  'maximum': None,
  'agg_linear_trend': [{'f_agg': 'min', 'chunk_len': 50, 'attr': 'intercept'}],
  'median': None,
  'abs_energy': None,
  'fft_coefficient': [{'coeff': 58, 'attr': 'angle'}],
  'approximate_entropy': [{'m': 2, 'r': 0.1}],
  'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 20}],
  'mean': None,
  'sum_of_reoccurring_data_points': None,
  'sample_entropy': None},
 'lat': {'maximum': None,
  'quantile': [{'q': 0.9},
   {'q': 0.1},
   {'q': 0.7},
   {'q': 0.6},
   {'q': 0.2},
   {'q': 0.8},
   {'q': 0.4}],
  'minimum': None,
  'number_cwt_peaks': [{'n': 1}],
  'sum_values': None,
  'abs_energy': None,
  'autocorrelation': [{'lag': 5}],
  'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2}],
  'spkt_welch_density': [{'coeff': 8}],
  'agg_linear_trend': [{'f_agg': 'max',
    'chunk_len': 50,
    'attr': 'intercept'}]},
 '速度': {'number_crossing_m': [{'m': 1}],
  'ratio_beyond_r_sigma': [{'r': 0.5}],
  'approximate_entropy': [{'m': 2, 'r': 0.9}],
  'partial_autocorrelation': [{'lag': 7}, {'lag': 6}, {'lag': 5}],
  'agg_autocorrelation': [{'f_agg': 'median', 'maxlag': 40}],
  'ar_coefficient': [{'k': 10, 'coeff': 1}],
  'fft_coefficient': [{'coeff': 5, 'attr': 'angle'}],
  'quantile': [{'q': 0.7}],
  'change_quantiles': [{'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.2}],
  'fft_aggregated': [{'aggtype': 'variance'}]},
 '方向': {'quantile': [{'q': 0.9}],
  'approximate_entropy': [{'m': 2, 'r': 0.1}],
  'ratio_beyond_r_sigma': [{'r': 0.5}],
  'fft_coefficient': [{'coeff': 5, 'attr': 'angle'}],
  'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}]}}