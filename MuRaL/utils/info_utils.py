import re
import sys

import pandas as pd
import numpy as np

"""
Utility functions for parsing the info field in BED/prediction files.

Info field format: 'chr1:238329;G>A;-1;3'
Extended format:   'chr1:238329;G>A;-1;3;AFR:2;EUR:1'
"""


def parse_info(info_str):
    """Parse info field from BED name column or prediction file.

    Args:
        info_str: the info string, e.g. 'chr1:238329;G>A;-1;3'
        key: None  -> return the count (4th semicolon-delimited field)
             str   -> return value for that key (e.g. key='AFR' -> 2)

    Returns:
        float: the parsed count or key value; 1.0 as default fallback
    """
    if info_str == '.' or info_str == '' or info_str == 'na' or info_str=='g':
        return 1.0
    parts = info_str.split(';')
    return float(parts[3])


def parse_pred_header(header_line):
    """Parse prediction file header and build column index.

    Handles both old format (no info column) and new format (with info column).

    Args:
        header_line: raw header line string (with or without trailing newline)

    Returns:
        dict with keys:
            'col_idx': dict mapping column name -> position index
            'has_info': bool, whether info column is present
            'prob_start': int, index of first prob column
            'n_prob': int, number of prob columns
    """
    header = header_line.strip().split('\t')
    col_idx = {name: i for i, name in enumerate(header)}

    has_info = 'info' in col_idx

    # prob columns start after mut_type (old format) or after strand (new format)
    # reliably find them by looking for prob0
    if 'prob0' not in col_idx:
        raise ValueError(f"Header missing 'prob0' column: {header}")

    prob_start = col_idx['prob0']
    prob_cols = [h for h in header if h.startswith('prob')]
    n_prob = len(prob_cols)

    # Detect mu columns (NB model output)
    mu_cols = [h for h in header if re.match(r'^mu\d+$', h)]
    n_mu = len(mu_cols)
    has_mu = n_mu > 0
    mu_start = col_idx.get('mu0') if has_mu else None

    return {
        'col_idx': col_idx,
        'has_info': has_info,
        'prob_start': prob_start,
        'n_prob': n_prob,
        'header': header,
        'has_mu': has_mu,
        'mu_start': mu_start,
        'n_mu': n_mu,
    }


def read_pred_line(fields, header_info, recurrent=False, use_mu=False):
    """Parse a single line from a prediction file.

    Args:
        fields: list of string fields from line.split('\\t')
        header_info: dict returned by parse_pred_header()
        recurrent: whether to extract count from info column
        use_mu: if True and mu columns exist, extract mu values instead of prob

    Returns:
        dict with keys: chrom, start, end, strand, mut_type, probs, count
    """
    col_idx = header_info['col_idx']
    if use_mu and header_info.get('has_mu', False):
        prob_start = header_info['mu_start']
        n_prob = header_info['n_mu']
    else:
        prob_start = header_info['prob_start']
        n_prob = header_info['n_prob']

    result = {
        'chrom': fields[col_idx['chrom']],
        'start': int(fields[col_idx['start']]),
        'end': int(fields[col_idx['end']]),
        'strand': fields[col_idx['strand']],
        'mut_type': int(fields[col_idx['mut_type']]),
        'probs': np.asarray(fields[prob_start:prob_start + n_prob], dtype='float'),
    }

    if recurrent and header_info['has_info']:
        result['count'] = parse_info(fields[col_idx['info']])
    else:
        result['count'] = 1

    return result

