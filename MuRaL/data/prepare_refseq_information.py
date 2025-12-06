import numpy as np
from MuRaL.data.preprocessing import bed_reader

# def prepare_step_avgmut():


#     single_base_info = {
#         'segment_avg_mut' : [],
#     }

def prepare_single_base_info(bed_regions, segment_length, seq_record, single_base_task_config):
    radius_length = single_base_task_config['radius_length']
    bin_size = single_base_task_config['bin_size']
    cumulated = single_base_task_config.get('cumulated')

    single_base_info = {
        'nuc_skew': [],
    }
    bed_generator = bed_reader(bed_regions, segment_length)
    chrom = None
    for batch, stand in bed_generator:
        if chrom != batch[0].chrom:
            chrom = batch[0].chrom
            long_seq = str(seq_record[chrom].seq)
            length = len(long_seq)

        nuc_skew = get_single_base_info_in_segment(batch, radius_length, length, long_seq, bin_size, cumulated)
        single_base_info['nuc_skew'].append(nuc_skew)
    return single_base_info

def get_single_base_info_in_segment(batch, radius_length, length, long_seq, bin_size, cumulated=False):
    S_value_list = []
    for region in batch:
        up_seq, down_seq = get_up_downstream_sequences(region, radius_length, length, long_seq)
        up_S_value = calc_profile_S(up_seq,  stand=region.strand, bin_size=bin_size)
        down_S_value = calc_profile_S(down_seq, stand=region.strand, bin_size=bin_size)
        S_value = np.concatenate([up_S_value, down_S_value])
        if cumulated:
            S_value = np.cumsum(S_value)
        S_value_list.append(S_value)
    return np.asarray(S_value_list)

def get_up_downstream_bound(start, stop, radius, length):
    up_boundary = max(0 , int(start) - radius)
    down_boundary = min(int(stop) + radius, length)
    return up_boundary, start, down_boundary

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))

def get_up_downstream_sequences(locus, radius, length, long_seq):
    chrom, start, stop, strand = str(locus.chrom), locus.start, locus.stop, locus.strand

    up, mid, end = get_up_downstream_bound(start, stop, radius, length)

    up_seq = long_seq[up:mid].upper()
    down_seq = long_seq[mid+1:end].upper()
    if strand == "-":
        up_seq = reverse_complement(up_seq)
        down_seq = reverse_complement(down_seq)
    return up_seq, down_seq

def calc_profile_S(seq, stand, bin_size=1000):
    coeff = 1 if stand == "+" else -1
    bin_number = int(np.ceil(len(seq) / bin_size))
    S_values = np.empty(bin_number)

    for idx in range(bin_number):
        bin_seq = seq[idx*bin_size:(idx+1)*bin_size]
        S_values[idx] = coeff * calc_S(bin_seq)
    return S_values

def calc_S(seq):
    ATGC_count = {base: seq.count(base) for base in 'ATGC'}
    TA_count = ATGC_count['A'] + ATGC_count['T']
    GC_count = ATGC_count['G'] + ATGC_count['C']
    
    S_TA = (ATGC_count['T'] - ATGC_count['A']) / TA_count if TA_count else 0
    S_CG = (ATGC_count['G'] - ATGC_count['C']) / GC_count if GC_count else 0
    
    S_value = S_TA + S_CG
    return S_value
