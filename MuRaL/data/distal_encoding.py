import numpy as np

def kmer_enc(long_seq, start, stop, chrom, strand, radius, index, order=6, end=False):
    digit_encoder = {'A':0,'C':1,'G':2,'T':3, 'N':-1}
    digit_encoder_rc = {'A':3, 'C':2, 'G':1, 'T':0, 'N':-1}

    short_seq = ['', '','']

    if start < 0:
        left_impute = 0 -start
        start = 0
        short_seq[0] = left_impute * 'N'
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'
    
    short_seq[1] = long_seq[start: stop].upper()
    short_seq = ''.join(short_seq)

    if strand == '+':
        digit_seq = np.asarray([digit_encoder[c] for c in short_seq])
    else:
        digit_seq = np.array([digit_encoder_rc[c] for c in short_seq[::-1]])

    if order > 1:
        seq_len = len(digit_seq)
        new_seq = []
        for i in range(seq_len - order +1):
            kmer = digit_seq[i:i+order]
            if min(kmer) < 0:
                new_seq.append(-1)
            else:
                digit = sum([kmer[d]*4**(order-d-1) for d in range(order)])
                new_seq.append(digit)
        digit_seq = np.array(new_seq)
        window_size = 2 * radius + 1 - order + 1
    else:
        window_size = 2 * radius + 1

    if strand == '+':
        digit_seq = np.asarray([digit_seq[start1:start1 + window_size] for start1 in index], dtype=np.int64)         
    else:
        digit_seq = np.asarray([digit_seq[-start1-window_size:-start1] if start1 else digit_seq[-start1-window_size:] for start1 in index], dtype=np.int64)
    
    digit_seq = np.where(np.logical_and(digit_seq>=0, digit_seq<=4**order), digit_seq, 4**order)
    return digit_seq       
    
def kmer_no_overlap_encode():
    return 