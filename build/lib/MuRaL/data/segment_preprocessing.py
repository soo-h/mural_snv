import numpy as np
import gzip 
from MuRaL.data.preprocessing import bed_reader, kmer_encoding_by_seqs
import pandas as pd
import sys

class GenomeFileReader:
    def __init__(self, genome_file):
        self.genome_file = genome_file
        self.reader = self.open_file()
        self.buffer_segment = []
        self.n_class = 4

    def open_file(self):
        return gzip.open(self.genome_file, mode='rt') if self.genome_file.endswith('.gz') else open(self.genome_file, mode='r')

    
    def read_sites_within_range(self, start: int, end: int) -> np.ndarray:
        used_sites = []

        if self.buffer_segment:
            sites = self.check_buffer_segment(start, end)
            if sites:
                used_sites.extend(sites)
            self.buffer_segment = []
        
        for site in self.reader:
            buffer_line = self.parse_site_line(site)
            self.buffer_segment.append(buffer_line)

            mut_type = self.check_buffer_line(start, end, buffer_line)
            if mut_type is None:
                break
            elif mut_type == 'cont':
                continue
            else:
                used_sites.append(mut_type)

        if not used_sites:
            print(site)
            raise ValueError(f"No sites found in region {start}-{end} in file {self.genome_file}")

        return np.asarray(used_sites)
    
    def read_sites_and_positions_within_range(self, start: int, end: int) -> np.ndarray:
        """Note: can not use read_sites_within_range and read_sites_and_positions_within_range at the same time"""
        used_lines = []
        if self.buffer_segment:
            lines = self.check_buffer_segment(start, end, all=True)
            if lines:
                used_lines += lines
            self.buffer_segment = []
        for line in self.reader:
            buffer_line = self.parse_site_line(line)
            self.buffer_segment.append(buffer_line)
            mut_type = self.check_buffer_line(start, end, buffer_line)
            if mut_type is None:
                break
            elif mut_type == 'cont':
                continue
            else:
                used_lines.append(buffer_line)
        if not used_lines:
            print(line)
            raise ValueError(f"No sites found in region {start}-{end} in file {self.genome_file}")
        return pd.DataFrame(used_lines, columns=['chrom', 'start', 'end', 'name', 'mut_type', 'strand'])

    def check_buffer_segment(self, start: int, end: int, all=False):
        used_sites = []
        if not self.buffer_segment:
            return used_sites

        start_pos, end_pos = self.buffer_segment[0][1], self.buffer_segment[-1][2]

        if end_pos < start:
            return used_sites
        
        for buffer_line in self.buffer_segment:
            mut_type = self.check_buffer_line(start, end, buffer_line)
            if mut_type is None:
                break
            elif mut_type == 'cont':
                continue
            used_sites.append(buffer_line if all else mut_type)

        return used_sites


    def check_buffer_line(self, start: int, end: int, buffer_line):
        chrom, start_pos, end_pos, _, mut_type, _ = buffer_line
        if start_pos < start:
            return 'cont'
        
        elif start_pos >= end:
            self.buffer_line = [chrom, start_pos, end_pos, _, mut_type, _]
            return None
        else:
            return mut_type

    def parse_site_line(self, line):
        line = line.strip().split()
        for idx in [1, 2, 4]:
            line[idx] = int(line[idx])
        return line
    
    def close(self):
        self.reader.close()


def calculate_mutation_frequency(genome_info, use_no_mutation=True):
    mut_info = genome_info['mut_type'].values
    if use_no_mutation:
        return [np.sum(mut_info == i) / len(genome_info) for i in range(4)]
    return [np.sum(mut_info == i) / len(genome_info) for i in range(1,4)]

def get_sites_information(long_seq, sites, radius, chrom, strand):
    """
    Check relationship of regions by radius and save the information for encoding.

    Yields:
    A tuple containing start position of encoding, stop position of encoding, chromosome,
             strand, index list, and a boolean indicating if imputation is needed
    - The index list used to find samples from encoding segment.

    """
    #seqs = []
    end=True
    init = False
    index = [0]
    for start in sites:
        # init 
        if not init:
            init = True
            chrom_init = chrom
            start0 = int(start) - radius
            stop0 = start0 + 1 + 2*radius
            leng_seq = len(long_seq)
            continue
            
        start1 = int(start) - radius
        stop1 = start1 + 1 + 2*radius
        
        if start1 > stop0:
            # one-hot encoding
            impute = False
            yield (start0, stop0, chrom, strand, index, impute)
            start0, stop0 = start1, stop1
            index = [0]
        else:
            stop0 = stop1
            index.append(start1-start0)
    if stop0 > leng_seq:
        impute = True
    else:
        impute = False
    yield (start0, stop0, chrom, strand, index, impute)

def get_genome_local_feature(genome_info, ref_records, local_order=1, local_radius=7):
    """
    处理 genome_info 并返回编码后的序列和突变类型数据集。

    参数:
    genome_info (DataFrame): 包含基因组信息的数据框。
    ref_records (dict): 参考序列记录。
    local_order (int): 局部序列编码的顺序。
    local_radius (int): 局部序列编码的半径。

    返回:
    DataFrame: 编码后的序列和突变类型数据集, 
    
    NOTE: N 位点将会被编码为4.
    """

    chromosomes = genome_info['chrom'].unique()
    if len(chromosomes) != 1:
        raise ValueError("genome_info should contain only one chromosome.")
    chromosome = chromosomes[0]

    strands = genome_info['strand'].unique()
    if len(strands) != 1:
        raise ValueError("genome_info should contain only one strand.")
    strand = strands[0]

    # 获取参考序列
    reference_sequence = str(ref_records[chromosome].seq)
    mutation_types = genome_info['mut_type']
    mutation_sites = genome_info['start'].values
    sample_count = len(mutation_types)

    # 获取局部序列信息并进行编码
    def get_encoded_sequences(sequence, sites, radius, chrom, strand, order):
        site_sequences = get_sites_information(sequence, sites, radius, chrom, strand)
        return kmer_encoding_by_seqs(sequence, site_sequences, sample_count, local_radius=radius, local_order=order)

    # 编码局部序列
    encoded_sequences_order_1 = get_encoded_sequences(reference_sequence, mutation_sites, local_radius, chromosome, strand, order=1)
    final_data = pd.concat([encoded_sequences_order_1, mutation_types], axis=1)
    
    return final_data

def calculate_kmer_mutation_rate(data, kmer_length, use_no_mutation=True):
    """
    计算特定突变类型的 k-mer 突变率。

    参数:
    data (DataFrame): 包含序列和突变类型的数据框。
    mutation_type (int): 突变类型。
    kmer_length (int): k-mer 的长度。

    返回:
    DataFrame: 分组后的 k-mer 突变率。
    """
    if kmer_length == 3:
        extract_kmer_mutation_rate = extract_3mer_mutation_rate
    else:
        raise ValueError('Only 3-mer mutation rate is supported.')

    left_base_type = right_base_type = 4
    start = 0 if use_no_mutation else 1
    prob_num = 4 - start
    d = kmer_length // 2
    mer_list = ['us' + str(i) for i in list(range(1, d + 1))[::-1]] + ['ds' + str(i) for i in list(range(1, d + 1))]
    values = np.zeros((prob_num, left_base_type**d, right_base_type**d))

    for idx, mut_type in enumerate(range(start, 4)):

        subset_data = pd.concat([data[mer_list], data['mut_type'] == mut_type], axis=1)
        extract_kmer_mutation_rate(subset_data.groupby(mer_list).mean(), values[idx])
    return values


def extract_3mer_mutation_rate(kmer_mutation_df, values):
    left_base_type = right_base_type = 4
    if values.shape != (left_base_type, right_base_type):
        raise ValueError('Invalid shape for values array in func <extract_3mer_mutation_rate>.')
    for us in range(left_base_type):
        for ds in range(right_base_type):
            values[us, ds] = kmer_mutation_df.loc[(us, ds)].values if (us, ds) in kmer_mutation_df.index else 0
    return values


def calculate_kmer_mutation_frequency(genome_info, ref_records, k=3, use_no_mutation=True):

    genome_local_feature = get_genome_local_feature(genome_info, ref_records)
    kmer_mutation_rate = calculate_kmer_mutation_rate(genome_local_feature, k, use_no_mutation)
    #kmer_mutation_rate = kmer_mutation_rate.ravel()
    return kmer_mutation_rate

class SegmentIndexFinder:
    def __init__(self, segment_length, chrom_length_dict=None):
        if chrom_length_dict is None:
            chrom_length_dict = {}
            path = '/public/home/songhui/project/Mural/segment_info_utils/chrom_length.config'
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    chrom_length_dict[line[0]] = int(line[2])
            print(chrom_length_dict)

            # chrom_length_dict = {
            # 'chr1' : 249240453,
            # 'chr2' : 243188422,
            # 'chr20' : 62965457
            # }

        self.chrom_length_dict = chrom_length_dict
        self.bins, self.bin_identifiers = self._split_chromosomes_into_bins(self.chrom_length_dict, segment_length)

    def get_segment_idx(self, segment_start_pos: int, segment_end_pos: int, chrom: str) -> int:
        closest_bin_id = self._find_closest_bin(chrom, segment_start_pos, self.bins, self.bin_identifiers)
        return closest_bin_id
    
    def _split_chromosomes_into_bins(self, chrom_lengths, bin_size):
        bins = {}
        bin_identifiers = {}
        bin_id = 0

        for chrom, length in chrom_lengths.items():
            # 按照染色体长度切分成多个bin
            bin_starts = np.arange(0, length + 1, bin_size)
            bins[chrom] = bin_starts
        
            # 为每个bin分配一个数值形式的标识符
            bin_identifiers[chrom] = {}
            for i in range(len(bin_starts)):  
                bin_identifiers[chrom][i] = bin_id
                bin_id += 1

        return bins, bin_identifiers
    
    def _find_closest_bin(self, chrom, position, bins, bin_identifiers):
        """
        定基因组坐标，返回最近bin的标识符。

        参数:
        - chrom: 染色体名称
        - position: 基因组上的位置
        - bins: 每个染色体的bin左端点数组字典
        - bin_identifiers: 每个bin的标识符字典

        返回:
        - closest_bin_id: 最近bin的标识符
        """
        if chrom not in bins:
            raise ValueError(f"Chromosome {chrom} not found in bins.")
    
        # 获取对应染色体的bin左端点数组
        bin_starts = bins[chrom]

        # 查找比position大的第一个bin的索引
        right_index = np.searchsorted(bin_starts, position, side="right")

        # 找到最近的左侧和右侧bin左端点
        left_index = right_index - 1

        # 如果position在最左边，返回第一个bin
        if left_index < 0:
            return bin_identifiers[chrom][0]

        # 如果position超出最右侧bin的范围，返回最后一个bin
        if right_index >= len(bin_starts):
            #return bin_identifiers[chrom][-1]
            return bin_identifiers[chrom][left_index]

        # 比较position到左侧bin和右侧bin的距离
        left_distance = position - bin_starts[left_index]
        right_distance = bin_starts[right_index] - position

        # 返回距离较近的bin标识符
        if left_distance <= right_distance:
            return bin_identifiers[chrom][left_index]
        else:
            return bin_identifiers[chrom][right_index]

class GenomeFileReaderFactory:
    def __init__(self, file_type=None):
        self.readers = {}
        FILE_DIR_PATH = {
            's10M' : '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000.SNV.AT_sites.subtypes.s500k.non_mut.s10M/',
            'half_1in2000_train': '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/training/',
            'half_1in2000_test' : '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/test/'
        }

        if file_type is None:
            self.dirpath='/public/home/songhui/project/Mural/segment_info_utils/split_1in2000/'
        else:
            self.dirpath = FILE_DIR_PATH[file_type]
        print(self.dirpath)

    def get_reader(self, chrom: str, strand: str) -> GenomeFileReader:

        genome_file = self.get_genome_file(chrom, strand)
        if (chrom, strand) not in self.readers:
            self.readers[(chrom, strand)] = GenomeFileReader(genome_file)

        return self.readers[(chrom, strand)]

    def get_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.positive.bed.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.neg.bed.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")


def calc_segment_strategy(method):

    SEGMENT_INFO_COMPUTATION_METHODS = {
        None : mut_freq_and_segment_id,
        'SegMut' : all_freq_only,
        'SegMutRate' : mut_freq_only,
        'AvgSegMutAndKmerMut' : mut_and_kmer_freq,
        'AvgSegKmerMut' : kmer_mut_freq
    }

    return SEGMENT_INFO_COMPUTATION_METHODS[method]

def kmer_mut_freq(genome_info, segment_start_pos, segment_end_pos, chrom, SegIndexFinder, ref_records):
    kmer_freq = calculate_kmer_mutation_frequency(genome_info, ref_records, k=3, use_no_mutation=False)
    segment_info = {
        'segment_avg_kmer_mut': kmer_freq,
    }
    return segment_info

def mut_freq_and_segment_id(genome_info, segment_start_pos, segment_end_pos, chrom, SegIndexFinder, ref_records):
    mut_freq = calculate_mutation_frequency(genome_info)
    segment_idx = SegIndexFinder.get_segment_idx(segment_start_pos, segment_end_pos, chrom)
    segment_info = {
        'segment_avg_mut': mut_freq,
        'segment_id': segment_idx
    }
    return segment_info

def mut_and_kmer_freq(genome_info, segment_start_pos, segment_end_pos, chrom, SegIndexFinder, ref_records):

    mut_freq = calculate_mutation_frequency(genome_info, use_no_mutation=False)
    kmer_freq = calculate_kmer_mutation_frequency(genome_info, ref_records, k=3, use_no_mutation=False)
    segment_info = {
        'segment_avg_mut': mut_freq,
        'segment_avg_kmer_mut': kmer_freq,
    }
    return segment_info

def all_freq_only(genome_info, segment_start_pos, segment_end_pos, chrom, SegIndexFinder, ref_records):
    mut_freq = calculate_mutation_frequency(genome_info)
    segment_info = {
        'segment_avg_mut': mut_freq
    }
    return segment_info

def mut_freq_only(genome_info, segment_start_pos, segment_end_pos, chrom, SegIndexFinder, ref_records):
    mut_freq = calculate_mutation_frequency(genome_info, use_no_mutation=False)
    segment_info = {
        'segment_avg_mut': mut_freq
    }
    return segment_info


class genoInfo_Generator():
    def __init__(self, segment_center, distal_radius, ref_records, method, file_type=None) -> None:
        self.reader_factory = GenomeFileReaderFactory(file_type)
        self.distal_radius = distal_radius
        self.segment_center = segment_center
        #segment_length = distal_radius * 2 + segment_center # may multi segment has same id
        self.SegIndexFinder = SegmentIndexFinder(segment_center)
        # self.calc_info_func = self.get_calc_info_func(calc_info_func_name)
        self.filter_number = 0
        self.calc_segment_strategy = calc_segment_strategy(method)

        self.ref_records = ref_records

    def get_infor(self, segment_label_sites):

        if len(segment_label_sites[0]) < self.filter_number:
            pass

        startSite_pos, endSite_pos, chrom, strand = self.extract_segment_info(segment_label_sites)
        segment_start_pos, segment_end_pos = self.calculate_segment_positions(startSite_pos, endSite_pos) 


        GenomeReader = self.reader_factory.get_reader(chrom, strand)

        #genome_info = GenomeReader.read_sites_within_range(segment_start_pos, segment_end_pos)
        genome_info = GenomeReader.read_sites_and_positions_within_range(segment_start_pos, segment_end_pos)

        return genome_info, segment_start_pos, segment_end_pos, chrom

    def calc_segment_info(self, segment_label_sites):
        genome_info, segment_start_pos, segment_end_pos, chrom = self.get_infor(segment_label_sites)
        segment_info = self.calc_segment_strategy(genome_info, segment_start_pos, segment_end_pos, chrom, self.SegIndexFinder, self.ref_records)
        return segment_info


    
    def extract_segment_info(self, segment):
        sites, strand = segment
        start_segment = sites[0].start
        end_segment = sites[-1].start
        chrom = sites[-1].chrom
        return start_segment, end_segment, chrom, strand
    
    def calculate_segment_positions(self, start, end):
        return start - self.distal_radius, end + self.distal_radius

class genoInfo_Generator2():
    def __init__(self, segment_center, distal_radius, ref_records, method, file_type=None) -> None:
        self.reader_factory = GenomeFileReaderFactory(file_type)
        self.distal_radius = distal_radius
        self.segment_center = segment_center
        #segment_length = distal_radius * 2 + segment_center # may multi segment has same id
        self.SegIndexFinder = SegmentIndexFinder(segment_center)
        # self.calc_info_func = self.get_calc_info_func(calc_info_func_name)
        self.filter_number = 0
        self.calc_segment_strategy = calc_segment_strategy(method)

        self.ref_records = ref_records

    def get_infor(self, segment_label_sites):

        if len(segment_label_sites[0]) < self.filter_number:
            pass

        startSite_pos, endSite_pos, chrom, strand = self.extract_segment_info(segment_label_sites)
        segment_start_pos, segment_end_pos = self.calculate_segment_positions(startSite_pos, endSite_pos) 


        GenomeReader = self.reader_factory.get_reader(chrom, strand)

        #genome_info = GenomeReader.read_sites_within_range(segment_start_pos, segment_end_pos)
        genome_info = GenomeReader.read_sites_and_positions_within_range(segment_start_pos, segment_end_pos)

        return genome_info, segment_start_pos, segment_end_pos, chrom

    def calc_segment_info(self, segment_label_sites):
        genome_info, segment_start_pos, segment_end_pos, chrom = self.get_infor(segment_label_sites)
        segment_info = self.calc_segment_strategy(genome_info, segment_start_pos, segment_end_pos, chrom, self.SegIndexFinder, self.ref_records)
        return segment_info
    
    def calc_segment_info_by_generator(self, segment_generator, segment_length):
        region_segments = []
        previous_segment = None
        previous_start = None
        previous_end = None

        for segment in segment_generator:
            genome_info, segment_start_pos, segment_end_pos, chrom = self.get_infor(segment)
            segment_info = mut_freq_only(genome_info, segment_start_pos, segment_end_pos, chrom, self.SegIndexFinder, self.ref_records)
            if previous_segment is None:
                previous_segment = segment_info
                previous_start = segment_start_pos
                previous_end = segment_end_pos
                #previous_stand = segment[-1]
                continue

            current_segment = segment_info
            current_start = segment_start_pos
            current_end = segment_end_pos
            #current_stand = segment[-1]

            if self.can_merge(previous_start, previous_end, current_start, current_end, segment_length):
                #print(f"Merge two segments: {previous_start} - {previous_end} and {current_start} - {current_end}")
                previous_segments_info, current_segments_info = previous_segment['segment_avg_mut'], current_segment['segment_avg_mut']
                prevous_region_segments_info = np.concatenate([previous_segments_info, current_segments_info])
                current_region_segments_info = np.concatenate([current_segments_info, previous_segments_info])
                region_segments.append(prevous_region_segments_info)
                region_segments.append(current_region_segments_info)
                previous_segment = None

            else:
                #print(f"Split two segments: {previous_start} - {previous_end} and {current_start} - {current_end}")
                region_segments.append(np.concatenate([previous_segment['segment_avg_mut'], [0]* len(previous_segment['segment_avg_mut'])]))
                previous_segment, previous_start, previous_end = current_segment, current_start, current_end
        return region_segments

    def can_merge(self, previous_start, previous_end, current_start, current_end, segment_length): 
        return (current_end - previous_start < segment_length + 2*self.distal_radius) and (previous_end - current_start < segment_length + 2 * self.distal_radius)

    
    
    def extract_segment_info(self, segment):
        sites, strand = segment
        start_segment = sites[0].start
        end_segment = sites[-1].start
        chrom = sites[-1].chrom
        return start_segment, end_segment, chrom, strand
    
    def calculate_segment_positions(self, start, end):
        return start - self.distal_radius, end + self.distal_radius

def prepare_soft_labelv2(bed_regions, segment_center, distal_radius, ref_records=None, method=None, path_type=None):
    method = 'SegMutRate'
    segment_generator = bed_reader(bed_regions, segment_center)
    segment_info_geno = genoInfo_Generator2(segment_center, distal_radius, ref_records, method, path_type)
    region_segments = segment_info_geno.calc_segment_info_by_generator(segment_generator, segment_center)
    segment_infomation = {'segment_avg_mut': region_segments}
    return segment_infomation


def prepare_soft_label(bed_regions, segment_center, distal_radius, ref_records=None, method=None, path_type=None):
    segmentInfoSaver = Segment_info_saver(method)

    segment_generator = bed_reader(bed_regions, segment_center)
    segment_info_geno = genoInfo_Generator(segment_center, distal_radius, ref_records, method, path_type)

    # get soft label
    for segment in segment_generator:
        segment_info  = segment_info_geno.calc_segment_info(segment)
        segmentInfoSaver.save(segment_info)


    # preprocessing soft label
    soft_label_dict = segmentInfoSaver.output()

    return soft_label_dict



# 提取段信息并存储
def store_segment_info(info_dict, chrom, start_segment, end_segment, segment_info):
    if chrom not in info_dict:
        info_dict[chrom] = {'start': [start_segment], 'end': [end_segment], 'info': [segment_info]}
    else:
        info_dict[chrom]['start'].append(start_segment)
        info_dict[chrom]['end'].append(end_segment)
        info_dict[chrom]['info'].append(segment_info)

# 获取指定链段信息
def get_strand_info_dict(strand, positive_dict, negative_dict):
    if strand == '+':
        return positive_dict
    elif strand == '-':
        return negative_dict
    else:
        sys.exit(f"Error: strand should be '+' or '-', but input is {strand}")

# 匹配样本段并保存信息
def match_and_save_info(sample_segment, info_dict):
    start_sample, end_sample, chrom_sample, _ = sample_segment
    
    if chrom_sample in info_dict:
        start_array = np.asarray(info_dict[chrom_sample]['start'])
        end_array = np.asarray(info_dict[chrom_sample]['end'])
        index = (start_sample >= start_array) & (end_sample <= end_array)

        if np.sum(index) >= 1:
            if np.sum(index) != 1:
                print(f"Segment Found {np.sum(index)} regions!")
            index = np.argmax(index)
            segment_info = info_dict[chrom_sample]['info'][index]
        else:
            sys.exit(f"Error: No segment found for sample segment {sample_segment}, found {np.sum(index)}")
    else:
        sys.exit(f"Error: Chromosome {chrom_sample} not found in segment information.")
    return segment_info

# 主函数
def prepare_soft_label2(bed_regions, segment_center, segment_info_length, distal_radius, ref_records=None, method=None, path_type=None):
    segmentInfoSaver = Segment_info_saver(method)

    # 初始化生成器
    segment_generator = bed_reader(bed_regions, segment_info_length)
    samples_segment_generator = bed_reader(bed_regions, segment_center)
    segment_info_geno = genoInfo_Generator(segment_info_length, distal_radius, ref_records, method, path_type)

    # 初始化存储软标签的字典
    segment_info_positive, segment_info_negative = {}, {}

    # 收集每个segment的信息
    for segment in segment_generator:
        start, end, chrom, strand = segment_info_geno.extract_segment_info(segment)
        segment_info = segment_info_geno.calc_segment_info(segment)
        
        # 获取正链或负链信息字典，并存储
        info_dict = get_strand_info_dict(strand, segment_info_positive, segment_info_negative)
        store_segment_info(info_dict, chrom, start, end, segment_info)

    # 对每个样本段进行匹配查找
    for sample_segment in samples_segment_generator:
        start_sample, end_sample, chrom_sample, strand_sample = segment_info_geno.extract_segment_info(sample_segment)
        
        # 获取当前样本段的正链或负链信息字典
        info_dict = get_strand_info_dict(strand_sample, segment_info_positive, segment_info_negative)
        
        # 匹配和保存信息
        segment_info = match_and_save_info((start_sample, end_sample, chrom_sample, strand_sample), info_dict)
        segmentInfoSaver.save(segment_info)

    # 输出结果
    soft_label_dict = segmentInfoSaver.output()
    return soft_label_dict

def prepare_soft_label2v2(bed_regions, segment_center, segment_info_length, distal_radius, ref_records=None, method=None, path_type=None):
    "diff with label2: segment_info computation ways"
    method = 'SegMutRate'
    segmentInfoSaver = Segment_info_saver(method)
    # 初始化生成器
    segment_generator = bed_reader(bed_regions, segment_info_length)
    samples_segment_generator = bed_reader(bed_regions, segment_center)
    segment_info_geno = genoInfo_Generator2(segment_info_length, distal_radius, ref_records, method, path_type)
    region_segments = segment_info_geno.calc_segment_info_by_generator(segment_generator, segment_info_length)

    # 初始化存储软标签的字典
    segment_info_positive, segment_info_negative = {}, {}

    # 收集每个segment的信息
    # 重新初始化
    segment_generator = bed_reader(bed_regions, segment_info_length)
    for idx, segment in enumerate(segment_generator):
        start, end, chrom, strand = segment_info_geno.extract_segment_info(segment)
        segment_info = region_segments[idx]
        
        # 获取正链或负链信息字典，并存储
        info_dict = get_strand_info_dict(strand, segment_info_positive, segment_info_negative)
        store_segment_info(info_dict, chrom, start, end, segment_info)

    # 对每个样本段进行匹配查找
    for sample_segment in samples_segment_generator:
        start_sample, end_sample, chrom_sample, strand_sample = segment_info_geno.extract_segment_info(sample_segment)
        
        # 获取当前样本段的正链或负链信息字典
        info_dict = get_strand_info_dict(strand_sample, segment_info_positive, segment_info_negative)
        
        # 匹配和保存信息, segment_info is list
        segment_info = match_and_save_info((start_sample, end_sample, chrom_sample, strand_sample), info_dict)
        #adapt segment_info to segment_info_saver
        segment_info = {'segment_avg_mut': segment_info}
        segmentInfoSaver.save(segment_info)

    # 输出结果
    soft_label_dict = segmentInfoSaver.output()
    return soft_label_dict

def collect_segment_info(segment_generator, segment_info_processor, strand_info_positive, strand_info_negative):
    """
    收集每个段的信息并存储在正链或负链字典中。
    """
    for segment in segment_generator:
        start, end, chrom, strand = segment_info_processor.extract_segment_info(segment)
        segment_info = segment_info_processor.calc_segment_info(segment)
        info_dict = get_strand_info_dict(strand, strand_info_positive, strand_info_negative)
        store_segment_info(info_dict, chrom, start, end, segment_info)

def match_and_save_sample_info(sample_segment_generator, segment_info_processor, strand_info_positive, strand_info_negative, segment_info_saver):
    """
    匹配样本段信息并保存。
    """
    for sample_segment in sample_segment_generator:
        start_sample, end_sample, chrom_sample, strand_sample = segment_info_processor.extract_segment_info(sample_segment)
        info_dict = get_strand_info_dict(strand_sample, strand_info_positive, strand_info_negative)
        segment_info = match_and_save_info((start_sample, end_sample, chrom_sample, strand_sample), info_dict)
        segment_info_saver.save(segment_info)
# 主函数
def prepare_soft_label3(bed_regions, segment_center, distal_radius, segment_length_config, ref_records=None, path_type=None):
    """
    split segment center and segment info length;
    each feature corresponds to a segment length
    """
    def process_segment_info(segment_length, method, segment_info_positive, segment_info_negative):
        segment_info_saver = Segment_info_saver(method)
        segment_generator = bed_reader(bed_regions, segment_length)
        sample_segment_generator = bed_reader(bed_regions, segment_center)
        info_geno = genoInfo_Generator(segment_length, distal_radius=distal_radius, ref_records=ref_records, method=method, file_type=path_type)
        collect_segment_info(segment_generator, info_geno, segment_info_positive, segment_info_negative)
        match_and_save_sample_info(sample_segment_generator, info_geno, segment_info_positive, segment_info_negative, segment_info_saver)
        return segment_info_saver.output()

    soft_label_dict = {}
    if 'kmer_mut' in segment_length_config:
        segment_length = segment_length_config['kmer_mut']
        soft_label_dict.update(process_segment_info(segment_length, 'AvgSegKmerMut', {}, {}))

    if 'avg_mut' in segment_length_config:
        segment_length = segment_length_config['avg_mut']
        soft_label_dict.update(process_segment_info(segment_length, 'SegMutRate', {}, {}))

    return soft_label_dict

class Segment_info_saver():
    def __init__(self, method):
        self.assert_method(method)
        self.method = method
        self._init(method)
    
    def _init(self, method):
        self.soft_label_dict = get_soft_label_dict(method)

    def save(self, segment_info):
        if self.method in ['AvgSegKmerMut', 'AvgSegMutAndKmerMut']:
            self.soft_label_dict['segment_avg_kmer_mut'].append(segment_info['segment_avg_kmer_mut'])
        
        if self.method != 'AvgSegKmerMut':
            self.soft_label_dict['segment_avg_mut'].append(segment_info['segment_avg_mut'])
        
        if self.method is None:
            self.soft_label_dict['segment_id'].append(segment_info['segment_id'])
    
    def output(self):
        return self.soft_label_dict
    
    def assert_method(self, method):
        if method not in ['SegMut', 'SegMutRate', 'AvgSegMutAndKmerMut', 'AvgSegKmerMut', None]:
            raise ValueError(f"Error: method should be 'SegMut' or None or 'SegMutRate', but input is {method}")

def get_soft_label_dict(method):
    if method == 'AvgSegKmerMut':
        soft_label_dict = {
            'segment_avg_kmer_mut' : []
        }
    else:
        soft_label_dict = {
            'segment_avg_mut': [],
        }

        if method is None:
            soft_label_dict['segment_id'] = []

        elif method == 'AvgSegMutAndKmerMut':
            soft_label_dict['segment_avg_kmer_mut'] = []

    return soft_label_dict
