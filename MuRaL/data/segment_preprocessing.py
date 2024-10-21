import numpy as np
import gzip 


class GenomeFileReader:
    def __init__(self, genome_file):
        self.genome_file = genome_file
        self.reader = self.open_file()
        self.buffer_line = None
        self.n_class = 4

    def open_file(self):
        return gzip.open(self.genome_file, mode='rt') if self.genome_file.endswith('.gz') else open(self.genome_file, mode='r')

    
    def read_sites_within_range(self, start: int, end: int) -> np.ndarray:
        used_sites = []

        if self.buffer_line:
            mut_type = self.check_buffer_line(start, end)
            if mut_type in list(range(self.n_class)):
                used_sites.append(mut_type)
        
        for site in self.reader:
            self.buffer_line = self.parse_site_line(site)
            mut_type = self.check_buffer_line(start, end)
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

    def check_buffer_line(self, start: int, end: int):
        chrom, start_pos, end_pos, _, mut_type, _ = self.buffer_line
        self.buffer_line = None
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

def calculate_mutation_frequency(genome_info):
    return [np.sum(genome_info == i) / len(genome_info) for i in range(4)]

class SegmentIndexFinder:
    def __init__(self, segment_length, chrom_length_dict=None):
        if chrom_length_dict is None:
            chrom_length_dict = {
            'chr1' : 249240453,
            'chr2' : 243188422
            }

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
    def __init__(self):
        self.readers = {}
        self.dirpath='/public/home/songhui/project/Mural/segment_info_utils/split_1in2000/'

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


class genoInfo_Generator():

    def __init__(self, segment_center, distal_radius) -> None:
        self.reader_factory = GenomeFileReaderFactory()
        self.distal_radius = distal_radius
        self.segment_center = segment_center
        #segment_length = distal_radius * 2 + segment_center # may multi segment has same id
        self.SegIndexFinder = SegmentIndexFinder(segment_center)
        # self.calc_info_func = self.get_calc_info_func(calc_info_func_name)
        self.filter_number = 0

    def get_infor(self, segment_label_sites, method=None):

        if len(segment_label_sites[0]) < self.filter_number:
            pass

        startSite_pos, endSite_pos, chrom, strand = self.extract_segment_info(segment_label_sites)
        segment_start_pos, segment_end_pos = self.calculate_segment_positions(startSite_pos, endSite_pos) 


        GenomeReader = self.reader_factory.get_reader(chrom, strand)

        genome_info = GenomeReader.read_sites_within_range(segment_start_pos, segment_end_pos)
        mut_freq = calculate_mutation_frequency(genome_info)
        segment_idx = self.SegIndexFinder.get_segment_idx(segment_start_pos, segment_end_pos, chrom)

        return mut_freq, segment_idx
    
    
    def extract_segment_info(self, segment):
        sites, strand = segment
        start_segment = sites[0].start
        end_segment = sites[-1].start
        chrom = sites[-1].chrom
        return start_segment, end_segment, chrom, strand
    
    def calculate_segment_positions(self, start, end):
        return start - self.distal_radius, end + self.distal_radius
    

