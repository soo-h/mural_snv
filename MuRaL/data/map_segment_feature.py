from MuRaL.data.preprocessing import bed_reader
import numpy as np
import gzip
import os


class RegionalFeatureReader():
    def __init__(self, genome_file):
        """
        Initialize the reader for the genome file.
        
        Parameters:
            genome_file (str): Path to the genome file (could be gzipped).
        """
        self.genome_file = genome_file
        self.reader = self._open_file(genome_file)
        self.buffer_line = None
        self.buffer_line2 = None

    def _open_file(self, file_path):
        """
        Open the genome file, either gzipped or plain text.
        
        Parameters:
            file_path (str): Path to the genome file.
        
        Returns:
            file object: Opened file object.
        """
        if file_path.endswith('.gz'):
            return gzip.open(file_path, mode='rt')
        else:
            return open(file_path, mode='r')

    def _extract_region_info(self, line):
        """
        Extracts the region information from a line in the file.
        
        Parameters:
            line (str): A line from the file to extract information from.
        
        Returns:
            tuple: A tuple containing (chrom, regional_start, region_mid, region_end, strand, values)
        """
        fields = line.strip().split('\t')
        chrom, regional_start, region_mid, region_end, strand = fields[:5]
        values = fields[5:]  # Remaining values after the region information
        return chrom, float(region_mid), values
    
    def find_region_feature(self, site):
        """
        Finds the feature values for a given site based on the region.
        
        Parameters:
            site (float): The site position to find the corresponding region feature.
        
        Returns:
            list: The feature values for the closest region.
        """
        closest_values = None
        min_delta = float('inf')  # Initialize to a large number
        
        # First check if the buffer line exists
        if self.buffer_line:
            chrom, region_mid, values = self._extract_region_info(self.buffer_line)
            min_delta = abs(region_mid - site)
            closest_values = values
        

            chrom, region_mid, values = self._extract_region_info(self.buffer_line2)
            delta = abs(region_mid - site)
            if delta < min_delta:
                closest_values = values
                min_delta = delta
                self.buffer_line = self.buffer_line2
            else:
                return np.asarray(closest_values, dtype=float)
                #return [float(x) for x in closest_values]

        # Process the file line by line
        for line in self.reader:
            if line.startswith('chrom'):  # Skip header line
                continue
            
            chrom, region_mid, values = self._extract_region_info(line)
            delta = abs(region_mid - site)

            # If this region is closer to the site, update the closest match
            if delta < min_delta:
                closest_values = values
                min_delta = delta
                self.buffer_line = line  # Buffer the current line for next check

            # If the region's midpoint is farther from the site, return the previous closest values
            else:
                self.buffer_line2 = line
                return np.asarray(closest_values, dtype=float)

        # Return the final closest region values
        return np.asarray(closest_values, dtype=float)
        #return [float(x) for x in closest_values]

class RegionalAvgMut2merReader(RegionalFeatureReader):
    def _extract_region_info(self, line):
        """
        Extracts the region information from a line in the file.
        
        Parameters:
            line (str): A line from the file to extract information from.
        
        Returns:
            tuple: A tuple containing (chrom, regional_start, region_mid, region_end, strand, values)
        """
        fields = line.strip().split('\t')
        chrom, regional_start, region_mid, region_end, strand = fields[:5]
        avg_mut_2mer = fields[5:]  # Remaining values after the region information
        avg_mut_2mer = self.choice_2mer(avg_mut_2mer)
        return chrom, float(region_mid), avg_mut_2mer   

    def choice_2mer(self, avg_mut_2mer):
        avg_mut_2mer = np.asarray(avg_mut_2mer, dtype=float).reshape(14,3,4) # us/ds, prob1-3, 4nucleotides
        return avg_mut_2mer

class EachSiteFeatureReader():
    def __init__(self, genome_file):

        self.genome_file = genome_file
        self.reader = self._open_file(genome_file)

    def _open_file(self, file_path):

        if file_path.endswith('.gz'):
            return gzip.open(file_path, mode='rt')
        else:
            return open(file_path, mode='r')

    def _extract_region_info(self, line):

        fields = line.strip().split('\t')
        site_start = int(fields[0])
        values = fields[1:]  # Remaining values after the region information
        return site_start, values
    
    def find_region_feature(self, site):
        """
        Finds the feature values for a given site based on the region.
        
        Parameters:
            site (float): The site position to find the corresponding region feature.
        
        Returns:
            list: The feature values for the closest region.
        """

        # Process the file line by line
        for line in self.reader:
            if line.startswith('site'):  # Skip header line
                continue
            
            site_start, values = self._extract_region_info(line)
            if site_start == site:
                return np.asarray(values, dtype=float)
            if site_start > site:
                raise ValueError(f"Error: site_start should be equal to site, but input is {site_start} and {site}")
            
class EachSiteFeatureReaderFactory:
    """
    A factory class for creating regional feature readers.
    It manages file paths and strategies for different chromosomal data.
    """

    FEATURE_READERS = {
        'step_avg_mut' : EachSiteFeatureReader,
    }



    def __init__(self, file_type=None, feature_type=None, slid_strategy=None):
        """
        Initialize the factory with a specified file type.
        Sets up the directory path and sliding window strategy accordingly.
        """
        self.dirpath = slid_strategy if slid_strategy.endswith('/') else slid_strategy + '/'
        self.file_type = file_type
        self.reader_class = self.FEATURE_READERS.get(feature_type, self.FEATURE_READERS['step_avg_mut'])

        print(f"Using directory path: {self.dirpath}")

        # 存储读取器对象以避免重复创建
        self.readers = {}

    def get_reader(self, chrom: str, strand: str):
        get_genome_file = self._reader_choice(self.file_type)
        regional_feature_map = get_genome_file(chrom, strand)
        if (chrom, strand) not in self.readers:
            self.readers[(chrom, strand)] = self.reader_class(regional_feature_map)

        return self.readers[(chrom, strand)]

    def _reader_choice(self, file_type):
        if file_type == 'half_1in2000_train' or file_type == 'half_1in2000_test' or 'half_' in file_type:
            return self.get_genome_file
        elif file_type == 'half2_1in2000_train' or file_type == 'half2_1in2000_test' or file_type == 'half2_1in20000_train':
            return self.get_nonCpG_CGsites_genome_file
        elif file_type == 'half3_5in1000_train' or file_type == 'half3_5in1000_test':
            return self.get_CpGsites_genome_file
    
    def get_CpGsites_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.CpG_sites.filtered.all.SNP.subtypes.positive.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.CpG_sites.filtered.all.SNP.subtypes.neg.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def get_nonCpG_CGsites_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.nonCpG_CG_sites.filtered.all.SNP.subtypes.positive.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.nonCpG_CG_sites.filtered.all.SNP.subtypes.neg.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def get_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.positive.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.neg.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def close(self):
        self.reader.close()

class RegionalFeatureReaderFactory:
    """
    A factory class for creating regional feature readers.
    It manages file paths and strategies for different chromosomal data.
    """

    # 常量配置类，避免硬编码
    DEFAULT_FILE_DIR_PATH = '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/training/windows_feature/'
    DEFAULT_SLIDE_WINDOW_STRATEGY = 'window50k_step1k_prob1to3'
    DEFAULT_FEATURE_DIR_PATH = 'average_mutation/'

    # 预定义的文件路径和滑动窗口策略
    FILE_DIR_PATH = {
        'half_1in2000_train': '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/training/windows_feature/',
        'half2_1in2000_train': '/public/home/songhui/project/Mural/segment_info_utils/split2_1in2000_train_test/training/windows_feature/',
        'half3_5in1000_train': '/public/home/songhui/project/Mural/segment_info_utils/split3_5in1000_train_test/training/windows_feature/',
        'half_1in20000_train': '/public/home/songhui/project/Mural/segment_info_utils/down_1in20000/split_1in20000/split_1in20000_train_test/training/windows_feature/',
        'half2_1in20000_train': '/public/home/songhui/project/Mural/segment_info_utils/down_1in20000/split_1in20000/split2_1in20000_train_test/training/windows_feature/',


        #'half_1in2000_test': '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/training/windows_feature/',
        #'half2_1in2000_test': '/public/home/songhui/project/Mural/segment_info_utils/split2_1in2000_train_test/training/windows_feature/',
        #'half3_5in1000_test': '/public/home/songhui/project/Mural/segment_info_utils/split3_5in1000_train_test/training/windows_feature/',
        #'half_1in20000_test': '/public/home/songhui/project/Mural/segment_info_utils/down_1in20000/split_1in20000/split_1in20000_train_test/training/windows_feature/',

        'half_1in2000_test': '/public/home/songhui/project/Mural/segment_info_utils/split_1in2000_train_test/test/windows_feature/',
        'half2_1in2000_test': '/public/home/songhui/project/Mural/segment_info_utils/split2_1in2000_train_test/test/windows_feature/',
        'half3_5in1000_test': '/public/home/songhui/project/Mural/segment_info_utils/split3_5in1000_train_test/test/windows_feature/',
        'half_1in20000_test': '/public/home/songhui/project/Mural/segment_info_utils/down_1in20000/split_1in20000/split_1in20000_train_test/test/windows_feature/',
        None: DEFAULT_FILE_DIR_PATH,
    }

    FEATURE_DIR_PATH = {
        'avg_mut': 'average_mutation/',
        '2mer_mut': 'adjoin7bp_2mer_average_mutation/',
        'step_avg_mut' : 'step_average_mutation/'
    }

    SLIDE_WINDOWS_STRATEGY = {
        'window50k_step1k_prob1to3': 'window50k_step1k_prob1to3',
        'window50k_step10k_prob1to3': 'window50k_step10k_prob1to3',
        'window50k_step50k_prob1to3': 'window50k_step50k_prob1to3',
        'radius10_bound100k_prob1to3' : 'radius10_bound100k_prob1to3',
        'radius10_bound100k_SNPbase_prob1to3' : 'radius10_bound100k_SNPbase_prob1to3',
        'radius10_bound100k_prob1to3v2' : 'radius10_bound100k_prob1to3v2',
        'radius10_bound100k_proball' : 'radius10_bound100k_proball',
        'radius10_bound100k_proball_reverse' : 'radius10_bound100k_proball_reverse',
        'radius10_bound100k_proball_histone' : 'radius10_bound100k_proball_histone',
        None: DEFAULT_SLIDE_WINDOW_STRATEGY,
    }

    FEATURE_READERS = {
        'avg_mut' : RegionalFeatureReader,
        '2mer_mut' : RegionalAvgMut2merReader,
        'step_avg_mut' : EachSiteFeatureReader,
    }

    def __init__(self, file_type=None, feature_type=None, slid_strategy=None):
        """
        Initialize the factory with a specified file type.
        Sets up the directory path and sliding window strategy accordingly.
        """
        self.file_type = file_type
        if slid_strategy is not None and os.path.isdir(slid_strategy):
            self.dirpath = slid_strategy if slid_strategy.endswith('/') else slid_strategy + '/'
            self.slid_window_strategy = self.dirpath.split('/')[-2]

        # to-del : save to backward compatibility, will remove later
        else:
            home_dirpath = self._get_file_dir_path(file_type)
            dirname = self._get_feature_dir_path(feature_type)
            self.slid_window_strategy = slid_strategy if slid_strategy else self.DEFAULT_SLIDE_WINDOW_STRATEGY
            #self.dirpath = f"{home_dirpath}{self.slid_window_strategy}/"
            self.dirpath = f"{home_dirpath}{dirname}{self.slid_window_strategy}/"

        self.reader_class = self.FEATURE_READERS.get(feature_type, self.FEATURE_READERS['avg_mut'])

        print(f"Using directory path: {self.dirpath}")

        # 存储读取器对象以避免重复创建
        self.readers = {}
    
    def _get_feature_dir_path(self, feature_type):
        """
        Helper method to fetch the correct feature directory path based on the feature type.
        """
        return self.FEATURE_DIR_PATH.get(feature_type, self.DEFAULT_FEATURE_DIR_PATH)

    def _get_file_dir_path(self, file_type):
        """
        Helper method to fetch the correct file directory path based on the file type.
        """
        return self.FILE_DIR_PATH.get(file_type, self.DEFAULT_FILE_DIR_PATH)

    def _get_slide_window_strategy(self, slid_strategy):
        """
        Helper method to fetch the correct sliding window strategy based on the file type.
        """
        return self.SLIDE_WINDOWS_STRATEGY.get(slid_strategy, self.DEFAULT_SLIDE_WINDOW_STRATEGY)

    def get_reader(self, chrom: str, strand: str):
        get_genome_file = self._reader_choice(self.file_type)
        regional_feature_map = get_genome_file(chrom, strand)
        if (chrom, strand) not in self.readers:
            self.readers[(chrom, strand)] = self.reader_class(regional_feature_map)

        return self.readers[(chrom, strand)]

    def _reader_choice(self, file_type):
        if file_type == 'half_1in2000_train' or file_type == 'half_1in2000_test' or 'half_' in file_type:
            return self.get_genome_file
        elif file_type == 'half2_1in2000_train' or file_type == 'half2_1in2000_test' or file_type == 'half2_1in20000_train':
            return self.get_nonCpG_CGsites_genome_file
        elif file_type == 'half3_5in1000_train' or file_type == 'half3_5in1000_test':
            return self.get_CpGsites_genome_file
    
    def get_CpGsites_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.CpG_sites.filtered.all.SNP.subtypes.positive._{self.slid_window_strategy}.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.CpG_sites.filtered.all.SNP.subtypes.neg._{self.slid_window_strategy}.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def get_nonCpG_CGsites_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.nonCpG_CG_sites.filtered.all.SNP.subtypes.positive._{self.slid_window_strategy}.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.nonCpG_CG_sites.filtered.all.SNP.subtypes.neg._{self.slid_window_strategy}.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def get_genome_file(self, chrom: str, strand: str) -> str:
        if strand == '+':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.positive._{self.slid_window_strategy}.csv.gz"
        elif strand == '-':
            return f"{self.dirpath}{chrom}.AT_sites.filtered.all.SNP.subtypes.neg._{self.slid_window_strategy}.csv.gz"
        else:
            raise ValueError(f"Error: strand should be + or -, but input is {strand}")

    def close(self):
        self.reader.close()


class Segment_info_saver():
    def __init__(self, method):
        self.assert_method(method)
        self.method = method
        self._init(method)
    
    def _init(self, method):
        self.segment_feature_dict = self.get_segment_feature_dict(method)

    def save(self, segment_feature):
        if self.method in ['AvgSegKmerMut', 'AvgSegMutAndKmerMut', 'AvgStepMutAndKmerMut']:
            self.segment_feature_dict['segment_avg_kmer_mut'].append(segment_feature['segment_avg_kmer_mut'])
        
        if self.method != 'AvgSegKmerMut':
            self.segment_feature_dict['segment_avg_mut'].append(segment_feature['segment_avg_mut'])
        
        if self.method is None:
            self.segment_feature_dict['segment_id'].append(segment_feature)
    
    def output(self):
        return self.segment_feature_dict
    
    def get_segment_feature_dict(self, method):
        return get_soft_label_dict(method)
    
    def assert_method(self, method):
        if method not in ['SegMut', 'SegMutRate', 'AvgSegMutAndKmerMut', 'AvgSegKmerMut', 'AvgStepMutAndKmerMut', None]:
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

        elif method == 'AvgSegMutAndKmerMut' or method == 'AvgStepMutAndKmerMut':
            soft_label_dict['segment_avg_kmer_mut'] = []

    return soft_label_dict

def prepare_segment_feature(bed_regions, segment_center, method=None, path_type=None, slid_strategy=None, step_avg_strategy=None):
    """
    slid_strategy : to_del , only used for windowsXk_stepYk_avg_mut_rate
    """
    segmentInfoSaver = Segment_info_saver(method)
    segment_generator = bed_reader(bed_regions, segment_center)
    segment_feature_finder_dict = {}
    if method == 'AvgSegMutAndKmerMut' or method == 'AvgStepMutAndKmerMut':
        segment_feature_finder_dict['segment_avg_kmer_mut'] = SegmentFeatureFinder(feature='2mer_mut', file_type=path_type, slid_strategy=slid_strategy)
    if method == 'AvgStepMutAndKmerMut' or method == 'AvgStepMut':
        if step_avg_strategy is None:
            calc_strategy = 'radius10_bound100k_prob1to3'
        else:
            calc_strategy = step_avg_strategy
        segment_feature_finder_dict['segment_avg_mut'] = SegmentFeatureFinder(feature='step_avg_mut', file_type=path_type, slid_strategy=calc_strategy)
    else:
        segment_feature_finder_dict['segment_avg_mut'] = SegmentFeatureFinder(feature='avg_mut', file_type=path_type, slid_strategy=slid_strategy)
    #segment_feature_finder = SegmentFeatureFinder(feature=method, file_type=path_type, slid_strategy=slid_strategy)

    # get soft label
    for segment in segment_generator:
        segment_feature = {}
        for key, segment_feature_finder in segment_feature_finder_dict.items():
            segment_feature[key] = segment_feature_finder.find(segment)
        segmentInfoSaver.save(segment_feature)
        #segment_feature  = segment_feature_finder.find(segment)
        #segmentInfoSaver.save(segment_feature)

    # preprocessing soft label
    segment_feature_dict = segmentInfoSaver.output()

    return segment_feature_dict

class SegmentFeatureFinder():
    def __init__(self, feature=None, file_type=None, slid_strategy=None) -> None:
        
        self.feature = feature
        reader_factory = self._reader_choice(slid_strategy)
        self.reader_factory = reader_factory(file_type=file_type, feature_type=feature, slid_strategy=slid_strategy)
        self.filter_number = 0

    def find(self, segment_label_sites):

        if len(segment_label_sites[0]) < self.filter_number:
            pass

        segment_sites, chrom, strand = self.extract_segment_info(segment_label_sites)
        regional_feature_reader = self.reader_factory.get_reader(chrom, strand)
        segment_feature = [regional_feature_reader.find_region_feature(site.start) for site in segment_sites]

        return segment_feature


    def extract_segment_info(self, segment):
        sites, strand = segment
        chrom = sites[-1].chrom
        return sites, chrom, strand
    
    def _reader_choice(self, slid_strategy):
        if slid_strategy is None:
            return RegionalFeatureReaderFactory

        if slid_strategy.startswith('radius10_bound100k_'):
            return RegionalFeatureReaderFactory
        
        if os.path.isdir(slid_strategy) and self.feature == 'step_avg_mut':
            return EachSiteFeatureReaderFactory

        return RegionalFeatureReaderFactory

