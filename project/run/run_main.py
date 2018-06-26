from first.run.command_line import CommandLine
from first.run.config import BenchmarkConfiguration, CountVectorizerConfiguration, TfidfTransformerConfiguration


def get_data_providers():
    return ['reuters', 'newsgroups']


def get_count_configurations():
    max_df = 0.50
    min_df = 0.01
    lowercase = True
    stop_words = 'english'
    analyzer = 'word'
    strip_accents = 'unicode'

    max_features_list = [100, 200, 300, 500, 100]

    count_configurations_list = []
    for max_features in max_features_list:
        tmp = CountVectorizerConfiguration(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents)
        count_configurations_list.append(tmp)

    return count_configurations_list


def get_tfidf_configurations():
    use_idf = True
    sublinear_tf = True
    norm = None

    return [TfidfTransformerConfiguration(use_idf, sublinear_tf, norm)]


def get_configurations():
    data_providers = get_data_providers()
    count_configurations = get_count_configurations()
    tfidf_configurations = get_tfidf_configurations()

    configuration_list = []
    for data_provider in data_providers:
        for count_configuration in count_configurations:
            for tfidf_configuration in tfidf_configurations:
                tmp = BenchmarkConfiguration(count_configuration, tfidf_configuration, data_provider)
                configuration_list.append(tmp)

    return configuration_list


if __name__ == '__main__':
    URL = r'mongodb://markomihajlovicfm:itisme1994@ds115124.mlab.com:15124/intsys'
    db_name = r'intsys'
    interpreter_path = r'C:\Users\Maki\Anaconda3\envs\sci\python.exe'
    file_path = r'C:\Users\Maki\Documents\projects\IntSys\similarities\first\src\main.py'

    configurations = get_configurations()
    for configuration in configurations:
        command_line = CommandLine(interpreter_path, file_path, configuration)
        command_line.run_configuration(True, True)
