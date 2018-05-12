from first.run.config import BenchmarkConfiguration, CountVectorizerConfiguration, TfidfTransformerConfiguration, \
    DataConfiguration, FeaturesConfiguration

def get_data_configurations():
    dataset_names = ['reuters', 'newsgroups']

    data_configuration_list = []
    for dataset_name in dataset_names:
        data_configuration_list.append(DataConfiguration(dataset_name))

    return data_configuration_list


def get_count_configurations():
    max_df = 0.50
    min_df = 0.01
    lowercase = True
    stop_words = 'english'
    analyzer = 'word'
    strip_accents = 'unicode'

    # max_features_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 948]
    max_features_list = [50]

    count_configurations_list = []
    for max_features in max_features_list:
        tmp = CountVectorizerConfiguration(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents)
        count_configurations_list.append(tmp)

    return count_configurations_list


def get_tfidf_configurations():
    use_idf = True
    sublinear_tf = True
    norm = 'l2'

    return [TfidfTransformerConfiguration(use_idf, sublinear_tf, norm)]


def get_features_configurations():
    min_length = 3
    drop_zero_vectors = True
    data_configurations = get_data_configurations()
    count_configurations = get_count_configurations()
    tfidf_configurations = get_tfidf_configurations()

    features_configuration_list = []
    for data_configuration in data_configurations:
        for count_configuration in count_configurations:
            for tfidf_configuration in tfidf_configurations:
                tmp = FeaturesConfiguration(min_length, drop_zero_vectors, count_configuration, tfidf_configuration,
                                            data_configuration)
                features_configuration_list.append(tmp)

    return features_configuration_list


def get_benchmark_configurations():
    features_configurations = get_features_configurations()

    benchmark_configuration_list = []
    for features_configuration in features_configurations:
        tmp = BenchmarkConfiguration(features_configuration)
        benchmark_configuration_list.append(tmp)

    return benchmark_configuration_list


if __name__ == '__main__':
    # TODO add unit tests
    # TODO fork and update
    # TODO add validation dataset for estimating biases for each feature in feature vectors
    # TODO then use those features to improve performances of other classification algorithms -> maybe generic algorithms or smth similar
    # TODO t distribution for the first 20 acquired documents
    # TODO try with l2 normalization
    # TODO reduce sparsity by adding tsss values to the cosine measure where zero
    # train_features.shape= (9506, 948) - 948 is Max for reuters dataset
    # train_features.shape= (11314, 1600) - 1600 is Max for newsgroups dataset
    benchmark_configurations = get_benchmark_configurations()
    for benchmark_configuration in benchmark_configurations:
        benchmark_configuration.evaluation()
