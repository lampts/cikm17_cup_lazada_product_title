import os, sys

class FeatureManagement(object):
    
    '''
    get the current working directory
    '''
    def __init__(self):
        self.HOME_DIR = os.path.dirname(os.path.abspath(__file__))
    
    '''
    get extra features generated from text processing
    '''
    def get_text_features(self, mode, type, is_clarity = False):
        if mode == 0:
            #text features generated from title field
            if type == 0:
                #text features for general title
                if is_clarity:
                    return ["ct_" + str(i + 1) for i in range(2500)] + ["ct_sum", "ct_common"]
                else:
                    return ["ct_" + str(i + 1) for i in range(2700)] + ["ct_sum", "ct_common"]
            if type == 1:
                #text features for title classified by cat1
                return ["ct1_" + str(i + 1) for i in range(500)] + ["ct1_sum"]
            if type == 2:
                #text features for title classified by cat2
                return ["ct2_" + str(i + 1) for i in range(50)]
            if type == 3:
                #text features for title classified by cat1
                return ["ct3_" + str(i + 1) for i in range(100)]
            if type == 4:
                #text features for title based on cat1 dict, other features are all zero
                return ["ctd1_2", "ctd1_6", "ctd1_7", "ctd1_8", "ctd1_9", "ctd1_14", "ctd1_16", "ctd1_17", "ctd1_18"]
                #return ["ctd1_" + str(i + 1) for i in range(19)]
            if type == 5:
                #text features for title based on cat2 dict
                return ["ctd2_" + str(i + 1) for i in range(70)]
            if type == 6:
                #text features for title based on cat3 dict
                return ["ctd3_" + str(i + 1) for i in range(274)]
        elif mode == 1:
            if type == 0:
                # text features for general title
                return ["cd_" + str(i + 1) for i in range(200)]
            if type == 1:
                #text features for title classified by cat1
                return ["cd1_" + str(i + 1) for i in range(200)]
            if type == 4:
                #text features for desc based on cat1 dict
                return ["cdd1_" + str(i + 1) for i in range(19)]
            if type == 5:
                #text features for desc based on cat2 dict
                return ["cdd2_" + str(i + 1) for i in range(70)]
            if type == 6:
                #text features for desc based on cat3 dict
                return ["cdd3_" + str(i + 1) for i in range(274)]

    '''
    get the 6 original features
    country, price, ptype did contribute to the accuracy of the model, sku_code0
    sku_code0 is good: comparision between submissions 216 and 221
    '''
    def get_original_features(self, is_clarity=False):
        if is_clarity:
            return ["country", "price", "ptype", \
                "sku_code0", "sku_code1", "sku_code2"]
        else:
            return ["country", "price", "ptype", \
                    "sku_code0", "sku_code1", "sku_code2"]

    '''
    get statistics features 
    digit_redundancy and mixed_redundancy are not good features: compare submissions 268 and 269
    '''
    def get_statistics_features(self, mode, is_clarity=False):
        if is_clarity:
            if mode == 0:
                features_set_1 = ["title_num_digit", "title_num_mixed", "title_num_alpha", "title_num_words"]
                features_set_2 = ["title_alpha_redundancy", "title_num_brackets", "title_length_adjusted"]
                features_set_3 = ["title_num_stopwords", "title_num_specialwords", "title_num_stopspecialwords"]

                features_set_4 = ["title_num_measurement_units"]
                for feature in ["title_num_words", "ct_sum"]:
                    features_set_4.append(feature + "_delta_mean_cat1")
                    features_set_4.append(feature + "_delta_mean_cat2")
                    features_set_4.append(feature + "_delta_mean_cat3")

                    if feature == "title_num_words":
                        features_set_4.append(feature + "_delta_max_cat1")
                        features_set_4.append(feature + "_delta_max_cat2")
                        features_set_4.append(feature + "_delta_max_cat3")

                return features_set_1 + features_set_2 + features_set_3 + features_set_4

            if mode == 1:

                features_set_1 = ["desc_num_digit", "desc_num_mixed", "desc_num_alpha", "desc_num_words"]
                features_set_2 = ["desc_alpha_redundancy", "desc_num_litags", "desc_length_adjusted"]
                features_set_3 = ["desc_num_stopwords", "desc_num_specialwords", "desc_num_stopspecialwords"]

                features_set_4 = []
                for feature in ["words"]:
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat1")
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat2")
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat3")

                    features_set_4.append("desc_num_" + feature + "_delta_max_cat1")
                    features_set_4.append("desc_num_" + feature + "_delta_max_cat2")
                    features_set_4.append("desc_num_" + feature + "_delta_max_cat3")

                return features_set_1 + features_set_2 + features_set_3

        else:
            if mode == 0:
                features_set_1 = ["title_num_digit", "title_num_mixed", "title_num_alpha", "title_num_words"]
                features_set_2 = ["title_alpha_redundancy", "title_num_brackets", "title_length_adjusted"]
                features_set_3 = ["title_num_stopwords", "title_num_specialwords", "title_num_stopspecialwords"]

                features_set_4 = []
                for feature in ["alpha", "mixed", "digit", "words", "stopwords"]:
                    features_set_4.append("title_num_" + feature + "_delta_mean_cat1")
                    features_set_4.append("title_num_" + feature + "_delta_mean_cat2")
                    features_set_4.append("title_num_" + feature + "_delta_mean_cat3")

                    features_set_4.append("title_num_" + feature + "_delta_max_cat1")
                    features_set_4.append("title_num_" + feature + "_delta_max_cat2")
                    features_set_4.append("title_num_" + feature + "_delta_max_cat3")

                    if feature == "words":
                        features_set_4.append("title_num_" + feature + "_delta_mean_cat1_correction")
                        features_set_4.append("title_num_" + feature + "_delta_mean_cat2_correction")
                        features_set_4.append("title_num_" + feature + "_delta_mean_cat3_correction")

                features_set_5 = []
                for feature in ["title_avg_sim_score_cat1", "title_max_sim_score_cat1"]:
                    features_set_5.append(feature + "_delta_mean_cat1")
                    features_set_5.append(feature + "_delta_mean_cat2")
                    features_set_5.append(feature + "_delta_mean_cat3")

                    if feature != "ct_sum":
                        features_set_5.append(feature + "_delta_max_cat1")
                        features_set_5.append(feature + "_delta_max_cat2")
                        features_set_5.append(feature + "_delta_max_cat3")

                return features_set_1 + features_set_2 + features_set_3 + features_set_4 + features_set_5

            if mode == 1:
                features_set_1 = ["desc_num_digit", "desc_num_mixed", "desc_num_alpha", "desc_num_words"]
                features_set_2 = ["desc_alpha_redundancy", "desc_num_litags", "desc_length_adjusted"]
                features_set_3 = ["desc_num_stopwords", "desc_num_specialwords", "desc_num_stopspecialwords"]

                features_set_4 = []
                for feature in ["alpha", "mixed", "digit", "words"]:
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat1")
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat2")
                    features_set_4.append("desc_num_" + feature + "_delta_mean_cat3")

                    features_set_4.append("desc_num_" + feature + "_delta_max_cat1")
                    features_set_4.append("desc_num_" + feature + "_delta_max_cat2")
                    features_set_4.append("desc_num_" + feature + "_delta_max_cat3")

                return features_set_1 + features_set_2 + features_set_3 + features_set_4

    def get_basic_features(self, is_clarity=False, is_included_desc=True):
        basic_features = self.get_original_features(is_clarity=is_clarity)
        basic_features += self.get_statistics_features(mode=0, is_clarity=is_clarity)
        if is_included_desc:
            basic_features += self.get_statistics_features(mode=1, is_clarity=is_clarity)
        return basic_features

#===============================================================================
if __name__ == "__main__":
    feature_man = FeatureManagement()
    features = feature_man.get_basic_features(is_clarity=True) + \
               feature_man.get_text_features(mode=0, type=0, is_clarity=True) + \
               feature_man.get_text_features(mode=0, type=1)
    print(len(features))