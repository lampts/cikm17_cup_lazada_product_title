import os, sys, gc, csv, string
import pandas as pd
import nltk, heapq

from nltk.stem.porter import *
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from HTMLParser import HTMLParser

class DataPreparation(object):
    
    '''
    get the current working directory
    '''
    def __init__(self):
        self.HOME_DIR = os.path.dirname(os.path.abspath(__file__))
        self.html_parser = HTMLParser()
        self.digit_code = "alanturing"
        self.alnum_code = "donaldknuth"
        self.unit_code = "dijkstra"
        self.stop_code = "johnmccarthy"
        self.stop_words = None

        reload(sys)
        sys.setdefaultencoding('utf8')

        try:
            self.stops = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except:
            nltk.download()
            self.stops = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()

        self.specials = [".", "-"]
        self.stops = [
            "x", "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone",
            "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
            "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
            "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
            "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can",
            "co", "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg",
            "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
            "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five",
            "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go",
            "had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
            "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
            "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
            "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move",
            "much", "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now",
            "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
            "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
            "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
            "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
            "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter",
            "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
            "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards",
            "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were",
            "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
            "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why",
            "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]
        # c: celsius degree
        self.measurement_units = ["v", "a", "c", "w", "lm", "k", "g", "oz", "kg", "kgs", "m", "cm", "mm", "ml", "mg", "tc", "gb", "tb", "mb", "mah", "mhz", "vac", "inch", "inches", "ft", "ghz"]

    def rectify_unit(self, sentence):
        # unit
        input_list = sentence.split()
        output_list = []

        for text in input_list:
            if any(x in text for x in self.measurement_units):
                text = re.sub(r"(\d+)g", lambda m: m.group(1) + ' g', text)
                text = re.sub(r"(\d+)kgs", lambda m: m.group(1) + ' kg', text)
                text = re.sub(r"(\d+)kg", lambda m: m.group(1) + ' kg', text)
                text = re.sub(r"(\d+)m", lambda m: m.group(1) + ' m', text)
                text = re.sub(r"(\d+)cm", lambda m: m.group(1) + ' cm', text)
                text = re.sub(r"(\d+)mm", lambda m: m.group(1) + ' mm', text)
                text = re.sub(r"(\d+)ml", lambda m: m.group(1) + ' ml', text)
                text = re.sub(r"(\d+)mg", lambda m: m.group(1) + ' mg', text)
                text = re.sub(r"(\d+)pcs", lambda m: m.group(1) + ' pc', text)
                text = re.sub(r"(\d+)pc", lambda m: m.group(1) + ' pc', text)
                text = re.sub(r"(\d+)gb", lambda m: m.group(1) + ' gb', text)
                text = re.sub(r"(\d+)tb", lambda m: m.group(1) + ' tb', text)
                text = re.sub(r"(\d+)mb", lambda m: m.group(1) + ' mb', text)
                text = re.sub(r"(\d+)mah", lambda m: m.group(1) + ' mah', text)
                text = re.sub(r"(\d+)mhz", lambda m: m.group(1) + ' mhz', text)
                text = re.sub(r"(\d+)vac", lambda m: m.group(1) + ' vac', text)
                text = re.sub(r"(\d+)inch", lambda m: m.group(1) + ' inch', text)
                text = re.sub(r"(\d+)inches", lambda m: m.group(1) + ' inch', text)
                text = re.sub(r"(\d+)ft", lambda m: m.group(1) + ' ft', text)
                text = re.sub(r"(\d+)oz", lambda m: m.group(1) + ' oz', text)
                text = re.sub(r"(\d+)ghz", lambda m: m.group(1) + ' ghz', text)

            output_list.append(text)
        return " ".join(output_list)

    '''
    parse string to return lists of words, numbers and mixed
    processing mode = 0: get original words, 1: stemming, 2: lemmatizing
    '''
    def process_string(self, str_input, cat1, cat2, cat3, \
                       processing_mode = 1, sim_score_calculation=False):

        words_list = str_input.split()
        cat1_list = cat1.split()
        cat2_list = cat2.split()
        cat3_list = cat3.split()

        alpha_list = []
        digit_list = []
        mixed_list = []
        check_list = []
        processed_words_list = []

        num_stopwords = 0
        num_specialwords = 0
        num_measurement_unit = 0

        for word in words_list:
            word = word.lower()
            if processing_mode == 0:
                if word in self.stops:
                    num_stopwords += 1
                else:
                    try:
                        float(word)
                        digit_list.append(word)
                        word = self.digit_code
                    except:
                        if word.isalpha():
                            alpha_list.append(word)
                            check_list.append(word)
                        elif word.isalnum():
                            mixed_list.append(word)
                            word = self.alnum_code

                    processed_words_list.append(word)

            if processing_mode == 1:
                if word in self.stops:
                    num_stopwords += 1
                    # word = self.stop_code
                    processed_words_list.append(word)
                elif word in self.specials:
                    # ignore
                    pass
                else:
                    check_word = word
                    is_alnum_flag = False

                    if word in self.measurement_units:
                        num_measurement_unit += 1
                        # word = self.unit_code + word
                    else:
                        word = self.stemmer.stem(word)

                    if word in self.stops:
                        num_specialwords += 1
                        word = check_word
                    else:
                        try:
                            float(word)
                            # word = self.digit_code
                            digit_list.append(word)
                        except:
                            if word.isalpha():
                                alpha_list.append(word)
                                check_list.append(check_word)
                            elif word.isalnum():
                                is_alnum_flag = True

                    word_list = self.rectify_unit(word).split()
                    if len(word_list) == 1:
                        if is_alnum_flag:
                            # word = self.alnum_code
                            mixed_list.append(word)
                        processed_words_list.append(word)
                    else:
                        num_measurement_unit += 1
                        processed_words_list.append(word_list[0])
                        processed_words_list.append(word_list[1])
                        #processed_words_list.append(self.digit_code)
                        #processed_words_list.append(self.unit_code)

        # calcualte similirity scores among words as well as words and cat1, cat2, cat3
        if sim_score_calculation and processing_mode == 1:
            similarity_scores_cat1 = []
            similarity_scores_cat2 = []
            similarity_scores_cat3 = []

            for i in range(0, len(check_list)):
                try:
                    word1 = wordnet.synset(words_list[i] + ".n.01")
                    for j in range(0, len(cat1_list)):
                        try:
                            word2 = wordnet.synset(cat1_list[j] + ".n.01")
                            sim_score_cat1 = word1.wup_similarity(word2)
                        except:
                            sim_score_cat1 = 0.0
                    for j in range(0, len(cat2_list)):
                        try:
                            word2 = wordnet.synset(cat2_list[j] + ".n.01")
                            sim_score_cat2 = word1.wup_similarity(word2)
                        except:
                            sim_score_cat2 = 0.0
                    for j in range(0, len(cat3_list)):
                        try:
                            word2 = wordnet.synset(cat3_list[j] + ".n.01")
                            sim_score_cat3 = word1.wup_similarity(word2)
                        except:
                            sim_score_cat3 = 0.0
                except:
                    sim_score_cat1 = 0.0
                    sim_score_cat2 = 0.0
                    sim_score_cat3 = 0.0

                similarity_scores_cat1.append(sim_score_cat1)
                similarity_scores_cat2.append(sim_score_cat2)
                similarity_scores_cat3.append(sim_score_cat3)

            if len(similarity_scores_cat1) == 0:
                sim_score_avg_cat1 = 0.0
                sim_score_max_cat1 = 0.0
            else:
                sim_score_avg_cat1 = sum(similarity_scores_cat1) / float(len(similarity_scores_cat1))
                sim_score_max_cat1 = max(similarity_scores_cat1)

            if len(similarity_scores_cat2) == 0:
                sim_score_avg_cat2 = 0.0
                sim_score_max_cat2 = 0.0
            else:
                sim_score_avg_cat2 = sum(similarity_scores_cat2) / float(len(similarity_scores_cat2))
                sim_score_max_cat2 = max(similarity_scores_cat2)

            if len(similarity_scores_cat3) == 0:
                sim_score_avg_cat3 = 0.0
                sim_score_max_cat3 = 0.0
            else:
                sim_score_avg_cat3 = sum(similarity_scores_cat3) / float(len(similarity_scores_cat3))
                sim_score_max_cat3 = max(similarity_scores_cat3)
        else:
            sim_score_avg_cat1 = 0.0
            sim_score_max_cat1 = 0.0
            sim_score_avg_cat2 = 0.0
            sim_score_max_cat2 = 0.0
            sim_score_avg_cat3 = 0.0
            sim_score_max_cat3 = 0.0

        return alpha_list, digit_list, mixed_list, words_list, " ".join(processed_words_list), \
               num_stopwords, num_specialwords, num_measurement_unit, sim_score_avg_cat1, sim_score_max_cat1, \
               sim_score_avg_cat2, sim_score_max_cat2, sim_score_avg_cat3, sim_score_max_cat3

    '''
    customize, combine training and testing data into a single file         
    '''
    def build_combination(self, processing_mode = 1, out_filename = "data_all.csv"):
        # filenames
        clarify_filename = os.path.join(self.HOME_DIR, "input", "clarity_train.labels")
        concise_filename = os.path.join(self.HOME_DIR, "input", "conciseness_train.labels")
        train_filename = os.path.join(self.HOME_DIR, "input", "data_train.csv")
        valid_filename = os.path.join(self.HOME_DIR, "input", "data_valid.csv")
        test_filename = os.path.join(self.HOME_DIR, "input", "data_test.csv")
        out_filename = os.path.join(self.HOME_DIR, "input", out_filename)

        # file handlers
        clarify_infile = open(clarify_filename, "r")
        concise_infile = open(concise_filename, "r")
        train_infile = open(train_filename, "r")
        valid_infile = open(valid_filename, "r")
        test_infile = open(test_filename, "r")
        outfile = open(out_filename, "w")

        # setup column names
        outfile.write("country,sku,title,cat1,cat2,cat3,desc,price,ptype,")
        outfile.write("sku_code0,sku_code1,sku_code2,sku_length,")
        outfile.write("title_length,title_num_alpha,title_num_digit,title_num_mixed,")
        outfile.write("title_num_words,title_num_brackets,title_length_adjusted,")
        outfile.write("title_avg_sim_score_cat1,title_max_sim_score_cat1,")
        outfile.write("title_avg_sim_score_cat2,title_max_sim_score_cat2,")
        outfile.write("title_avg_sim_score_cat3,title_max_sim_score_cat3,")
        outfile.write("title_all_redundancy,title_digit_redundancy,title_mixed_redundancy,title_alpha_redundancy,")
        outfile.write("title_num_stopwords,title_num_specialwords,title_num_measurement_units,")

        outfile.write("desc_length,desc_num_alpha,desc_num_digit,desc_num_mixed,")
        outfile.write("desc_num_words,desc_num_litags,desc_length_adjusted,")
        outfile.write("desc_avg_sim_score_cat1,desc_max_sim_score_cat1,")
        outfile.write("desc_avg_sim_score_cat2,desc_max_sim_score_cat2,")
        outfile.write("desc_avg_sim_score_cat3,desc_max_sim_score_cat3,")
        outfile.write("desc_all_redundancy,desc_digit_redundancy,desc_mixed_redundancy,desc_alpha_redundancy,")
        outfile.write("desc_num_stopwords,desc_num_specialwords,desc_num_measurement_units,")
        outfile.write("left_alpha,middle_alpha,right_alpha,")
        outfile.write("left_digit,middle_digit,right_digit,")
        outfile.write("left_mixed,middle_mixed,right_mixed,")
        outfile.write("label,clarity,conciseness\n")

        special_chars = ["(", ")", "/", ":", "[", "]", "+", ";", ","]
        p_tag_removal = re.compile(r'<.*?>')
        p_non_ascii_removal = re.compile(r'[^\x00-\x7F]')

        count = 0
        for index in range(3):
            if index == 0:
                in_reader = csv.reader(train_infile)
            elif index == 1:
                in_reader = csv.reader(valid_infile)
                #continue
            else:
                in_reader = csv.reader(test_infile)

            for row in in_reader:
                count+=1
                if count%1000 == 0:
                    print("Finished {} lines".format(count))

                sku = row[1]
                sku_code0 = sku[:5]
                sku_code1 = sku[5:7]
                sku_code2 = sku[7:9]
                sku_length = len(sku)

                # process title
                #row[2] = "Reflective&nbsp;Shoe&nbsp;Laces&nbsp;Runner&nbsp;Shoelaces&nbsp;3M&nbsp;1Pair&nbsp;Black&nbsp;"
                row[2] = p_non_ascii_removal.sub(' ', row[2])
                row[2] = self.html_parser.unescape(row[2])
                row[2] = " ".join(row[2].split())
                num_brackets_title = len(row[2].split("(")) - 1
                length_title = len(row[2])
                text_title = row[2]
                for special_char in special_chars:
                    text_title = text_title.replace(special_char, " ")
                text_title = "".join([c for c in text_title if c.isalnum() or c in string.whitespace or c=="-" or c=="."])
                text_title = " ".join(re.split('(\d+[\.]?\d*)', text_title))

                if processing_mode == 0:
                    text_title = self.rectify_unit(text_title)
                alpha_list_title, digit_list_title, mixed_list_title, all_list_title, text_title, \
                    num_stopwords_title, num_specialwords_title, num_measurement_units_title, \
                    avg_sim_score_title_cat1, max_sim_score_title_cat1, \
                    avg_sim_score_title_cat2, max_sim_score_title_cat2, \
                    avg_sim_score_title_cat3, max_sim_score_title_cat3 = \
                    self.process_string(text_title, row[3],  row[4], row[5], processing_mode=processing_mode, sim_score_calculation=True)
                length_title_adjusted = len(text_title)
                redundant_all_title = len(all_list_title) - len(set(all_list_title))
                redundant_digit_title = len(digit_list_title) - len(set(digit_list_title))
                redundant_mixed_title = len(mixed_list_title) - len(set(mixed_list_title))
                redundant_alpha_title = len(alpha_list_title) - len(set(alpha_list_title))
                row[2] = text_title

                # process desc
                row[6] = p_non_ascii_removal.sub(' ', row[6])
                row[6] = self.html_parser.unescape(row[6])
                row[6] = " ".join(row[6].split())
                num_li_desc = row[6].count("<li>")
                length_desc = len(row[6])
                text_desc = p_tag_removal.sub(' ', row[6])
                for special_char in special_chars:
                    text_desc = text_desc.replace(special_char, " ")
                text_desc = "".join([c for c in text_desc if c.isalnum() or c in string.whitespace or c=='-' or c=="."])
                text_desc = " ".join(re.split('(\d+[\.]?\d*)', text_desc))
                if processing_mode == 0:
                    text_desc = self.rectify_unit(text_desc)
                alpha_list_desc, digit_list_desc, mixed_list_desc, all_list_desc, text_desc, \
                    num_stopwords_desc, num_specialwords_desc, num_measurement_units_desc, \
                    avg_sim_score_desc_cat1, max_sim_score_desc_cat1, \
                    avg_sim_score_desc_cat2, max_sim_score_desc_cat2, \
                    avg_sim_score_desc_cat3, max_sim_score_desc_cat3 = \
                    self.process_string(text_desc, row[3], row[4], row[5], processing_mode=processing_mode, sim_score_calculation=True)
                length_desc_adjusted = len(text_desc)
                redundant_all_desc = len(all_list_desc) - len(set(all_list_desc))
                redundant_digit_desc = len(digit_list_desc) - len(set(digit_list_desc))
                redundant_mixed_desc = len(mixed_list_desc) - len(set(mixed_list_desc))
                redundant_alpha_desc = len(alpha_list_desc) - len(set(alpha_list_desc))
                row[6] = text_desc

                #get overlapping words between title and desc
                overlapping_alpha = list(set(alpha_list_title) & set(alpha_list_desc))
                overlapping_digit = list(set(digit_list_title) & set(digit_list_desc))
                overlapping_mixed = list(set(mixed_list_title) & set(mixed_list_desc))

                #extra values from set of words in title and desc
                left_alpha = len(alpha_list_title) - len(overlapping_alpha)
                left_digit = len(digit_list_title) - len(overlapping_digit)
                left_mixed = len(mixed_list_title) - len(overlapping_mixed)
                middle_alpha = len(overlapping_alpha)
                middle_digit = len(overlapping_digit)
                middle_mixed = len(overlapping_mixed)
                right_alpha = len(alpha_list_desc) - len(overlapping_alpha)
                right_digit = len(digit_list_desc) - len(overlapping_digit)
                right_mixed = len(mixed_list_desc) - len(overlapping_mixed)

                outline = row[0]
                for i in range(1, 9):
                    if row[i].find(",") != -1:
                        outline += ",\"" + row[i] + "\""
                    else:
                        outline += "," + row[i]

                #add extra features
                outline += "," + sku_code0
                outline += "," + sku_code1
                outline += "," + sku_code2
                outline += "," + str(sku_length)

                outline += "," + str(length_title)
                outline += "," + str(len(alpha_list_title))
                outline += "," + str(len(digit_list_title))
                outline += "," + str(len(mixed_list_title))
                outline += "," + str(len(all_list_title))
                outline += "," + str(num_brackets_title)
                outline += "," + str(length_title_adjusted)

                outline += "," + str(avg_sim_score_title_cat1)
                outline += "," + str(max_sim_score_title_cat1)
                outline += "," + str(avg_sim_score_title_cat2)
                outline += "," + str(max_sim_score_title_cat2)
                outline += "," + str(avg_sim_score_title_cat3)
                outline += "," + str(max_sim_score_title_cat3)

                outline += "," + str(redundant_all_title)
                outline += "," + str(redundant_digit_title)
                outline += "," + str(redundant_mixed_title)
                outline += "," + str(redundant_alpha_title)
                outline += "," + str(num_stopwords_title)
                outline += "," + str(num_specialwords_title)
                outline += "," + str(num_measurement_units_title)

                outline += "," + str(length_desc)
                outline += "," + str(len(alpha_list_desc))
                outline += "," + str(len(digit_list_desc))
                outline += "," + str(len(mixed_list_desc))
                outline += "," + str(len(all_list_desc))
                outline += "," + str(num_li_desc)
                outline += "," + str(length_desc_adjusted)

                outline += "," + str(avg_sim_score_desc_cat1)
                outline += "," + str(max_sim_score_desc_cat1)
                outline += "," + str(avg_sim_score_desc_cat2)
                outline += "," + str(max_sim_score_desc_cat2)
                outline += "," + str(avg_sim_score_desc_cat3)
                outline += "," + str(max_sim_score_desc_cat3)

                outline += "," + str(redundant_all_desc)
                outline += "," + str(redundant_digit_desc)
                outline += "," + str(redundant_mixed_desc)
                outline += "," + str(redundant_alpha_desc)
                outline += "," + str(num_stopwords_desc)
                outline += "," + str(num_specialwords_desc)
                outline += "," + str(num_measurement_units_desc)

                outline += "," + str(left_alpha)
                outline += "," + str(middle_alpha)
                outline += "," + str(right_alpha)
                outline += "," + str(left_digit)
                outline += "," + str(middle_digit)
                outline += "," + str(right_digit)
                outline += "," + str(left_mixed)
                outline += "," + str(middle_mixed)
                outline += "," + str(right_mixed)

                if index == 0:
                    clarify = clarify_infile.readline().strip()
                    concise = concise_infile.readline().strip()
                    outline += ",0," + clarify + "," + concise +  "\n"
                else:
                    outline += "," + str(index) + ",0.5,0.5\n"
                outfile.write(outline)

        #close files
        clarify_infile.close()
        concise_infile.close()
        train_infile.close()
        valid_infile.close()
        test_infile.close()
        outfile.close()

    '''
    fillNA
    '''
    def fill_NA(self, df_input):
        checking_features = ["country", "title", "cat1", "cat2", "cat3", "ptype", "desc", "price"]
        for feature in checking_features:
            if feature != "price":
                df_input[feature].fillna("", inplace=True)
            else:
                df_input[feature].fillna(0, inplace=True)

    '''
    generate a new data frame counting number of good only keywords and bad only keywords in text column
    '''
    def get_num_good_bad_keywords(self, df_input, text_column, target_column, new_feature_prefix):
        print("Expected_length {}".format(len(df_input)))

        # check isnull
        if df_input[text_column].isnull().sum() > 0:
            print(df_input[(df_input[text_column].isnull())])
            df_input[text_column].fillna("", inplace=True)

        df_input_pos = df_input[(df_input[target_column] == 1)]
        df_input_neg = df_input[(df_input[target_column] == 0)]

        vectorizer_pos = CountVectorizer(stop_words=self.stop_words)
        vectorizer_pos.fit_transform(df_input_pos[text_column])
        keywords_pos = vectorizer_pos.get_feature_names()

        vectorizer_neg = CountVectorizer(stop_words=self.stop_words)
        vectorizer_neg.fit_transform(df_input_neg[text_column])
        keywords_neg = vectorizer_neg.get_feature_names()
        print("Total number of pos keywords {}, neg keywords {}".format(len(keywords_pos), len(keywords_neg)))

        good_only_keywords = list(set(keywords_pos) - set(keywords_neg))
        bad_only_keywords = list(set(keywords_neg) - set(keywords_pos))
        common_keywords = list(set(keywords_pos) & set(keywords_neg))

        print("Good only keywords {}, bad only keywords {}, common keywords {}".format(len(good_only_keywords), len(bad_only_keywords), len(common_keywords)))
        #list_keywords = [good_only_keywords, bad_only_keywords, common_keywords]
        list_keywords = [common_keywords]
        column_values = []

        for list_index in range(1):
            tmp_file = os.path.join(self.HOME_DIR, "tmp", new_feature_prefix + str(list_index + 1) + "_tmpgb.csv")

            selected_features = list_keywords[list_index]
            print("Length of selected features {}".format(len(selected_features)))
            vectorizer_dict = {k: v for v, k in enumerate(selected_features)}
            vectorizer = CountVectorizer(stop_words=self.stop_words, vocabulary=vectorizer_dict)

            vectors_raw = vectorizer.fit_transform(df_input[text_column])
            vectors = vectors_raw.todense()

            # generate new dataframe and return the result
            df_tmp = pd.DataFrame(vectors)
            df_tmp.to_csv(tmp_file, index=False, header=None)

            del df_tmp
            del vectors
            del vectors_raw
            del vectorizer
            del vectorizer_dict
            gc.collect()

            CHUNK_SIZE = 5000
            list_values = []
            chunks = pd.read_csv(tmp_file, header=None, chunksize=CHUNK_SIZE)
            for chunk in chunks:
                sum_values = chunk.sum(axis=1)
                list_values.extend(sum_values)
                print("Length of list_values {}".format(len(list_values)))

            column_values.append(list_values)
            del chunks
            gc.collect()

        #column_names = [new_feature_prefix + "good", new_feature_prefix + "bad", new_feature_prefix + "common"]
        column_names = [new_feature_prefix + "common"]
        sum_data = {column_names[0]: column_values[0]}
        #column_names[1]: column_values[1],
        #column_names[2]: column_values[2]}

        df_output = pd.DataFrame(sum_data)
        #df_output[new_feature_prefix + "ratio"] = df_output[column_names[2]]/(df_output[column_names[0]] + df_output[column_names[1]] + df_output[column_names[2]])
        print(df_output.head(5))
        return df_output

    '''
    select the top or bottom keywords to build dict
    position: "top" or "bottom"
    flag index: 0 for df_train_pos, 1 for df_train_neg
    '''
    def select_keywords(self, df_input, text_column, target_column, num_of_keywords, position = "top", flags = [True, True, True]):
        print("Select num of keywords {}, position {}, flags {}, text_column {}".format(num_of_keywords, position, flags, text_column))

        # separate train and test data
        df_train = df_input[(df_input["label"] == 0)]
        df_test = df_input[(df_input["label"] != 0)]

        # separate train into positive and negative training
        df_train_pos = df_train[df_train[target_column] == 1]
        df_train_neg = df_train[df_train[target_column] == 0]

        # get total number of keywords in train
        vectorizer = CountVectorizer(stop_words=self.stop_words, vocabulary=None)
        vectorizer.fit_transform(df_train[text_column])
        all_keywords_train = set(vectorizer.get_feature_names())
        num_keywords_train = len(all_keywords_train)
        print("Total number of keywords in train {}".format(num_keywords_train))

        # get total number of keywords in train_pos
        vectorizer = CountVectorizer(stop_words=self.stop_words)
        vectorizer.fit_transform(df_train_pos[text_column])
        all_keywords_train_pos = set(vectorizer.get_feature_names())
        num_keywords_train_pos = len(all_keywords_train_pos)
        print("Total nuber of keywords in train_pos {}".format(num_keywords_train_pos))

        # get total number of keywords in train_neg
        vectorizer = CountVectorizer(stop_words=self.stop_words)
        vectorizer.fit_transform(df_train_neg[text_column])
        all_keywords_train_neg = set(vectorizer.get_feature_names())
        num_keywords_train_neg = len(all_keywords_train_neg)
        print("Total number of keywords in train_neg {}".format(num_keywords_train_neg))

        # get total number of keywords in test
        vectorizer = CountVectorizer(stop_words=self.stop_words)
        vectorizer.fit_transform(df_test[text_column])
        all_keywords_test = set(vectorizer.get_feature_names())
        num_keywords_test = len(all_keywords_test)
        print("Total number of keywords in test {}".format(num_keywords_test))

        all_keywords_train_both = all_keywords_train_pos & all_keywords_train_neg
        print("Total number of keywords in both train_pos and train_neg {}".format(len(all_keywords_train_both)))

        # initialize max keywords to collect
        if position == "top":
            max_keywords_train = num_of_keywords
            max_keywords_test = num_of_keywords
        else:
            max_keywords_train = min(num_keywords_train, num_keywords_test) - num_of_keywords
            max_keywords_test = min(num_keywords_train, num_keywords_test) - num_of_keywords

        expected_num_of_keywords = num_of_keywords
        while 1:
            print("Max features train {}, test {}".format(max_keywords_train, max_keywords_test))

            vectorizer = CountVectorizer(max_features=max_keywords_train, stop_words=self.stop_words)
            vectorizer.fit_transform(df_train_pos[text_column])
            top_keywords_train_pos = set(vectorizer.get_feature_names())
            bottom_keywords_train_pos = all_keywords_train_pos - top_keywords_train_pos

            vectorizer = CountVectorizer(max_features=max_keywords_train, stop_words=self.stop_words)
            vectorizer.fit_transform(df_train_neg[text_column])
            top_keywords_train_neg = set(vectorizer.get_feature_names())
            bottom_keywords_train_neg = all_keywords_train_neg - top_keywords_train_neg

            if flags[0] and flags[1] and flags[2]:
                vectorizer = CountVectorizer(max_features=max_keywords_train, stop_words=self.stop_words)
                vectorizer.fit_transform(df_train[text_column])
                top_keywords_train = set(vectorizer.get_feature_names())
                bottom_keywords_train = all_keywords_train - top_keywords_train
            else:
                if flags[0] and flags[1]:
                    top_keywords_train = top_keywords_train_pos & top_keywords_train_neg
                    bottom_keywords_train = bottom_keywords_train_pos & bottom_keywords_train_neg
                else:
                    if flags[0]:
                        top_keywords_train = top_keywords_train_pos
                        bottom_keywords_train = bottom_keywords_train_pos
                    elif flags[1]:
                        top_keywords_train = top_keywords_train_neg
                        bottom_keywords_train = bottom_keywords_train_neg

            vectorizer = CountVectorizer(max_features=max_keywords_test, stop_words=self.stop_words)
            vectorizer.fit_transform(df_test[text_column])
            top_keywords_test = set(vectorizer.get_feature_names())
            bottom_keywords_test = all_keywords_test - top_keywords_test

            if position == "top":
                select_keywords = top_keywords_train & top_keywords_test
            else:
                select_keywords = bottom_keywords_train & bottom_keywords_test

            if not flags[2]:
                if not flags[0]:
                    select_keywords = select_keywords - all_keywords_train_pos
                if not flags[1]:
                    select_keywords = select_keywords - all_keywords_train_neg

            print("Number of select keywords {}".format(len(select_keywords)))
            num_missing_keywords = num_of_keywords - len(select_keywords)

            if num_missing_keywords == 0:
                break

            if position == "top":
                if num_missing_keywords <= 2:
                    if max_keywords_train == max_keywords_test:
                        max_keywords_train += 1
                    else:
                        max_keywords_test += 1
                else:
                    max_keywords_train += (num_missing_keywords / 2)
                    max_keywords_test += (num_missing_keywords / 2)
            else:
                if num_missing_keywords <= 2:
                    if max_keywords_train == max_keywords_test:
                        max_keywords_train -= 1
                    else:
                        max_keywords_test -= 1
                else:
                    max_keywords_train -= (num_missing_keywords / 2)
                    max_keywords_test -= (num_missing_keywords / 2)

        return select_keywords

    '''
    encode sku
    '''
    def build_sku_features(self, df_input, max_features):
        vectorizer = CountVectorizer(max_features = max_features, ngram_range=(2, 3), analyzer='char')
        training_vectors_raw = vectorizer.fit_transform(df_input["sku"])
        training_vectors = training_vectors_raw.todense()
        extra_columns = ["sku_" + str(i + 1) for i in range(max_features)]

        # generate new dataframe and return the result
        df_output = pd.DataFrame(training_vectors)
        df_output.columns = extra_columns
        return df_output

    '''
    get delta length according to cats
    '''
    def get_delta_length(self, df_input, column_name, cat_name="cat1", is_mean=True):

        '''
        print("Get delta length of column {}, is_mean {}".format(column_name, is_mean))
        if column_name == "price":
            print(df_input[["price", "cat1"]])
            df_input["price"].fillna(0, inplace=True)
        '''

        if is_mean:
            new_feature = column_name + '_delta_mean_' + cat_name
            df_mean = df_input.groupby([cat_name])[column_name].mean()
            df_mean = df_mean.reset_index()
            df_mean.columns = [cat_name, new_feature]
            df_output = pd.merge(df_input, df_mean, on=cat_name, how="left")
            df_output[new_feature + "_correction"] = df_output[new_feature] - df_output[column_name]
        else:
            new_feature = column_name + '_delta_max_' + cat_name
            df_max = df_input.groupby([cat_name])[column_name].max()
            df_max = df_max.reset_index()
            df_max.columns = [cat_name, new_feature]
            df_output = pd.merge(df_input, df_max, on=cat_name, how="left")
            df_output[new_feature + "_correction"] = df_output[new_feature] - df_output[column_name]

        return df_output

    '''
    generate a new data frame containing generated features
    '''
    def generate_features(self, df_input, text_column, target_column, num_top_features, num_common_features, num_bottom_features, new_feature_prefix, new_feature_index=0):

        # check isnull
        if df_input[text_column].isnull().sum() > 0:
            print(df_input[(df_input[text_column].isnull())])
            df_input[text_column].fillna("", inplace=True)

        top_features = set([])
        if num_top_features > 0:
            # get top good keywords (they are keywords in train_pos and in test, but not in train_neg)
            top_features = self.select_keywords(df_input, text_column, target_column, num_top_features, position="bottom", flags = [True, False, False])

        bottom_features = set([])
        if num_bottom_features > 0:
            # get top bad keywords (they are keywords in train_neg and in test, but not in train_pos
            bottom_features = self.select_keywords(df_input, text_column, target_column, num_bottom_features, position="top", flags = [False, True, False])

        common_features = set([])
        if num_common_features > 0:
            # get top common keywords that appear in both train and test
            common_features = self.select_keywords(df_input, text_column, target_column, num_common_features, position="top", flags = [True, True, True])

        print("Number of top, bottom, common keywords {}, {}, {}".format(len(top_features), len(bottom_features), len(common_features)))
        selected_features = list(top_features.union(bottom_features).union(common_features))

        if (len(selected_features) != (num_top_features + num_bottom_features + num_common_features)):
            print("Attention! Number of selected features {}".format(len(selected_features)))

        # get top features from all data
        vectorizer_dict = {k: v for v, k in enumerate(selected_features)}
        num_extra_features = len(vectorizer_dict)

        vectorizer = CountVectorizer(stop_words=self.stop_words, vocabulary=vectorizer_dict)
        training_vectors_raw = vectorizer.fit_transform(df_input[text_column])
        training_vectors = training_vectors_raw.todense()
        extra_columns = [new_feature_prefix + str(i + 1) for i in range(num_extra_features)]

        #generate new dataframe and return the result
        df_output = pd.DataFrame(training_vectors)
        df_output.columns = extra_columns
        df_output[new_feature_prefix + "sum"] = df_output.sum(axis=1)

        if new_feature_prefix == "ct1_":
            df_output[extra_columns] += (1000 * new_feature_index)
            df_output["ct1_sum"] += (1000 * new_feature_index)

        return df_output

    '''
    convert categorical data to numerical data 
    '''
    def clean_data(self, target_column, verbose = 0):
        #get input
        in_filename = os.path.join(self.HOME_DIR, "input", "data_all_" + target_column + ".csv")
        df_input = pd.read_csv(in_filename)
        if "index" not in df_input.columns:
            print("Adding index column")
            df_input.to_csv(in_filename, index=True, index_label="index")
            df_input = pd.read_csv(in_filename)

        if "sku_config0" not in df_input.columns:
            print("Adding sku_config0 column")
            df_input["sku_config0"] = df_input["sku_code0"]
        if "sku_config1" not in df_input.columns:
            print("Adding sku_config1 column")
            df_input["sku_config1"] = df_input["sku_code1"]
        if "sku_config2" not in df_input.columns:
            print("Adding sku_config2 column")
            df_input["sku_config2"] = df_input["sku_code2"]

        #fill NA values
        self.fill_NA(df_input)

        # build dictionaries for cats at 3 levels
        dict_list = []
        for i in range(3):
            vectorizer = CountVectorizer()
            vectorizer.fit_transform(df_input["cat" + str(i + 1)])
            dict_cat = {k: v for v, k in enumerate(vectorizer.get_feature_names())}
            dict_list.append(dict_cat)

            del vectorizer
            gc.collect()

        # convert categorical features to numerical features
        encoder = LabelEncoder()
        categorical_features = ["country", "cat1", "cat2", "cat3", "ptype", "sku_code0", "sku_code1", "sku_code2"]
        for feature in categorical_features:
            df_input[feature].fillna("unknown", inplace=True)
            encoder.fit(df_input[feature])
            df_input[feature] = encoder.transform(df_input[feature])

        print("Before getting general extra features {}".format(df_input.shape))
        df_input["ratio_title_desc"] = df_input["title_num_words"] / df_input["desc_num_words"]

        for feature in ["title_avg_sim_score_cat1", "title_max_sim_score_cat1", \
                        "title_avg_sim_score_cat2", "title_max_sim_score_cat2", \
                        "title_avg_sim_score_cat3", "title_max_sim_score_cat3",
                        "ratio_title_desc", "price", "country"]:
            df_input = self.get_delta_length(df_input, feature, "cat1", is_mean=True)
            df_input = self.get_delta_length(df_input, feature, "cat3", is_mean=True)
            df_input = self.get_delta_length(df_input, feature, "cat2", is_mean=True)

            df_input = self.get_delta_length(df_input, feature, "cat1", is_mean=False)
            df_input = self.get_delta_length(df_input, feature, "cat2", is_mean=False)
            df_input = self.get_delta_length(df_input, feature, "cat3", is_mean=False)

        for posfix in ["words", "alpha", "mixed", "digit", "stopwords", "specialwords"]:
            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat1", is_mean=True)
            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat2", is_mean=True)
            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat3", is_mean=True)

            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat1", is_mean=False)
            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat2", is_mean=False)
            df_input = self.get_delta_length(df_input, "title_num_" + posfix, "cat3", is_mean=False)

            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat1", is_mean=True)
            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat2", is_mean=True)
            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat3", is_mean=True)

            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat1", is_mean=False)
            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat2", is_mean=False)
            df_input = self.get_delta_length(df_input, "desc_num_" + posfix, "cat3", is_mean=False)

        # df_extra_sku = self.build_sku_features(df_input, max_features=100)
        df_extra_title = self.generate_features(df_input, "title", target_column, 0, 2700, 0, "ct_")
        df_sum_title = self.get_num_good_bad_keywords(df_input, "title", target_column, "ct_")
        df_input = pd.concat([df_input, df_extra_title, df_sum_title], axis=1)

        for feature in ["ct_sum"]:
            df_input = self.get_delta_length(df_input, feature, "cat1", is_mean=True)
            df_input = self.get_delta_length(df_input, feature, "cat3", is_mean=True)
            df_input = self.get_delta_length(df_input, feature, "cat2", is_mean=True)

            df_input = self.get_delta_length(df_input, feature, "cat1", is_mean=False)
            df_input = self.get_delta_length(df_input, feature, "cat2", is_mean=False)
            df_input = self.get_delta_length(df_input, feature, "cat3", is_mean=False)

        print("Before getting cat extra features {}".format(df_input.shape))

        # generate extra features for cat1, cat2 and cat3
        for cat in ["1"]:
            concat_list = []
            if cat == "1":
                number_of_cats = 9
                num_top_features = 0
                num_common_features = 500
                num_bottom_features = 0
            if cat == "2":
                number_of_cats = 57
                num_top_features = 0
                num_common_features = 0
                num_bottom_features = 0
            if cat == "3":
                number_of_cats = 185
                num_top_features = 10
                num_common_features = 0
                num_bottom_features = 0

            for i in range(number_of_cats):
                df_train = df_input[(df_input["cat" + str(cat)]==i)]
                print("Initial {}".format(df_train.shape))

                tmp_file = os.path.join(self.HOME_DIR, "tmp", "tmp.cat" + cat + "." + str(i))
                df_train.to_csv(tmp_file, index=False)
                df_train = pd.read_csv(tmp_file)

                print("Before {}".format(df_train.shape))

                df_extra_title = self.generate_features(df_train, "title", target_column, num_top_features, num_common_features, num_bottom_features, "ct" + cat + "_", new_feature_index=i)
                df_sum_title = self.get_num_good_bad_keywords(df_train, "title", target_column, "ct" + cat + "_")

                print("Extra title {}, sum title {}".format(df_extra_title.shape, df_sum_title.shape))
                df_train = pd.concat([df_train, df_extra_title, df_sum_title], axis=1)
                print("After {}".format(df_train.shape))
                df_train.to_csv(tmp_file, index=False)
                concat_list.append(df_train)

            df_input = pd.concat(concat_list, axis=0)
            tmp_file = os.path.join(self.HOME_DIR, "input", "tmp.csv.all")
            df_input.to_csv(tmp_file, index=False)

            #clear memory
            for df_tmp in concat_list:
                del df_tmp
            del concat_list
            del df_input
            gc.collect()

            df_input = pd.read_csv(tmp_file, dtype={"sku": str, "title": str, "desc": str})
            self.fill_NA(df_input)
            print(df_input.shape)

        #continue to process
        print("Done, continue!")
        print(df_input.shape)
        concat_list = [df_input]

        #extra features
        for i in range(3):
            dict_cat = dict_list[i]
            num_extra_features = len(dict_cat)

            vectorizer_title = CountVectorizer(stop_words=self.stop_words, vocabulary=dict_cat)
            training_vectors_raw_title = vectorizer_title.fit_transform(df_input["title"])
            training_vectors_title = training_vectors_raw_title.todense()
            extra_columns_title = ["ctd" + str(i + 1) + "_" + str(j + 1) for j in range(num_extra_features)]
            df_extra_title = pd.DataFrame(training_vectors_title)
            df_extra_title.columns = extra_columns_title
            df_extra_title["ctd" + str(i + 1) + "_sum"] = df_extra_title.sum(axis=1)
            concat_list.append(df_extra_title)

            vectorizer_desc = CountVectorizer(stop_words=self.stop_words, vocabulary=dict_cat)
            training_vectors_raw_desc = vectorizer_desc.fit_transform(df_input["desc"])
            training_vectors_desc = training_vectors_raw_desc.todense()
            extra_columns_desc = ["cdd" + str(i + 1) + "_" + str(j + 1) for j in range(num_extra_features)]
            df_extra_desc = pd.DataFrame(training_vectors_desc)
            df_extra_desc.columns = extra_columns_desc
            df_extra_title["cdd" + str(i + 1) + "_sum"] = df_extra_desc.sum(axis=1)
            concat_list.append(df_extra_desc)

        #concat_list.insert(0, df_input)
        df_output = pd.concat(concat_list, axis=1)

        #reshape the file according to index
        df_output.sort_values(by=["index"], inplace=True)

        #export to file
        out_filename = os.path.join(self.HOME_DIR, "input", "data_ready_" + target_column + ".csv")
        df_output.to_csv(out_filename, index=False)
        print(df_output.shape)

        del df_input
        del df_output

        gc.collect()

#===============================================================================
if __name__ == '__main__':
    dp = DataPreparation()
    processing_steps = [True, True]
    labels = [True, False]

    if labels[0]:
        if processing_steps[0]:
            dp.build_combination(processing_mode=1, out_filename="data_all_conciseness.csv")
        if processing_steps[1]:
            dp.clean_data(target_column="conciseness")
    if labels[1]:
        if processing_steps[0]:
            dp.build_combination(processing_mode=0, out_filename="data_all_clarity.csv")
        if processing_steps[1]:
            dp.clean_data(target_column="clarity")