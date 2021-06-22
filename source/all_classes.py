# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-08-20 14:52:25
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-08-23 23:58:10

# Modified by Xingchen Wan on 3 Sep to make the file Python 3 compatible

import os

# import community
import networkx as nx
import numpy as np
import copy

from source.utils import *

ignore_entity_dict = {"ADDITIONAL", "REUTERS", "JOURNAL", "U", "CO.", "ALISTER BULL", "EDITING", "REUTERS ESTIMATES",
                      "REPORTING", "COMPILED BY ASIA COMMODITIES"}


class News:
    def __init__(self, news_dir):
        self.null_news = False
        self.title = ""
        self.dir = news_dir
        self.newsID = -1
        self.date = datetime.date(1929, 10, 12)
        self.date_format = "EDT"  # EDT/EST
        self.time = datetime.time(0, 0, 0)
        self.entity_occur = {}
        self.entity_sentiment = {}
        self.url = ""
        # self.token_sentence = []
        self.load_file(news_dir)
        self.positive_sentiment_sentences = []
        self.positive_sentiment_entities = []
        self.negative_sentiment_sentences = []
        self.negative_sentiment_entities = []

    def load_file(self, news_dir=None):
        if news_dir is None:
            news_dir = self.dir
        if os.path.isfile(news_dir):
            the_news = read_news_file(news_dir)
            if the_news == []:
                self.null_news = True
            else:
                self.title = the_news['title']
                self.date = the_news['date']
                self.time = the_news['time']
                self.url = the_news['url']
                # self.token_sentence = the_news['sentences']

    def extract_token_sentences(self):
        if os.path.isfile(self.dir):
            the_news = read_news_file(self.dir)
            if the_news == []:
                return []
            else:
                return the_news['sentences']
        return []

    def build_entity_sentiment(self, entity_sentiment_list):
        """
        input:
            entity_sentiment_list in one news: [[entity1, sentiment1], [entity2, sentiment2],[entity3, sentiment3]], entity may duplicate, and sentiment = {-1,0,1}
        """
        entity_dict = {}
        for each_list in entity_sentiment_list:
            entity, sentiment = each_list
            sentiment = int(sentiment)
            # # entity = company_name_normalize(entity)
            # if (len(name_mapping_dict) > 0) and (entity in name_mapping_dict):
            # 	entity = company_name_normalize(name_mapping_dict[entity])
            # entity = entity.upper()
            if entity in entity_dict:
                entity_dict[entity][0] += 1
                entity_dict[entity][1] += sentiment
            else:
                entity_dict[entity] = [1, sentiment]
        for entity, num_sent in entity_dict.items():
            avg_sentiment = (num_sent[1] + 0.) / num_sent[0]
            avg_sentiment = round(avg_sentiment, 3)
            self.entity_occur[entity] = num_sent[0]
            self.entity_sentiment[entity] = avg_sentiment


class Day:
    def __init__(self, day_folder, each_date):
        self.null_today = False
        self.date = each_date
        self.day_news = []
        self.news_number = 0
        self.load_folder(day_folder)
        self.entity_occur_day = {}
        self.entity_sentiment_day = {}
        self.day_folder = day_folder

    def load_folder(self, day_folder):
        if day_folder is None:
            day_folder = self.day_folder
        for path, subdirs, files in os.walk(day_folder):
            for name in files:
                the_file = os.path.join(path, name)
                the_news = News(the_file)
                if (the_news.null_news is False) and (the_news.date == self.date):
                    self.day_news.append(the_news)
        if len(self.day_news) == 0:
            self.null_today = True
        self.news_number = len(self.day_news)

    def assign_sentence_to_news(self, sentence_sentiment):
        """
        input:
            sentence_sentiment in one day: [[sentence, entity, sentiment1], [sentence, entity, sentiment1]...], entity may duplicate, and sentiment = {-1,0,1}
            name_mapping_dict: convert entity names
        """
        # load sentence dict
        sentence_dict = {}
        entity_sentiment_for_news = []
        for idx in range(len(self.day_news)):
            entity_sentiment_for_news.append([])
            token_sentences = self.day_news[idx].extract_token_sentences()
            for each_sent in token_sentences:
                sentence_dict[each_sent] = idx

        ### assign entity/sentiment pair to news based on sentence
        count_assigned_sentence = 0
        for sentence, entity, sentiment in sentence_sentiment:
            if sentence in sentence_dict:
                sentence_pos = sentence_dict[sentence]
                entity_sentiment_for_news[sentence_pos].append([entity, sentiment])
                count_assigned_sentence += 1
                if sentiment == -1:
                    # Negative
                    self.day_news[sentence_pos].negative_sentiment_sentences.append(sentence)
                    self.day_news[sentence_pos].negative_sentiment_entities.append(entity)
                elif sentiment == 1:
                    self.day_news[sentence_pos].positive_sentiment_sentences.append(sentence)
                    self.day_news[sentence_pos].positive_sentiment_entities.append(entity)

        # print("Total sentiment sentence: %s, assigned sentence: %s, assign rate: %s"%(len(sentence_sentiment), count_assigned_sentence, (count_assigned_sentence+0.)/len(sentence_sentiment)))
        # print("Total sentence in date: %s is %s"%(self.date.strftime('%Y%m%d'), len(sentence_dict)))
        ### for each news, calculate entity_occur and entity_sentiment
        for idx in range(len(self.day_news)):
            self.day_news[idx].build_entity_sentiment(entity_sentiment_for_news[idx])

        ## calculate entity occur num and avg sentiment within one day
        total_sentiment = {}
        for each_news in self.day_news:
            for entity, num in each_news.entity_occur.items():
                if entity in self.entity_occur_day:
                    self.entity_occur_day[entity] += num
                    total_sentiment[entity] += num * each_news.entity_sentiment[entity]
                else:
                    self.entity_occur_day[entity] = num
                    total_sentiment[entity] = num * each_news.entity_sentiment[entity]
        for entity, num in self.entity_occur_day.items():
            self.entity_sentiment_day[entity] = total_sentiment[entity] / num


class FullData:
    def __init__(self, data_folder, start_date=datetime.date(1929, 10, 12), end_date=datetime.date(2029, 10, 12)):
        self.start_date = start_date
        self.end_date = end_date
        self.entity2id = {}
        self.id2entity = {}
        self.days = []
        self.date2daysID = {}
        self.news_number = 0
        self.data_folder = data_folder
        self.load_full_data(data_folder)
        self.entity_occur_interval = {}
        self.entity_sentiment_interval = {}


    def load_full_data(self, date_folder):
        '''
        input:
			date_folder: dir of the whole data, e.g. "../../ReutersNews106521/"
		'''
        print("Load folder: %s, from [date %s to date %s)" % (
            date_folder, self.start_date.strftime('%Y/%m/%d'), self.end_date.strftime('%Y/%m/%d')))
        date_list = []
        for path, subdirs, files in os.walk(date_folder):
            if files == [".DS_Store"]:
                date_list = subdirs
                break
        if len(date_list) == 0:
            print("No subfolder founded!")
            exit(0)

        count_days = 0
        for each_date_string in date_list:
            each_date = string2date(each_date_string)
            if each_date >= self.start_date and each_date < self.end_date:
                sub_folder_dir = os.path.join(path, each_date_string)
                the_day = Day(sub_folder_dir, each_date)
                self.days.append(the_day)
                self.date2daysID[the_day.date] = len(self.days) - 1
                self.news_number += the_day.news_number
                if count_days % 30 == 0:
                    print("	Loading... loaded days:", count_days)
                count_days += 1
        print("Days number: %s" % (len(self.days)))

    def load_sentiment_result(self, sentiment_in, sentiment_out, sentence_mapping_file, entity_mapping_file):
        # load files
        abbreviation_dict = reform_abbreviate_name_dict(entity_mapping_file)
        sentiment_map_dict = load_sentence_map_file(sentence_mapping_file)
        sentiment_in_lines = open(sentiment_in, 'r').readlines()
        sentiment_out_lines = open(sentiment_out, 'r').readlines()
        line_num = len(sentiment_in_lines)
        assert (line_num == len(sentiment_out_lines))
        # read each line and assign [sentence, entity, sentiment] to each Day
        sentence_entity_sentiment_in_all = []
        for idx in range(len(self.days)):
            sentence_entity_sentiment_in_all.append([])

        current_date = datetime.date(1929, 10, 12)
        for idx in range(line_num):
            line = sentiment_in_lines[idx]
            if "#DATE:data/" in line:
                current_date = string2date(file_name_alphabet2number(line.split(" ")[0][-8:]))
                continue
            else:
                sentence, entity = read_sentiment_in_line(line)
                if current_date >= self.start_date and current_date < self.end_date:
                    if ("(Additional reporting by" not in sentence) and ("(Reporting by" not in sentence) and (
                                "COMPILED BY" not in sentence.upper()):
                        ## convert sentence
                        if sentence in sentiment_map_dict:
                            sentence = sentiment_map_dict[sentence]
                        entity = " ".join(entity)
                        if entity in abbreviation_dict:
                            entity = abbreviation_dict[entity]
                        if " " not in entity:
                            entity = remove_stock_abbre_suffix(entity)
                        entity = entity.upper()
                        if entity == "LEHMAN":
                            entity = "LEHMQ"
                        if (entity not in ignore_entity_dict) and len(entity) > 1 and ("000" not in entity):
                            days_id = self.date2daysID[current_date]
                            sentiment = int(sentiment_out_lines[idx].strip("\n"))
                            sentence_entity_sentiment_in_all[days_id].append([sentence, entity, sentiment])

        ### for each day, calculate the entity/sentiment
        for idx in range(len(self.days)):
            # print self.days[idx].date
            self.days[idx].assign_sentence_to_news(sentence_entity_sentiment_in_all[idx])
        ### extract entity_avg_sentiment for time interval
        total_sentiment = {}
        for each_day in self.days:
            for entity, num in each_day.entity_occur_day.items():
                if entity in self.entity_occur_interval:
                    self.entity_occur_interval[entity] += num
                    total_sentiment[entity] += num * each_day.entity_sentiment_day[entity]
                else:
                    self.entity_occur_interval[entity] = num
                    total_sentiment[entity] = num * each_day.entity_sentiment_day[entity]
        for entity, num in self.entity_occur_interval.items():
            self.entity_sentiment_interval[entity] = total_sentiment[entity] / num

    def show_information(self):
        for each_day in self.days:
            for each_news in each_day.day_news:
                print(each_day.date, each_news.entity_occur, each_news.entity_sentiment)

    def count_entity_occurance_news(self, top=-1):
        # count the number of entity occurs in news. Notice it only count how many news mentioned entity, not count the occurance time
        entity_count_dict = {}
        for each_day in self.days:
            for each_news in each_day.day_news:
                for entity, occ in each_news.entity_occur.items():
                    if entity in entity_count_dict:
                        entity_count_dict[entity] += 1
                    else:
                        entity_count_dict[entity] = 1
        # print_dict(entity_count_dict)
        # print("Total news number:", self.news_number)
        d_view = [(v, k) for k, v in entity_count_dict.items()]
        d_view.sort(reverse=True)  # natively sort tuples by first element
        if top > 0:
            if len(d_view) > top:
                return d_view[:top]
            else:
                return d_view
        else:
            return d_view

    def build_occurrence_network_graph(self, top=-1, focus_iterable=None, weight_threshold=None):
        entity_occ_list = self.count_entity_occurance_news(top)
        entity_list = []
        for num, entity in entity_occ_list:
            if len(focus_iterable) > 0:
                if entity in focus_iterable:
                    entity_list.append(entity)
            else:
                entity_list.append(entity)
        ## entity_occ_list: [[num, entity], [num, entity]]
        entity_num = len(entity_list)
        occurance_array = np.zeros((entity_num, self.news_number))
        sentiment_array = np.zeros((entity_num, self.news_number))
        entity2ID = {}
        ID2entity = {}
        for idx in range(entity_num):
            entity = entity_list[idx]
            entity2ID[entity] = idx
            ID2entity[idx] = entity

        ## build the entity-news matrix
        newsID = 0
        for each_day in self.days:
            for each_news in each_day.day_news:
                overlap_entity_set = set(each_news.entity_occur.keys()) & set(entity_list)
                for each_entity in overlap_entity_set:
                    the_occ = each_news.entity_occur[each_entity]
                    the_sentiment = each_news.entity_sentiment[each_entity]
                    enity_id = entity2ID[each_entity]
                    occurance_array[enity_id, newsID] = the_occ
                    sentiment_array[enity_id, newsID] = the_sentiment
                newsID += 1
        print("Occurrence full shape:", occurance_array.shape)
        # print occurance_array
        # print sentiment_array
        # print np.count_nonzero(occurance_array)
        # print np.count_nonzero(sentiment_array)

        ## calculate degree/centrality/community
        # build networkX graph
        occ_edge_list = []

        for s_id in range(entity_num - 1):
            for t_id in range(s_id + 1, entity_num):
                weight = cosine_distance(occurance_array[s_id], occurance_array[t_id])
                if weight > 0:
                    s_entity = ID2entity[s_id]
                    t_entity = ID2entity[t_id]
                    occ_edge_list.append((s_entity, t_entity, weight))
        occ_graph = nx.Graph()
        occ_graph.add_weighted_edges_from(occ_edge_list)
        return occ_graph

    def get_group_avg_sentiment(self, start_date, end_date, entity_list):
        if start_date < self.start_date or end_date >= self.end_date:
            print("Error: Date exceed the interval!")
            exit(0)

        if len(self.entity_occur_interval) == 0:
            print("Error: Empty entity sentiment information in interval! Please run load_sentiment_result first!")
            exit(0)
        ## convert to list if entity_list is dict
        if isinstance(entity_list, dict):
            entity_list = entity_list.keys()

        ### extract the entity/sentiment in focus date interval
        focus_total_sentiment = {}
        focus_entity_occur_interval = {}
        focus_entity_sentiment_interval = {}
        for single_date in daterange(start_date, end_date):
            if single_date not in self.date2daysID:  ## 2013/08/18 does not have news
                continue
            the_day_idx = self.date2daysID[single_date]
            each_day = self.days[the_day_idx]
            for entity, num in each_day.entity_occur_day.items():
                if entity in focus_entity_occur_interval:
                    focus_entity_occur_interval[entity] += num
                    focus_total_sentiment[entity] += num * each_day.entity_sentiment_day[entity]
                else:
                    focus_entity_occur_interval[entity] = num
                    focus_total_sentiment[entity] = num * each_day.entity_sentiment_day[entity]
        for entity, num in focus_entity_occur_interval.items():
            focus_entity_sentiment_interval[entity] = focus_total_sentiment[entity] / num

        ## calculate group sentiment
        total_sentiment = 0
        total_number = 0
        for entity in entity_list:
            if entity not in focus_entity_occur_interval:
                print("Error: Cannot get entity sentiment information! Entity:", entity)
                continue
            else:
                total_sentiment += focus_entity_occur_interval[entity] * focus_entity_sentiment_interval[entity]
                total_number += focus_entity_occur_interval[entity]
        if total_number == 0:
            return -2, 0
        return total_sentiment / total_number, total_number


def get_coexisting_entity(start_date, month_interval, top):
    sentiment_in = "../data/2006-2013.content.filter.senti.in"
    sentiment_out = "../data/2006-2013.content.filter.senti.out"
    sentence_map = "../data/2006-2013.content.ner.out.filter.map"
    abbreviation_file = "../data/abbreviation_pairs.pkl"
    all_dict = []
    for idx in range(30):
        end_date = add_months(start_date, month_interval)
        if end_date > datetime.date(2013, 10, 20):
            break
        full_date = FullData("../../ReutersNews106521/", start_date, end_date)
        full_date.load_sentiment_result(sentiment_in, sentiment_out, sentence_map, abbreviation_file)
        current_list = full_date.count_entity_occurance_news(top)
        current_dict = {}
        current_set = set()
        for num, ent in current_list:
            current_dict[ent] = num
            current_set.add(ent)

        if idx == 0:
            entity_set = current_set
        else:
            entity_set = entity_set & current_set
        all_dict.append(current_dict)
        start_date = end_date
        print("current accumulate entity set size: ", len(entity_set))
    print(entity_set)
    print(len(entity_set))
    entity_interval_occ = {}
    for entity in entity_set:
        entity_interval_occ[entity] = []
        for each_dict in all_dict:
            entity_interval_occ[entity].append(each_dict[entity])
    print(entity_interval_occ)
    print(len(entity_interval_occ))
    fout = open("coexisting_entity." + str(month_interval) + ".txt", 'w')
    for k, v in entity_interval_occ.items():
        fout.write(k + ": " + ' '.join(str(e) for e in v) + "\n")


if __name__ == '__main__':
    sentiment_in = "../data/2006-2013.content.filter.senti.in"
    sentiment_out = "../data/2006-2013.content.filter.senti.out"
    sentence_map = "../data/2006-2013.content.ner.out.filter.map"
    abbreviation_file = "../data/abbreviation_pairs.pkl"
    all_data = FullData("../data/ReutersNews106521/", datetime.date(2013, 1, 1), datetime.date(2013, 3, 31))
    all_data.load_sentiment_result(sentiment_in, sentiment_out, sentence_map, abbreviation_file)

    # the_day = Day("../../ReutersNews106521/20061020/")
    # print the_day.date
    # print len(the_day.day_news)
    # full_date = FullData("../../ReutersNews106521/", datetime.date(2007, 1, 1), datetime.date(2007, 4, 1))

    # get_coexisting_entity(datetime.date(2007,1,1), 3, 1000)
    # all_data.load_sentiment_result(sentiment_in, sentiment_out, sentence_map, abbreviation_file)
    # focus_dict = load_focus_dict("coexisting_entity.3month_clean.sector.manual.txt")
    # all_data.build_occurence_network(-1, focus_dict)
    # print full_date.count_entity_occurance_news(200)
