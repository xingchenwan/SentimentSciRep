# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-08-20 14:52:25
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-08-23 14:51:54

# Modified by Xingchen Wan on 3 Sep to make the file Python 3 compatible

import datetime
import nltk
import re
from os.path import isfile, join
import numpy as np
from numpy.linalg import norm
# import cPickle as pickle
import pickle
import calendar

month_to_num = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
                "Nov": 11, "Dec": 12}


def read_news_file(news_file):
    fins = open(news_file, 'r').readlines()
    news = {}
    line_num = len(fins)
    content = ""
    if line_num < 1:
        return []
    for idx in range(line_num):
        if idx == 0:
            news["title"] = [re.sub('\d', '0', fins[idx].strip('-- \n'))]
        elif idx == 1:
            continue
        elif idx == 2:
            # print type(fins[idx])
            alltimes = fins[idx].strip('-- \n').split()  ## 'Mon Oct 30, 2006 1:55pm EST'.split()
            assert (len(alltimes) == 6)
            if alltimes[1] not in month_to_num:
                print("Month convert error!", "Month string: ", alltimes[1], " at file:", news_file)
                exit(0)
            month = month_to_num[alltimes[1]]
            day = int(alltimes[2].strip(','))
            year = int(alltimes[3])
            news['date'] = datetime.date(year, month, day)
            hour = int(alltimes[4].split(":")[0])
            if alltimes[4][-2:] == "pm":
                if hour < 12:
                    hour += 12
            minute = int(alltimes[4].split(":")[1][:-2])
            news['time'] = datetime.time(hour, minute)

        elif idx == 3:
            news["url"] = fins[idx].strip('-- \n')
        else:
            each_line = fins[idx].strip('-- \n')
            if "(Reuters) -" in each_line:
                each_line = each_line.split("(Reuters) -")[-1]
            if len(each_line) > 0:
                content += each_line + " "
    origin_content = nltk.sent_tokenize(re.sub('\d', '0', content))
    sentence = news['title'] + origin_content
    token_sentences = []
    for sent in sentence:
        new_sent = " ".join(nltk.word_tokenize(sent))
        token_sentences.append(new_sent)
    news["sentences"] = token_sentences
    return news


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def cosine_distance(np_a, np_b):
    # print np_a, np_b
    cos_sim = np.inner(np_a, np_b) / (norm(np_a) * norm(np_b))
    return cos_sim


def print_dict(input_dict):
    d_view = [(v, k) for k, v in input_dict.iteritems()]
    d_view.sort(reverse=True)  # natively sort tuples by first element
    for v, k in d_view:
        print("%s: %d" % (k, v))


def file_name_alphabet2number(input_string):
    convert_dict = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5", "G": "6", "H": "7", "J": "8", "K": "9"}
    # convert_dict = {"0":"A", "1":"B", "2":"C", "3":"D", "4":"E", "5":"F", "6":"G", "7":"H", "8":"J", "9":"K", }

    if len(input_string) < 8:
        print("wrong name:", input_string)
        exit(0)
    out = input_string[0:-8]
    for c in input_string[-8:]:
        out += convert_dict[c]
    return out


def load_company_mapping(company_name_file):
    print("Load company mapping:", company_name_file)
    name_mapping = {}
    fins = open(company_name_file, 'r').readlines()
    for line in fins:
        pair = line.strip('\n').split(': ', 1)
        name_mapping[pair[0]] = pair[1]
    print("Company mapping size:", len(name_mapping))
    return name_mapping


def string2date(input_string):
    assert (len(input_string) == 8)
    year = int(input_string[:4])
    month = int(input_string[4:6])
    day = int(input_string[6:])
    return datetime.date(year, month, day)


def company_name_normalize(input_company_name):
    input_company_name = sepecial_stock_token(input_company_name).upper()

    if ", INC" in input_company_name:
        input_company_name = input_company_name.split(', INC')[0]
    elif " INC" in input_company_name:
        input_company_name = input_company_name.split(' INC')[0]
    if " 'S" in input_company_name:
        input_company_name = input_company_name.split(" 'S")[0]
    if " CORP." in input_company_name:
        input_company_name = input_company_name.split(" CORP.")[0]
    elif " Corp" == input_company_name[-5:]:
        input_company_name = input_company_name[:-5]
    elif " CORPORATION" in input_company_name:
        input_company_name = input_company_name.split(" CORPORATION")[0]
    elif " CO." in input_company_name:
        input_company_name = input_company_name.split(" CO.")[0]
    elif input_company_name[-3:] == " CO":
        input_company_name = input_company_name[:-3]
    elif " COS." in input_company_name:
        input_company_name = input_company_name.split(" COS.")[0]
    if " COMPANY" in input_company_name:
        input_company_name = input_company_name.rsplit(" COMPANY", 1)[0]
    elif " GROUP" in input_company_name:
        input_company_name = input_company_name.rsplit(" GROUP", 1)[0]
    if "&AMP;" in input_company_name:
        input_company_name = input_company_name.replace("&AMP;", "&")
    if " P.L.C." == input_company_name[-7:]:
        input_company_name = input_company_name[:-7]
    elif " PLC" == input_company_name[-4:]:
        input_company_name = input_company_name[:-4]
    elif " LIMITED" == input_company_name[-8:]:
        input_company_name = input_company_name[:-8]
    elif " LTD" == input_company_name[-4:]:
        input_company_name = input_company_name[:-4]
    elif " LTD/PLC" == input_company_name[-8:]:
        input_company_name = input_company_name[:-8]

    if " &" == input_company_name[-2:]:
        input_company_name = input_company_name[:-2]
    elif " & CO" == input_company_name[-5:]:
        input_company_name = input_company_name[:-5]
    elif " LP" == input_company_name[-3:]:
        input_company_name = input_company_name[:-3]
    elif " LLC" == input_company_name[-4:]:
        input_company_name = input_company_name[:-4]
    elif " PLC" == input_company_name[-4:]:
        input_company_name = input_company_name[:-4]
    elif " LP" == input_company_name[-3:]:
        input_company_name = input_company_name[:-3]
    if " AND " in input_company_name:
        input_company_name = input_company_name.replace(" AND ", " & ")
    return input_company_name


def sepecial_stock_token(input_token):
    if len(input_token) < 4:
        return input_token
    elif input_token[-1].isupper() and (input_token[-2] == ".") and (input_token[:-2].isupper()):
        return input_token[:-2]
    elif input_token[-1].isupper() and (input_token[-2] == "^") and (input_token[:-2].isupper()):
        return input_token[:-2]
    return input_token


def reform_abbreviate_name_dict(input_pairs_pkl):
    abbre_dict = pickle.load(open(input_pairs_pkl, 'rb'), )
    long2short_dict = {}
    print("Total abbreviate pair num:", len(abbre_dict))
    conflict_num = 0
    for K, V in abbre_dict.items():
        if K in ["Reporting", "Additional", "Writing", "Editing"] or ("000" in K) or (')' in K):
            continue
        K = remove_stock_abbre_suffix(K)
        for v in V:
            if v in long2short_dict:
                if long2short_dict[v] != K:
                    conflict_num += 1
                # print v, ";",long2short_dict[v],";", K
            long2short_dict[v] = K
    print("Reformed abbreviate dict size:", len(long2short_dict), "conflict_num:", conflict_num)
    return long2short_dict


def remove_stock_abbre_suffix(input_string):
    if input_string.isupper():
        if "." in input_string:
            return input_string.split('.')[0]
        else:
            return input_string
    return input_string.upper()


def load_sentence_map_file(sent_map_file):
    fins = open(sent_map_file, 'r').readlines()
    new = ""
    old = ""
    map_dict = {}
    for idx in range(len(fins)):
        if (idx + 1) % 3 == 1:
            new = fins[idx].strip('\n')
        elif (idx + 1) % 3 == 2:
            old = fins[idx].strip('\n')
        else:
            map_dict[new] = old
    print("Sentence map size:", len(map_dict))
    return map_dict


def read_sentiment_in_line(line):
    """
        input: line "Wells Fargo to pay Citi $ 000 million over Wachovia ||| 0 1 ||| 0"
        output: sentence, entity
    """
    splits = line.split('|||')
    sent = splits[0].strip(" ")
    [entity_start, entity_end] = splits[1].strip(" ").split()
    entity = sent.split(" ")[int(entity_start):(int(entity_end) + 1)]
    return sent, entity


def load_focus_dict(focus_file):
    fins = open(focus_file, 'r').readlines()
    focus_dict = {}
    for line in fins:
        entity = line.split(":")[0]
        full_name = line.split(":")[1]
        if entity not in focus_dict:
            focus_dict[entity] = full_name
    return focus_dict


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def community_to_dicts(community_file):
    fins = open(community_file, 'r').readlines()
    entity2id = {}
    id2community = {}
    id2noncommunity = {}
    all_entity_set = set()
    for line in fins:
        line = line.strip("\n")
        pair = line.split(": ")
        community_id = pair[0]
        entity_list = pair[1].split(",")
        for entity in entity_list:
            entity2id[entity] = community_id
        id2community[community_id] = entity_list
        all_entity_set = all_entity_set | set(entity_list)
    for c_id, community in id2community.items():
        noncommunity = list(all_entity_set - set(community))
        id2noncommunity[c_id] = noncommunity
    return entity2id, id2community, id2noncommunity


if __name__ == '__main__':
    news = read_news_file("../data/ReutersNews106521/ReutersNews106521/20061030/idUSL3090279820061030")

    for k, v in news.items():
        print(k, v)

