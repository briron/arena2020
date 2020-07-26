import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from gensim.models import Word2Vec
import heapq
import pickle

from google.colab import drive
drive.mount('/content/gdrive')


import io
import os
import distutils.dir_util
from collections import Counter
from tqdm import tqdm

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import re
from khaiii import KhaiiiApi
from collections import Counter
from typing import *


def re_sub(series: pd.Series) -> pd.Series:
    series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)
    series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)
    series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True) 
    series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True) 
    return series


def flatten(list_of_list : List) -> List:
    flatten = [j for i in list_of_list for j in i]
    return flatten

def get_token(title: str, tokenizer)-> List[Tuple]:
    
    if len(title)== 0 or title== ' ':  # 제목이 공백인 경우 tokenizer에러 발생
        return []
    
    result = tokenizer.analyze(title)
    result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
    return result

def get_all_tags(df) -> List:
    tag_list = df['tags'].values.tolist()
    tag_list = flatten(tag_list)
    return tag_list

def write_json(data, fname):
  def _conv(o):
    if isinstance(o, (np.int64, np.int32)):
      return int(o)
    raise TypeError
  with io.open(BASE_DIR + "res/" + fname, "w", encoding="utf-8") as f:
    json_str = json.dumps(data, ensure_ascii=False, default=_conv)
    f.write(json_str)


BASE_DIR = './'
song_meta_fname = "song_meta.json"
train_fname = "train.json"
test_fname = "test.json"

class W2VModel:
  def __init__(self, train_data, col):
    if type(train_data) != type(pd.DataFrame()):
      self.model = train_data
      self.col = col
      return
    self.col = col
    corpus = train_data[self.col].apply(lambda x : [ str(item) for item in x ])
    self.model = Word2Vec(corpus[:], min_count=3, size=100, window=5, sg=1)

  def recommand(self, test_data, playlist_id, bucket_num):
    base_tag = test_data[test_data.id == playlist_id][self.col]
    search_tag = []
    for t in base_tag.values[0]:
      if t in self.model.wv.vocab:
        search_tag.append(t)
    if len(search_tag) == 0:
      return []
    si = self.model.wv.most_similar(positive=search_tag, topn=bucket_num)
    return [ item[0] for item in si ]

def dictToProb(d):
  for song, freq in d.items():
    total = 0
    for count in freq.values():
      total += count
    for key in freq.keys():
      d[song][key] /= total
  return d

def sortDict(source):
  sortedDict = {}
  for i, innerDict in source.items():
    l = []
    for k, val in innerDict.items():
      l.append([k, val])
    l.sort(key = lambda x : x[1], reverse=True)
    sortedDict[i] = l
  return sortedDict

def getFreqDict(train_data):
  tag_freq_by_song = {} # 각 song 별 tag freq 을 저장
  song_freq_by_tag = {} # 각 tag 별 song freq 을 저장
  for i, row in train_data[['tags','songs']].iterrows():
    for tag in row['tags']:
      for song in row['songs']:
        if song not in tag_freq_by_song:
          tag_freq_by_song[song] = {}
        if tag not in tag_freq_by_song[song]:
          tag_freq_by_song[song][tag] = 0
        if tag not in song_freq_by_tag:
          song_freq_by_tag[tag] = {}
        if song not in song_freq_by_tag[tag]:
          song_freq_by_tag[tag][song] = 0
        tag_freq_by_song[song][tag] += 1
        song_freq_by_tag[tag][song] += 1
  tag_freq_by_song = sortDict(dictToProb(tag_freq_by_song))
  song_freq_by_tag = sortDict(dictToProb(song_freq_by_tag))
  return tag_freq_by_song, song_freq_by_tag


def getTopItem(itemList, bucket):
  return [ itemList[i][0] for i in range(min(len(itemList), bucket)) ] 

def getSongByTagFreq(song_freq_by_tag, tags, bucket = 100):
  freq_song = {}
  for tag in tags:
    if tag not in song_freq_by_tag:
      continue
    for i, song in enumerate(getTopItem(song_freq_by_tag[tag], bucket * 2)):
      if song not in freq_song:
        freq_song[song] = 0.0
      freq_song[song] += song_freq_by_tag[tag][i][1]
  return heapq.nlargest(bucket, freq_song, key=freq_song.get)

def getTagBySongFreq(tag_freq_by_song, songs, bucket = 10):
  freq_tag = {}
  for song in songs:
    if song not in tag_freq_by_song:
      continue
    for i, tag in enumerate(getTopItem(tag_freq_by_song[song], bucket * 2)):
      if tag not in freq_tag:
        freq_tag[tag] = 0.0
      freq_tag[tag] += tag_freq_by_song[song][i][1]
  return heapq.nlargest(bucket, freq_tag, key=freq_tag.get)


def getPopularDict(train_data):
  popular_song_by_year = {}
  popular_tag_by_year = {}
  for i, row in train_data[['tags','songs','updt_year']].iterrows():
    year = row['updt_year']
    for tag in row['tags']:
      for song in row['songs']:
        if year not in popular_song_by_year:
          popular_song_by_year[year] = {}
        if year not in popular_tag_by_year:
          popular_tag_by_year[year] = {}
        if song not in popular_song_by_year[year]:
          popular_song_by_year[year][song] = 0
        if tag not in popular_tag_by_year[year]:
          popular_tag_by_year[year][tag] = 0
        popular_song_by_year[year][song] += 1
        popular_tag_by_year[year][tag] += 1
  popular_song_by_year = sortDict(popular_song_by_year)
  popular_tag_by_year = sortDict(popular_tag_by_year)
  return popular_song_by_year, popular_tag_by_year

def getSongByYear(popular_song_by_year, year, bucket = 100):
  return getTopItem(popular_song_by_year[year], bucket)

def getTagByYear(popular_tag_by_year, year, bucket = 10):
  return getTopItem(popular_tag_by_year[year], bucket)


def fillAnswer(left, items, bucket, id_dict, id, updt_dict = None, year = None):
  count = 0
  for item in left:
    if updt_dict != None and updt_dict[str(item)]['issue_year'] > year:
      continue
    if item not in items and item not in id_dict[id]:
      items.append(item)
      count += 1
    if len(items) >= bucket:
      return count
  return count

def getTotalDict(train_data, test_data):
  song_dic = {}
  tag_dic = {}
  title_dic = {}
  total = []
  for i, q in tqdm(pd.concat([train_data, test_data]).iterrows()):
    items = []
    song_dic[str(q['id'])] = []
    tag_dic[str(q['id'])] = []
    title_dic[str(q['id'])] = []
    if type(q['songs']) == list:
      song_dic[str(q['id'])] = q['songs']
      items += list(map(str,q['songs']))
    if type(q['tags']) == list: 
      tag_dic[str(q['id'])] = q['tags']
      items += q['tags']
    if type(q['ply_token']) == list: 
      title_dic[str(q['id'])] = q['ply_token']
      items += q['ply_token']
    total.append(items)
  total = [x for x in total if len(x)>1]
  return total, song_dic, tag_dic, title_dic

def updateP2V(train_data, test_data, w2v_model, p2v_model, song_dict, tag_dict, title_dict):
  ID = []   
  vec = []
  for i, q in tqdm(pd.concat([train_data, test_data]).iterrows()):
    tmp_vec = 0
    id = str(q['id'])
    if len(song_dict[id]) >= 1 or len(tag_dict[id]) >= 1 or len(title_dict[id]) >= 1:
      items = []
      for item in song_dict[id] + tag_dict[id] + title_dict[id]:
        try: 
          tmp_vec += w2v_model.wv.get_vector(str(item))
        except KeyError:
          pass
    if type(tmp_vec) != int:
      ID.append(id)    
      vec.append(tmp_vec)
  p2v_model.add(ID, vec)

def getItemById(most_id_list, item_dict, bucket):
  items = []
  for ID in most_id_list:
    items += item_dict[ID]
  items = list(pd.value_counts(items)[:bucket].index)
  return items

def getUpdtDict(song_meta_data):
  updt_dict = song_meta_data[['issue_year', 'id']].set_index('id').transpose().to_dict()
  return updt_dict

def tokenize(train_data, test_data):
  tokenizer = KhaiiiApi()
  all_tag = get_all_tags(pd.concat([train_data,test_data]))
  token_tag = [get_token(x, tokenizer) for x in all_tag]
  token_itself = list(filter(lambda x: len(x)==1, token_tag))
  token_itself = flatten(token_itself)
  flatten_token = flatten(token_tag)

  train_data['plylst_title'] = re_sub(train_data['plylst_title'])
  train_data.loc[:, 'ply_token'] = train_data['plylst_title'].map(lambda x: get_token(x, tokenizer))

  test_data['plylst_title'] = re_sub(test_data['plylst_title'])
  test_data.loc[:, 'ply_token'] = test_data['plylst_title'].map(lambda x: get_token(x, tokenizer))

  using_pos = ['NNG','SL','NNP','MAG','SN']
  train_data['ply_token'] = train_data['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
  test_data['ply_token'] = test_data['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
  unique_tag = set(token_itself)
  unique_word = [x[0] for x in unique_tag]
  train_data['ply_token'] = train_data['ply_token'].map(lambda x: list(filter(lambda x: x[0][0] in unique_word, x)))
  test_data['ply_token'] = test_data['ply_token'].map(lambda x: list(filter(lambda x: x[0][0] in unique_word, x)))
  train_data['ply_token'] = train_data['ply_token'].apply(lambda x: [i[0] for i in x])
  test_data['ply_token'] = test_data['ply_token'].apply(lambda x: [i[0] for i in x])

def run(song_meta_data, train_data, test_data):  
  train_data['updt_year'] = train_data['updt_date'].str.slice(start=0, stop=4)
  test_data['updt_year'] = test_data['updt_date'].str.slice(start=0, stop=4)
  song_meta_data['issue_year'] = song_meta_data['issue_date'].str.slice(start=0, stop=4)
  song_meta_data['id'] = song_meta_data['id'].astype(str)

  print("Tokenize...")
  tokenize(train_data, test_data)


  train_data = train_data.sort_values(by='updt_date').reset_index(drop=True)
  test_data = test_data.sort_values(by='updt_date').reset_index(drop=True)
  print("Total Dict Loading")
  if os.path.exists(BASE_DIR + 'model/total_data_final.pickle') and os.path.exists(BASE_DIR + 'model/song_dict_final.pickle') and os.path.exists(BASE_DIR + 'model/tag_dict_final.pickle') and os.path.exists(BASE_DIR + 'model/title_dict_final.pickle'):
    with open(BASE_DIR + 'model/total_data_final.pickle', 'rb') as handle:
      total_data = pickle.load(handle)
    with open(BASE_DIR + 'model/song_dict_final.pickle', 'rb') as handle:
      song_dict = pickle.load(handle)
    with open(BASE_DIR + 'model/tag_dict_final.pickle', 'rb') as handle:
      tag_dict = pickle.load(handle)
    with open(BASE_DIR + 'model/title_dict_final.pickle', 'rb') as handle:
      title_dict = pickle.load(handle)
  else:
    print("Total Dict Not Existing... Calculating")
    total_data, song_dict, tag_dict, title_dict = getTotalDict(train_data, test_data)
    with open(BASE_DIR + 'model/total_data_final.pickle', 'wb') as handle:
      pickle.dump(total_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(BASE_DIR + 'model/song_dict_final.pickle', 'wb') as handle:
      pickle.dump(song_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(BASE_DIR + 'model/tag_dict_final.pickle', 'wb') as handle:
      pickle.dump(tag_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(BASE_DIR + 'model/title_dict_final.pickle', 'wb') as handle:
      pickle.dump(title_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print("Frequency Loading...")
  if os.path.exists(BASE_DIR + 'model/tag_freq_by_song.pickle') and os.path.exists(BASE_DIR + 'model/song_freq_by_tag.pickle'):
    with open(BASE_DIR + 'model/tag_freq_by_song.pickle', 'rb') as handle:
      tag_freq_by_song = pickle.load(handle)
    with open(BASE_DIR + 'model/song_freq_by_tag.pickle', 'rb') as handle:
      song_freq_by_tag = pickle.load(handle)
  else:
    print("Frequency Not Existing... Calculating")
    tag_freq_by_song, song_freq_by_tag = getFreqDict(train_data)
    with open(BASE_DIR + 'model/tag_freq_by_song.pickle', 'wb') as handle:
      pickle.dump(tag_freq_by_song, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(BASE_DIR + 'model/song_freq_by_tag.pickle', 'wb') as handle:
      pickle.dump(song_freq_by_tag, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print("Update Date Loading...")
  if os.path.exists(BASE_DIR + 'model/updt_dict.pickle'):
    with open(BASE_DIR + 'model/updt_dict.pickle', 'rb') as handle:
      updt_dict = pickle.load(handle)
  else:
    print("Update Date Not Existing... Calculating")
    updt_dict = getUpdtDict(song_meta_data)
    with open(BASE_DIR + 'model/updt_dict.pickle', 'wb') as handle:
      pickle.dump(updt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print("Song Popularity Loading...")
  if os.path.exists(BASE_DIR + 'model/popular_song_by_year.pickle') and os.path.exists(BASE_DIR + 'model/popular_tag_by_year.pickle'):
    with open(BASE_DIR + 'model/popular_tag_by_year.pickle', 'rb') as handle:
      popular_tag_by_year = pickle.load(handle)
    with open(BASE_DIR + 'model/popular_song_by_year.pickle', 'rb') as handle:
      popular_song_by_year = pickle.load(handle)
  else:
    print("Song Popularity Not Existing... Calculating")
    popular_song_by_year, popular_tag_by_year = getPopularDict(train_data)
    with open(BASE_DIR + 'model/popular_tag_by_year.pickle', 'wb') as handle:
      pickle.dump(popular_tag_by_year, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(BASE_DIR + 'model/popular_song_by_year.pickle', 'wb') as handle:
      pickle.dump(popular_song_by_year, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print("Word2Vec Model Loading...")
  if os.path.exists(BASE_DIR + 'model/w2v_model_sg_title.model'):
    w2v_model = Word2Vec.load(BASE_DIR + 'model/w2v_model_sg_title.model')
  else:
    print("Word2Vec Model Not Found !")
    print("Training...")
    w2v_model = Word2Vec(total_data, min_count = 3, size = 100, window = 210, sg = 1)
    w2v_model.save(BASE_DIR + 'model/w2v_model_sg_title.model')

  print("Training...")
  p2v_model = WordEmbeddingsKeyedVectors(100)
  updateP2V(train_data, test_data, w2v_model, p2v_model, song_dict, tag_dict, title_dict)

  print("Word2Vec Second Model Loading...")
  if os.path.exists(BASE_DIR + 'model/w2v_tag_final.model') and os.path.exists(BASE_DIR + 'model/w2v_song_final.model'):
    tag_model = Word2Vec.load(BASE_DIR + 'model/w2v_tag_final.model')
    song_model = Word2Vec.load(BASE_DIR + 'model/w2v_song_final.model')
    mt = W2VModel(tag_model, "tags")
    ms = W2VModel(song_model, "songs")
  else:
    print("Word2Vec Second Model Not Found !")
    print("Tag Training...")
    mt = W2VModel(pd.concat([train_data, test_data]), "tags")
    mt.model.save(BASE_DIR + 'model/w2v_tag_final.model')
    print("Song Training...")
    ms = W2VModel(pd.concat([train_data, test_data]), "songs")
    ms.model.save(BASE_DIR + 'model/w2v_song_final.model')



  print("start")
  answer = []
  for i, row in tqdm(test_data.iterrows()):
    year = str(row['updt_year'])
    id = str(row['id'])
    songs = []
    tags = []
    try:
      most_id_list = [x[0] for x in p2v_model.most_similar(id, topn=200)]
      fillAnswer(getItemById(most_id_list, song_dict, 200), songs, 100, song_dict, id, updt_dict, year)
      fillAnswer(getItemById(most_id_list, tag_dict, 20), tags, 10, tag_dict, id)
    except:
      pass

    if len(songs) < 100:
      fillAnswer(ms.recommand(test_data, int(row['id']), 200), songs, 100, song_dict, id, updt_dict, year)

    if len(tags) < 10:
      fillAnswer(mt.recommand(test_data, int(row['id']), 20), tags, 10, tag_dict, id)

    if len(songs) < 100:
      fillAnswer(getSongByTagFreq(song_freq_by_tag, row['tags'], 200), songs, 100, song_dict, id, updt_dict, year)
    if len(tags) < 10:
      fillAnswer(getTagBySongFreq(tag_freq_by_song, row['songs'], 20), tags, 10, tag_dict, id)


    if len(songs) < 100:
      fillAnswer(getSongByYear(popular_song_by_year, year, 200), songs, 100, song_dict, id, updt_dict, year)
    if len(tags) < 10:
      fillAnswer(getTagByYear(popular_tag_by_year, year, 20), tags, 10, tag_dict, id)

    if len(songs) < 100:
      try:
        fillAnswer(getSongByYear(popular_song_by_year, str(int(year)-1), 20), songs, 100, song_dict, id, updt_dict, year)
      except:
        fillAnswer(getSongByYear(popular_song_by_year, str(int(year)+1), 200), songs, 100, song_dict, id, updt_dict, year)
    if len(tags) < 10:
      try:
        fillAnswer(getTagByYear(popular_tag_by_year, str(int(year)-1), 20), tags, 10, tag_dict, id)
      except:
        fillAnswer(getTagByYear(popular_tag_by_year, str(int(year)+1), 200), tags, 10, tag_dict, id)

    if len(songs) < 100:
      print("song 의 개수가 적습니다. id : ", str(row['id']), str(year))
    if len(tags) < 10:
      print("tag 의 개수가 적습니다. id : ", str(row['id']), str(year))


    answer.append({
      "id": row["id"],
      "songs": songs,
      "tags": tags
    })

  write_json(answer, "results.json")



print("Data Loading...")
with open(BASE_DIR + 'song_meta.json', 'r') as lf:
    song_meta_raw = json.loads(lf.read())
song_meta_data = pd.DataFrame(song_meta_raw)

with open(BASE_DIR + train_fname, 'r') as lf:
  train_raw = json.loads(lf.read())
train_data = pd.DataFrame(train_raw)

with open(BASE_DIR + test_fname, 'r') as lf:
  test_raw = json.loads(lf.read())
test_data = pd.DataFrame(test_raw)

print("Data Loaded !!")


run(song_meta_data, train_data, test_data)