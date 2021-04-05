import traceback
import json
import re
import os
import time
import pickle
import unicodedata
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity

from flashtext import KeywordProcessor
id_to_ans_kp = KeywordProcessor()

from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])

from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en

stopwords_en = list(stopwords_en)

mapping_dict = {"en":[stopwords_en, spacy_en]}

import config as cf


class AnswerQuestion(object):

	def __init__(self):
		# self.model_name = "norwegian_labour_law"

		# keep_words_list = ['no', 'nor', 'not', 'what', 'which', 'who', 'whom', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below']
		# self.stop_words = [wrd for wrd in stopwords_list if wrd not in keep_words_list]
		pass

	def save_pickle(self, filename, data_to_Save):
		with open(filename + ".pickle", "wb") as pickle_obj:
			pickle.dump(data_to_Save, pickle_obj)

	def load_pickle(self, filename):
		loaded_data = ""
		with open(filename + '.pickle', "rb") as pickle_obj:
			loaded_data = pickle.load(pickle_obj)
		return loaded_data

	def normalize_text(self, text):
		text = unicodedata.normalize("NFKD", text)
		text = re.sub(r"[^a-zA-Z0-9ÅÆÄÖØÜåæäöøüß.]", " ", text)
		text = re.sub(r"\s+", " ", text)
		return text

	# def get_lemma(self, text):
	# 	return " ".join([tok.lemma_.lower().strip() for tok in spacy_en(text) if tok.lemma_ != '-PRON-' and tok.lemma_ not in self.stop_words])

	def get_lemma_tokens(self, text, lang):
		return [tok.lemma_.lower().strip() for tok in mapping_dict[lang][1](text) if tok.lemma_ != '-PRON-' and tok.lemma_ not in mapping_dict[lang][0]]

	# def load_w2v(self):
	# 	self.w2v_data = KeyedVectors.load_word2vec_format('../../Downloads/GoogleNews-vectors-negative300.bin.gz', limit=500000, binary=True)

	def load_word2vec(self, lang):
		st = time.time()
		# self.model = Word2Vec.load('20_newsgroup_word2vec.model')
		# fasttext_path = "../word2vec_models/fasttext_300_" + lang + ".bin"
		# self.model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(fasttext_path, binary=False, limit=100000)
		self.model = KeyedVectors.load_word2vec_format(cf.word2vec_model_path, limit=100000, binary=True)
		print("\n time to load the fasttext model --- ", time.time() - st)


	def creat_ans(self, dic, id_to_dic):
		final_ans = ""

		if dic["tag_name"] == "b" and dic["childs"]:
			for child_id in dic["childs"]:
				final_ans += id_to_dic[child_id]["text"]
		elif dic["tag_name"] == "b" and not dic["childs"]:
			final_ans += dic["text"]
		elif dic["tag_name"] == "p" and dic["parent"] and dic["parent"] != [0]:
			parent_id = dic["parent"][0]
			parent_dic = id_to_dic[parent_id]
			for child_id in parent_dic["childs"]:
				final_ans += id_to_dic[child_id]["text"]

		# elif dic["tag_name"] == "p" and dic["parent"] and dic["parent"] == [0]:

		return re.sub(r"\s+", " ",final_ans)


	def create_and_save_answers(self, sentences_with_id, id_to_dic):
		id_to_ans = defaultdict()
		for dic in sentences_with_id:
			ans = self.creat_ans(dic, id_to_dic)
			if ans:id_to_ans[dic["my_id"]] = [ans]
		
		with open(self.ans_file_name, "w+") as fs:
			fs.write(json.dumps(id_to_ans, indent=4))
		# id_to_ans_kp.add_keywords_from_dict(id_to_ans)
		# self.save_pickle("id_to_ans_kp", id_to_ans_kp)


	def sentence_adder(self, dic):
		clean_text = self.normalize_text(dic["text"])
		self.train_sents.append(clean_text)
		# return {"text":clean_text, "my_id":dic["my_id"], "childs":dic["childs"], "tag_name": dic["tag_name"]}
		return dic


	def make_dir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)
			print("\n directory created for path : ",path)


	def create_bot_structure(self, path, lang, model_name, model_type):
		bot_base_path = "bots"
		self.make_dir(bot_base_path)
		self.make_dir(os.path.join(bot_base_path, model_name))
		self.make_dir(os.path.join(bot_base_path, model_name, lang))
		self.trained_models_dir = os.path.join(bot_base_path, model_name, lang, "trained_models")
		self.make_dir(self.trained_models_dir)
		self.traininig_data_dir = os.path.join(bot_base_path, model_name, lang, "training_data_jsons")
		self.make_dir(self.traininig_data_dir)
		self.extracted_html_dir = os.path.join(bot_base_path, model_name, lang, "extracted_html_jsons")
		self.make_dir(self.extracted_html_dir)

		# if not os.path.exists(bot_base_path):
		# 	os.path.mkdirs(bot_base_path)
		# if not os.path.exists(os.path.join(bot_base_path, "training_data_jsons")):
		# 	os.path.mkdirs(os.path.join(bot_base_path, "training_data_jsons"))
		# if not os.path.exists(os.path.join(bot_base_path, "extracted_html_jsons")):
		# 	os.path.mkdirs(os.path.join(bot_base_path, "extracted_html_jsons"))

		self.ans_file_name = os.path.join(self.traininig_data_dir, "id_to_ans_" + model_name + ".json")
		self.sentences_file_name = os.path.join(self.traininig_data_dir, "sentences_with_id_" + model_name + ".json")
		self.dic_file_name = os.path.join(self.traininig_data_dir, "id_to_dic_" + model_name + ".json")
		print("\n created directory structure ::: ")


	def read_data(self, lang, model_name, path):
		# json_data = json.load(open("extracted_html_jsons/scs_emp_manual.json"))
		# json_data = json.load(open(os.path.join(self.extracted_html_dir, model_name + ".json")))
		json_data = json.load(open(path))

		id_to_dic = defaultdict()
		for dic in json_data:
			id_to_dic[dic["my_id"]] = dic

		self.train_sents = []

		""" for b tags """
		# sentences_with_id = [self.sentence_adder(val_dict) for key, val_dict in id_to_dic.items() if val_dict['tag_name'] == "b" and self.normalize_text(val_dict["text"]) and val_dict["childs"]]

		""" for all tags """
		sentences_with_id = [self.sentence_adder(val_dict) for key, val_dict in id_to_dic.items() if self.normalize_text(val_dict["text"])]

		with open(self.sentences_file_name, "w+") as fs:
			fs.write(json.dumps({"sentences":sentences_with_id}, indent=4))
			print("\n training sentences written successfully in json!")

		with open(self.dic_file_name, "w+") as fs:
			fs.write(json.dumps(id_to_dic, indent=4))
			print("\n id_to_dic written successfully in json!")

		self.create_and_save_answers(sentences_with_id, id_to_dic)

		return self.train_sents

	def train(self, path, lang, model_type, model_name):
		st=time.time()
		self.create_bot_structure(path, lang, model_name, model_type)
		cleaned_sentences = self.read_data(lang, model_name, path)

		import pdb;pdb.set_trace()
		training_corpus = [self.get_lemma_tokens(sent, lang) for sent in cleaned_sentences]
		self.load_word2vec(lang)
		instance_wmd = WmdSimilarity(training_corpus, self.model)
		print("\n training model is done !!!")
		# instance_wmd.save("models/scs_emp_manual_wmd_model")
		model_path = os.path.join(self.trained_models_dir, model_name + ".model")
		del self.model
		instance_wmd.save(model_path)


	# def test(self, query, lang):

	# 	st = time.time()
	# 	print("\n query ==>> ",query)
	# 	query_tokens = self.get_lemma_tokens(self.normalize_text(query), lang)
	# 	print("\n query_tokens ==>> ",query_tokens)
	# 	model_path = os.path.join("models/", self.model_name + ".model")
	# 	instance_wmd = gensim.similarities.docsim.Similarity.load(model_path)
	# 	wmd_sims = instance_wmd[query_tokens]
	# 	wmd_sims = sorted(enumerate(wmd_sims), key=lambda item: -item[1])
	# 	# print("\n wmd_sims --- ",wmd_sims)
		
	# 	training_docs = json.load(open(self.sentences_file_name))["sentences"]
	# 	# id_to_ans_kp = self.load_pickle("id_to_ans_kp") 
		 
	# 	id_to_ans = json.load(open(self.ans_file_name))
	# 	id_to_dic = json.load(open(self.dic_file_name))

	# 	top_answers = 3
	# 	similar_docs = [(score, training_docs[idx]) for idx, score in wmd_sims[:top_answers]]
	# 	# print("\n similar_docs --- ", similar_docs)
	# 	similar_questions = [(tpl[0], tpl[1]['text']) for tpl in similar_docs]
	# 	# print("\n top 5 --- ", similar_questions)
	# 	print("=="*30)
	# 	print("\n top matches --- \n")
	# 	for tpl in similar_questions:
	# 		print(tpl)
	# 	print("=="*30)

	# 	similar_docs = [(tpl[0], id_to_ans[str(tpl[1]["my_id"])]) if str(tpl[1]["my_id"]) in id_to_ans else (tpl[0], id_to_dic[str(tpl[1]["my_id"] + 1)]["text"]) for tpl in similar_docs]
	# 	# id_to_dic
	# 	similar_docs = similar_docs[:1]
	# 	for tpl in similar_docs:
	# 		print("\n 1st Answer => ",tpl[1])
	# 	print("\n total prediction time --- ", time.time() - st)


if __name__ == "__main__":
	obj = AnswerQuestion()
	lang = "nb"
	obj.train(lang)

	### prediction on scs employee manual

	while True:
		query = input("\nEnter the question : ")
		if query == "exit": break
		obj.test(query, lang)
		print("==="*30)


	""" failed queries
	"employees should be comply with what ?"
	"when employees should be ethical and responsible"
	"victimization"
	"rules on integrity"
	"can i expect gifts from our employee"
	"bribery rules"
	"benefits from internal party"
	"abusive behavior from manager"
	"employees are expected to follow whose instructions?"
	"punctuality"
	"punctuality of employee"
	"what things employees are expected to avoid"
	"what if i dont communicate with my collegues"
	"""


	### pred on qnamaker structured
	# vars = ["purpose behind developing bot framework", "version 4 of sdk", "v4", "sdk", "i want to deploy bot on our vm"]
	# query = "which urls need to be approved for our company firewall"
	# obj.test(query, lang)

	### pred on qnamaker semi structured
	# query="create a blob container"
	# obj.test(query, lang)	
