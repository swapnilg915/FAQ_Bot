# -*- coding: utf-8 -*-

import json
import os
import time
import re
from datetime import datetime
import traceback
import csv
import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, render_template, flash, redirect
from werkzeug.utils import secure_filename

from excel_bot.excel_worker import ExcelSemanticSimilarity
excel_worker_obj = ExcelSemanticSimilarity()

from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()

UPLOAD_FOLDER = 'file_uploads'
if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['csv', 'xlsx','pdf', 'docx'])
training_info = json.load(open("training_info.json"))

app = Flask(__name__)
app.debug = True
app.url_map.strict_slashes = False
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict')
def home():
	return render_template('home.html')


@app.route('/upload')
def upload_train():
	return render_template('upload_new.html')


@app.route('/upload', methods = ['GET', 'POST'])
def train():
	if request.method == 'POST':
		# check if the post request has the file part
		
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']
		required_values = ["model_name", "lang"]
		input_dict = {}
		for key in required_values:
			input_dict[key] = ''
			input_dict[key] = request.values[key]

		now = datetime.now()
		input_dict["model_name"] = "_".join(input_dict["model_name"].split())
		training_info.append({"filename":str(file.filename), "timestamp": str(now), "model_name": input_dict["model_name"]})

		with open("training_info.json", "w+") as fs:
			fs.write(json.dumps(training_info, indent=4))
			print("\n training info written in json successfully !!! ")
		
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)

		if file and allowed_file(file.filename):

			filename = secure_filename(file.filename)
			print("\n === ", app.config['UPLOAD_FOLDER'])
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			input_dict["file_path"] = file_path
			file.save(file_path)
			flash('File successfully uploaded')

			### decide pdf / excel
			train_excel(input_dict)
			
			return redirect('/upload')
			# return redirect(request.url)
		else:
			flash('Allowed file types are : csv, xlsx, pdf, docx')
			return redirect(request.url)


def train_excel(input_dict):
	"""
	train bot on excel file
	"""
	excel_worker_obj.train(input_dict)



@app.route('/predict', methods=['POST', 'GET'])
def main():
	try:
		st = time.time()
		required_keys = ['model_type', 'model_name', 'query', 'lang']
		input_dict = {}
		for key in required_keys:
			input_dict[key] = ""
			input_dict[key] = request.values[key]

		orig_query = input_dict["query"]
		input_dict["model_name"] = "_".join(input_dict["model_name"].split())
		
		if not input_dict["model_type"] or not input_dict["lang"] or not input_dict["query"]:
			top_answers = [(0.0, ["Please Enter the all required inputs !!! "])]			
		else:
			print("\n api input --- ", input_dict["query"], input_dict["lang"])
			query_token = cleaning_pipeline_obj.cleaning_pipeline(input_dict)

			bot_base_path = "bots"
			models_path = os.path.join(bot_base_path, input_dict["model_name"], input_dict["lang"], "trained_models", input_dict["model_name"] + ".model")
			load_time = time.time()
			instance_wmd = gensim.similarities.docsim.Similarity.load(models_path)
			print("\n time to load model --- ", time.time() - load_time)
			json_base_path = os.path.join(bot_base_path, input_dict["model_name"], input_dict["lang"], "training_data_jsons")
			training_sentences = json.load(open(os.path.join(json_base_path, "training_sentences_" + input_dict["model_name"] + ".json")))['sentences']
			qna_dict = json.load(open(os.path.join(json_base_path, "qna_dict.json")))
			wmd_sims = instance_wmd[query_token]
			wmd_sims = sorted(enumerate(wmd_sims), key=lambda item: -item[1])
			similar_docs = [(s, training_sentences[i])  for i,s in wmd_sims]
			top_answers = similar_docs[:3]
			print("\n top_answer --- ",top_answers)
			if top_answers:
				top_ans = qna_dict[top_answers[0][1]]
				top_ans = top_ans.replace("<p>", "").replace("</p>", "")
				top_ans = top_ans.strip()
				top_ans_tuple = [(top_answers[0][0], top_ans)]
				print("\n top ans === ", top_ans_tuple)
			else:
				top_ans_tuple = [(0.0, "Answer not found!")]
			print("\n total prediction time --- ", time.time() - st)

	except Exception as e:
		print("\n Error in qnamaker API main() --- ", e, "\n ",traceback.format_exc())

	return render_template('result.html',query=orig_query, len=len(top_ans_tuple),prediction = top_ans_tuple)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)