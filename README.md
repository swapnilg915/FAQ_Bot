# FAQ_Bot
Create your own FAQ bot with python and NLP.

It uses googles pre-trained word2vec model along with word movers distance (WMD) NLP algorithm.

steps to create your own FAQBot:

setup the project environment
1. Create virtual environment using the command : python3.7 -m venv faqbot_env_3.7
2. Activate the virtual environment using : source faqbot_env_3.7/bin/activate
3. Run the above command to install setup tools : python3 -m pip install setuptools pip install -U wheel
4. Install all the required python packages using : python3 -m pip install -r requirements.txt
5. Run the flask API : python3 upload_train_predict_api.py
6. Download the googles word2vec model from : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
and keep it in any directory and update that path in config.py file. 

Training:
1. keep your faq data ready in :
	A] xlsx file format with 2 columns namely "Question" and "Answer".
	B] pdf file with a format same as of the file "structured_faq.pdf" in dataset folder.
The sample data used in this project is available in "dataset" folder.
2. In browser run: http://0.0.0.0:5000/upload
3. Provide the necessary inputs like language and model name, select the appropriate path of xlsx / pdf file and click upload.
	Note:Please save the model name somewhere. It will be usefull for prediction.
4. Thats it!!!training is done.

Prediction: 
1. In browser run: http://0.0.0.0:5000/predict
2. Provide the necessary inputs along with a question and hit "Get Answer". You will get an asnwer along with the confidence score.
Thats it!!!


NOTE: The data used for this project is downloaded from microsoft azure website. It is used only for study purpose: https://docs.microsoft.com/en-us/azure/cognitive-services/QnAMaker/concepts/data-sources-and-content

References:
https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html


