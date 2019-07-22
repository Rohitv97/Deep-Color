from flask import Flask, flash, request, redirect, render_template, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from remove import rem
from cnn import color
from inception import color_inception
from xception import color_xception
import time

UPLOAD_FOLDER = './static/color'
CNN_FOLDER = './static/cnn'
ALLOWED_EXTENSIONS = set(['txt','pdf','png','jpg','jpeg','gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])

def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return render_template('dashboard.html', filename = filename)

	return render_template('index.html')

@app.route('/static/<filename>')
def send_file(filename):
	return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/index.html', methods=['GET','POST'])
def render_init():
	rem()
	if request.method == 'POST':

		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return render_template('/dashboard.html', filename = filename)

	return render_template('index.html')

@app.route('/dashboard.html')
def dashboard():
	return render_template('dashboard.html')

@app.route('/cnn.html')
def use_cnn():
	color()
	message = "Colorized using CNN"
	f = os.path.join('static','cnn','img_cnn_0.png')
	return render_template('/cnn.html', message = message, image= f)

@app.route('/inception.html')
def use_inception():
	color_inception()
	message = "Colorized using CNN and Inception-Resnet"
	f = os.path.join('static','inc','img_inception_0.png')
	return render_template('inception.html', message = message, image = f)

@app.route('/xception.html')
def use_xception():
	color_xception()
	message = "Colorized using CNN and Xception"
	f = os.path.join('static','xce','img_xception_0.png')
	return render_template('xception.html', message = message, image = f)

if __name__ == '__main__':
	app.debug = True
	app.run()
