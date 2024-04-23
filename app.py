from flask import *
import os,sys
app = Flask(__name__)

user_id='admin'
user_pwd='admin123'


@app.route('/submit_log',methods=['POST','GET'])
def loginfo():
    if request.method=='POST':
        user_name = request.form['l_user_name']
        user_password = request.form['l_user_password']
        if user_name == user_id and user_password == user_pwd:
            return redirect(url_for('home'))
        else:
            return render_template("login.html",error='Invalid credentials')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')
    

@app.route('/detector')
def detector():
    return render_template('apple_detector.html')

@app.route('/model_parameter')
def model_parameter():
    return render_template('model_parameter.html')

# @app.route('/display/<filename>')
# def display_image(filename):
#     # print('display_image filename: ' + filename)
#     return redirect(url_for('static_new', filename='uploads/' + filename))

@app.route('/submit_detector', methods=['POST'])
def choose_file():
    if request.method == 'POST':
        keyword = request.form['url']
        return render_template('apple_detector.html',text =keyword)
                                            




if __name__=='__main__':
    app.run(debug=True)