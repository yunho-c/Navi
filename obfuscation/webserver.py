# import

#  @Bek Brace [ Twitter - Dev.to - GitHub ]
#  VueJs - Flask Full-Stack Web Application
#  bekbrace.com - info@bekbrace.com
#  Source Code : Michael Hermann [ mjheaO ]

from flask import Flask, request, flash, redirect, jsonify, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import os

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# @app.route('/uploads/<name>')
# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



CORS(app, resources={r"/*":{'origins':"*"}})
# CORS(app, resources={r'/*':{'origins': 'http://localhost:8080',"allow_headers": "Access-Control-Allow-Origin"}})

# hello world route
@app.route('/', methods=['GET'])
def greetings():
    return("Hello, world!")

@app.route('/shark', methods=['GET'])
def shark():
    return("Shark🦈!")


GAMES = [

    {   'id': uuid.uuid4().hex,
        'title':'2k21',
        'genre':'sports',
        'played': True,
    },
    {   'id': uuid.uuid4().hex,
        'title':'Evil Within',
        'genre':'horror',
        'played': False,
    },
    {   'id': uuid.uuid4().hex,
        'title':'the last of us',
        'genre':'survival',
        'played': True,
    },
    {  'id': uuid.uuid4().hex,
        'title':'days gone',
        'genre':'horror/survival',
        'played': False,
    },
    {   'id': uuid.uuid4().hex,
        'title':'mario',
        'genre':'retro',
        'played': True,
    }

]

@app.route('/process', methods=['GET'])
def process_image():
    response_object = {}
    request.get_json()


# The GET and POST route handler
@app.route('/games', methods=['GET', 'POST'])
def all_games():
    response_object = {'status':'success'}
    if request.method == "POST":
        post_data = request.get_json()
        GAMES.append({
            'id' : uuid.uuid4().hex,
            'title': post_data.get('title'),
            'genre': post_data.get('genre'),
            'played': post_data.get('played')})
        response_object['message'] =  'Game Added!'
    else:
        response_object['games'] = GAMES
    return jsonify(response_object)


#The PUT and DELETE route handler
@app.route('/games/<game_id>', methods =['PUT', 'DELETE'])
def single_game(game_id):
    response_object = {'status':'success'}
    if request.method == "PUT":
        post_data = request.get_json()
        remove_game(game_id)
        GAMES.append({
            'id' : uuid.uuid4().hex,
            'title': post_data.get('title'),
            'genre': post_data.get('genre'),
            'played': post_data.get('played')
        })
        response_object['message'] =  'Game Updated!'
    if request.method == "DELETE":
        remove_game(game_id)
        response_object['message'] = 'Game removed!'    
    return jsonify(response_object)


# Removing the game to update / delete
def remove_game(game_id):
    for game in GAMES:
        if game['id'] == game_id:
            GAMES.remove(game)
            return True
    return False

if __name__ == "__main__":
    app.run(debug=True)

