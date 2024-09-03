from flask import Flask, request, render_template, jsonify, send_file, make_response
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_sqlalchemy import SQLAlchemy
from flask_babel import Babel  # Import Flask-Babel instead of Flask-BabelEx
from transformers import MarianMTModel, MarianTokenizer
import os  # Importing os module for file handling
import sqlite3
import csv
from io import StringIO
import boto3
from datetime import datetime

app = Flask(__name__)

# Configure Flask and SQLAlchemy
app.config['SECRET_KEY'] = 'mysecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///translations.db'

# Initialize SQLAlchemy and Flask-Babel
db = SQLAlchemy(app)
babel = Babel(app)  # Initialize Flask-Babel

# Model for storing AWS credentials
class AWSCredentials(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    access_key = db.Column(db.String(50))
    secret_key = db.Column(db.String(50))
    bucket_name = db.Column(db.String(50))

# Admin setup
admin = Admin(app, name='AWS S3 Management', template_mode='bootstrap4')
admin.add_view(ModelView(AWSCredentials, db.session))

# Load model and tokenizer for English to Arabic translation
model_name = 'Helsinki-NLP/opus-mt-en-ar'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize the database
def init_db():
    with sqlite3.connect('translations.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT,
                translated_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Function to save translation to the database
def save_to_db(original_text, translated_text):
    with sqlite3.connect('translations.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO translations (original_text, translated_text)
            VALUES (?, ?)
        ''', (original_text, translated_text))
        conn.commit()

# Function to upload files to S3
def upload_to_s3(file_name, credentials):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key
    )
    try:
        s3_client.upload_file(file_name, credentials.bucket_name, file_name)
        return True
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = None

    if 'text_file' in request.files and request.files['text_file'].filename != '':
        text_file = request.files['text_file']
        text = text_file.read().decode('utf-8')
    else:
        text = request.form.get('text')

    if not text:
        return jsonify({'error': 'No text provided for translation'}), 400

    inputs = tokenizer(text, return_tensors='pt', padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Remove duplicate words
    words = translated_text[0].split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    unique_translated_text = " ".join(unique_words)

    # Save translation to database
    save_to_db(text, unique_translated_text)

    return jsonify({'translated_text': unique_translated_text, 'original_text': text})

@app.route('/export', methods=['POST'])
def export():
    translated_text = request.form.get('translated_text')
    original_text = request.form.get('original_text')
    
    # Prepare the CSV content in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Original Text', 'Translated Text', 'Timestamp'])
    writer.writerow([original_text, translated_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    
    # Return the CSV file as a response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=translation_output.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/export-last-20', methods=['POST'])
def export_last_20():
    file_name = "last_20_translations.csv"
    
    try:
        output = StringIO()

        with sqlite3.connect('translations.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT original_text, translated_text, timestamp
                FROM translations
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            rows = cursor.fetchall()

        if not rows:
            return jsonify({'error': 'No data available for export'}), 400

        # Write the CSV data to the output buffer
        writer = csv.writer(output)
        writer.writerow(['Original Text', 'Translated Text', 'Timestamp'])
        for row in rows:
            writer.writerow([row[0], row[1], row[2]])

        # Rewind the buffer
        output.seek(0)

        # Save the CSV file temporarily
        with open(file_name, 'w', newline='') as csvfile:
            csvfile.write(output.getvalue())

        # Retrieve the first set of AWS credentials from the database
        credentials = AWSCredentials.query.first()
        if credentials and upload_to_s3(file_name, credentials):
            os.remove(file_name)  # Remove the temporary file after uploading
            return jsonify({'message': 'File uploaded to S3 successfully'})
        else:
            os.remove(file_name)  # Remove the temporary file on failure
            return jsonify({'error': 'Failed to upload to S3'}), 500

    except Exception as e:
        return jsonify({'error': f'Error during export: {str(e)}'}), 500

@app.route('/recent', methods=['GET'])
def recent():
    with sqlite3.connect('translations.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT original_text, translated_text, timestamp
            FROM translations
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        rows = cursor.fetchall()

    return render_template('recent.html', translations=rows)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables if they don't exist
    init_db()  # Initialize the translation database
    app.run(debug=True, port=9000)
