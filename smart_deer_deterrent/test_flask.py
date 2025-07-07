from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Flask is working!'

@app.route('/mobile')
def mobile():
    return 'Mobile interface is working!'

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(host='0.0.0.0', port=5001, debug=False)