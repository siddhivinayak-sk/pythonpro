from flask import Flask, render_template
from flask_cors import CORS
from routes.users import users_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(users_bp)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
