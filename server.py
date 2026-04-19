from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Sirve el index.html que está en la carpeta /templates
    return render_template('index.html')

if __name__ == '__main__':
    # El sitio correrá en http://localhost:5000
    app.run(debug=True, port=5000)