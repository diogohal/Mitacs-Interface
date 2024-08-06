from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        return render_template('results.html', start_date=start_date)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)