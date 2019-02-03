from flask import render_template, flash, redirect, request
from main import app

@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if _is_good(request.form):
            return render_template('good.html')
        else:
            return render_template('bad.html')
    else:
        return render_template('index.html')

def _is_good(form_data):
    return True


@app.route('/good.html')
def good():
    return render_template('good.html')


@app.route('/bad.html')
def bad():
    return render_template('bad.html')
