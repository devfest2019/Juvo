from flask import render_template, flash, redirect, request, url_for

from main import app
from main import bank
from main import model_api

import os
print(os.getcwd())

# :(
customer_name = None
url = None

@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    global customer_name

    if request.method == 'POST':
        customer_name = request.form['first']
        return redirect(url_for('form'))
    else:
        return render_template('index.html')


@app.route('/education.html')
def education():
    return redirect("https://www.edx.org/course/entrepreneurship-in-emerging-economies")


@app.route('/form.html', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        if _is_good(request.form) and 'home' in request.form:
            return render_template('good.html', url=url)
        else:
            return render_template('bad.html')
    else:
        return render_template('form.html')


def _is_good(form_data):
    global url

    if model_api.determine_acceptance(form_data.to_dict()):
        ccs = 750
        url = bank.make_account(customer_name)
        bank.make_loan(url, ccs)
        print(url)
        return True
    else:
        return False


@app.route('/good.html')
def good():
    return render_template('good.html')


@app.route('/bad.html')
def bad():
    return render_template('bad.html')
