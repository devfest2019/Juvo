from app import app

app.config['SECRET_KEY'] = 'you-will-never-guess'
app.run(debug=True)