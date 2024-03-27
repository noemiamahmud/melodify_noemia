import os

from flask import Flask, render_template, request

from app import routes


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.register_blueprint(routes.bp)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'app.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def new_page():
        #return render_template('pages/index.html')
        return render_template('pages/index.html')
    
    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    #from .import routes
    #app.register_blueprint(routes.bp)

    #from . import music
    #app.register_blueprint(music.bp)

    return app
'''
    @app.route('/submit', methods=['POST'])
    def submit():
        if request.method == 'POST':
            data = request.form['data']
            return 'Data recieved: ' + data
        else:
            return 'Method not allowed', 405
            '''



if __name__=="__main__":
    app = create_app()

    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8888)))
    
'''
    @app.route('/process_data', methods=['POST'])
    def process_data():
        if (request.method == "POST"):
            input_text = request.form['inputText']  # Access the inputText field from form data
            # Now you can process the input_text as needed
            print('Received input:', input_text)
            # You can return a response if needed
            return 'Data received successfully'
            '''
    