import model_pred
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('form1.html')


@app.errorhandler(Exception)
def server_error(err):
    app.logger.exception(err)
    return "Error", 500


@app.route('/getForecast')
def get_forecast():
    PRICE_NOM = request.args.get('PRICE_NOM')
    PRICE_NOM = float(PRICE_NOM)

    CHANGE = request.args.get('CHANGE')
    CHANGE = float(CHANGE)

    SALE = request.args.get('SALE')
    SALE = float(SALE)

    YEAR = request.args.get('YEAR')
    YEAR = float(YEAR)

    COST = request.args.get('COST')
    COST = float(COST)

    CHANGE_NOM = request.args.get('CHANGE_NOM')
    CHANGE_NOM = float(CHANGE_NOM)

    TRACT = request.args.get('TRACT')
    TRACT = float(TRACT)

    User_parameters = [[PRICE_NOM, CHANGE, SALE, COST, YEAR,
                        CHANGE_NOM, TRACT]]

    answer = model_pred.prediction(User_parameters)
    return render_template('answer_forecast.html', answer=answer, parameters=User_parameters)


if __name__ == '__main__':
    app.run(host='localhost', port=3000)
