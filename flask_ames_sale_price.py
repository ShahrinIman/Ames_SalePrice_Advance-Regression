from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from joblib import dump, load


import numpy as np  



def return_prediction(model,scaler,sample_json):
    
    lot_front = sample_json['Lot Frontage']
    lot_area = sample_json['Lot Area']
    ovr_qual = sample_json['Overall Qual']
    yr_built = sample_json['Year Built']
    yr_remod = sample_json['Year Remod/Add']
    bsmt_fin = sample_json['BsmtFin SF 1']
    t_bsmt_sf = sample_json['Total Bsmt SF']
    first_flr_sf = sample_json['1st Flr SF']
    gr_liv_area = sample_json['Gr Liv Area']
    gr_cars = sample_json['Garage Cars']
    gr_area = sample_json['Garage Area']
        
    
    sale_price = [[lot_front,lot_area,ovr_qual,yr_built,yr_remod,bsmt_fin,
                   t_bsmt_sf,first_flr_sf,gr_liv_area,gr_cars,gr_area]]
    
    sale_price = scaler.transform(sale_price)
    
    prediction = model.predict(sale_price)

    prediction = prediction**2

    prediction = np.round(prediction,2)
    
    return prediction[0]





app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
sale_price_model = load("ames_model.h5")
sale_price_scaler = load("ames_scaler.pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class SalePriceForm(FlaskForm):
    lot_front = TextField('Lot Frontage (lft)')
    lot_area = TextField('Lot Area (m2)')
    ovr_qual = TextField('Overall Qual (nos)')
    yr_built = TextField('Year Built (year)')
    yr_remod = TextField('Year Remodeled (year)')
    bsmt_fin = TextField('Basement Finished Square Feet Type 1 (m2)')
    t_bsmt_sf = TextField('Total Basement Square Feet (m2)')
    first_flr_sf = TextField('First Floor Square Feet (m2)')
    gr_liv_area = TextField('Garage Living Area (m2)')
    gr_cars = TextField('Garage Cars (nos)')
    gr_area = TextField('Garage Area (m2)')
        
    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = SalePriceForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['lot_front'] = form.lot_front.data
        session['lot_area'] = form.lot_area.data
        session['ovr_qual'] = form.ovr_qual.data
        session['yr_built'] = form.yr_built.data
        session['yr_remod'] = form.yr_remod.data
        session['bsmt_fin'] = form.bsmt_fin.data
        session['t_bsmt_sf'] = form.t_bsmt_sf.data
        session['first_flr_sf'] = form.first_flr_sf.data
        session['gr_liv_area'] = form.gr_liv_area.data
        session['gr_cars'] = form.gr_cars.data
        session['gr_area'] = form.gr_area.data      
        
        return redirect(url_for("prediction"))


    return render_template('ames_homepage.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Lot Frontage'] = float(session['lot_front'])
    content['Lot Area'] = float(session['lot_area'])
    content['Overall Qual'] = float(session['ovr_qual'])
    content['Year Built'] = float(session['yr_built'])
    content['Year Remod/Add'] = float(session['yr_remod'])
    content['BsmtFin SF 1'] = float(session['bsmt_fin'])
    content['Total Bsmt SF'] = float(session['t_bsmt_sf'])
    content['1st Flr SF'] = float(session['first_flr_sf'])
    content['Gr Liv Area'] = float(session['gr_liv_area'])
    content['Garage Cars'] = float(session['gr_cars'])
    content['Garage Area'] = float(session['gr_area'])
        
    results = return_prediction(model=sale_price_model,scaler=sale_price_scaler,sample_json=content)

    return render_template('ames_prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)