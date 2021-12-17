import pickle
from flask import Flask,render_template, request
import ee
import numpy as np
import logging

ee.Initialize()

model = pickle.load(open("model.pkl",'rb'))

# Import the MODIS land cover collection.
lc = ee.ImageCollection('MODIS/006/MCD12Q1')

# Import the USGS ground elevation image.
elv = ee.Image('USGS/SRTMGL1_003')

# Import the Landsat 8 collection for NDVI.
l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')

# Import the MERIT Hydro: global hydrography datasets for Flow Accumulation
fd = ee.Image("MERIT/Hydro/v1_0_1")

app = Flask(__name__)

@app.route('/')
def map_func():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    u_lat = request.form['lat']
    u_lon = request.form['long']
    u_poi = ee.Geometry.Point(float(u_lon), float(u_lat))
    scale = 1000  # scale in meters

    elv_urban_point = elv.sample(u_poi, scale).first().get('elevation').getInfo()
    #print('Ground elevation at urban point:', elv_urban_point, 'm')

    lc_urban_point = lc.first().sample(u_poi, scale).first().get('LC_Type1').getInfo()
    #print('Land cover value at urban point is:', lc_urban_point)

    aspect = ee.Terrain.aspect(elv)
    aspect_urban_point = aspect.sample(u_poi, scale).first().get('aspect').getInfo()
    #print(aspect_urban_point)

    slope = ee.Terrain.slope(elv)
    slope_urban_point = slope.sample(u_poi, scale).first().get('slope').getInfo()
    #print(slope_urban_point)

    img = ee.Image(l8.filterBounds(u_poi).filterDate('2021-01-01', '2021-01-31').sort('CLOUD_COVER').first())

    def meanNDVICollection(img):
        nir = img.select('B5');
        red = img.select('B4');
        ndviImage = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

    # Compute the mean of NDVI over the 'region'
        ndviValue = ndviImage.reduceRegion(**{
        'geometry': img.geometry(),
        'reducer': ee.Reducer.mean(),
        'scale': 30,
        'maxPixels': 1e9
    }).get('NDVI');  # result of reduceRegion is always a dictionary, so get the element we want

        newFeature = ee.Feature(None, {
        # Adding computed NDVI value
        'NDVI': ndviValue
    }).copyProperties(img, [
        # Picking properties from original image
        'system:time_start',
        'SUN_ELEVATION'
    ])

        return newFeature

    ndvi = meanNDVICollection(img)

    ndvi = ndvi.get('NDVI').getInfo()
    #print(ndvi)

    fa = fd.select('upa')
    fa_urban_point = fa.sample(u_poi, scale).getInfo()
    fac = fa_urban_point['features'][0]['properties']['upa']
    #print(fac)

    try:
        twi = np.log(fac*1000/np.tan(slope_urban_point))
    except:
        twi = 10
    #print(twi)
    try:
        spi = fac*np.tan(slope_urban_point)
    except:
        spi = 10
    #print(spi)

    rainfall = 28

    fs= [[elv_urban_point,slope_urban_point,aspect_urban_point,twi,spi,ndvi,rainfall,lc_urban_point]]

    pd = model.predict_proba(fs)*100
    value = {
        "Landslide" : round(pd[0][0],2),
        "No_Landslide": round(pd[0][1],2)
    }

    return render_template('value.html', value=value)

@app.route('/lscoord')
def ls_func():
	return render_template('lscoord.html')

if __name__ == '__main__':
    app.run(debug = True)