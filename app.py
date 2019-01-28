#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
from flask import Flask, request, render_template, jsonify
from flask_bootstrap import Bootstrap   
from sklearn.decomposition import PCA
import os
import numpy as np
import pandas as pd
import requests
import io
from waitress import serve
from bokeh.plotting import figure
from bokeh.embed import json_item
from bokeh.palettes import Viridis
from bokeh.models import LinearColorMapper,CategoricalColorMapper, ColorBar
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
import json






#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#
app = Flask(__name__)
app.config.from_object('config')
tz = "Asia/Singapore"
bootstrap = Bootstrap(app)

colormap = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
column_names_colormap={'sepal length':'#b3ffb3','sepal width':'#99ddff', 'petal length':'#ffd699',\
                       'petal width':'#ffff00', 'species':'#3333ff'}
species=['Iris-setosa','Iris-versicolor','Iris-virginica']

#-------------------------------#
# All Visualizations are built on bokeh 1.0.4
#Load the iris data
# Returns a pandas dataframe of the data contained in the link from the ics.uci.edu 

def load_data(url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', column_names=['sepal length','sepal width', 'petal length', 'petal width', 'species'], rows=5):
    try:
        urlData=requests.get(url).content
        data=pd.read_csv(io.StringIO(urlData.decode('utf-8')), names=column_names)
        if rows=='all':
            return data
        else:
            return data.iloc[:rows]
    except Exception as e:
        print(e)
#to create the vertical bar charts for the distribution of the counts of the different species of flowers in the closest matches
def create_pred_dist(data):
    data=data.groupby(by='species',as_index=False).count()[['species','sepal length']]
    p = figure(x_range=data['species'],plot_width = 500, plot_height = 300, title="Species Counts")
    
    p.vbar(x='species', top='sepal length', width=0.9, color=factor_cmap('species', palette=Spectral6, \
                                                                         factors=data['species'].unique()), source=data)
    p.yaxis.axis_label = 'Count'
    return p
# to provide the variation in the features e.g. petal length , petal width for the closest matches
def plot_pred_variation(data):
    p = figure(plot_width = 550, plot_height = 300,\
               title = 'Variation of parameters for the closest mattching points')
    p.xaxis.axis_label = 'Index'
    p.yaxis.axis_label = 'Value'
    
    for column in list(column_names_colormap.keys())[:-1]:
        p.circle(np.arange(0,len(data),1),data[column],color=column_names_colormap[column], fill_alpha=0.6,legend=column, size=5)
    p.legend.location='top_left'
    p.legend.label_text_font_size ='8pt'
    return p

    
    

# Used for EDA to find the distribution of the features is the dataset  
def create_histogram(current_feature_name,bins):
    hist, edges = np.histogram(iris_data[current_feature_name], density=False, bins=bins)
    p = figure(plot_width = 500, plot_height = 300, \
               title='Histogram of {}'.format(current_feature_name),background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],\
           fill_color=column_names_colormap[current_feature_name], line_color='black'\
           , alpha=0.5)
    x_edges=[(edges[i]+edges[i+1])/2  for i in range(len(edges)-1 )]
    p.line(x_edges,hist,line_color="blue", line_width=2, alpha=0.7, legend=current_feature_name)
    # Set the x axis label
    p.xaxis.axis_label = current_feature_name
    # Set the y axis label
    p.yaxis.axis_label = 'Count'
    p.legend.location='top_right'
    return p
# Used for EDA to generate the scatter plot between the various feature combinations
def plot_variation(x_axis, y_axis):
    p = figure(plot_width = 550, plot_height = 300, 
                  title = 'Variation of {} with {} '.format(x_axis,y_axis))
    p.xaxis.axis_label = x_axis
    p.yaxis.axis_label = y_axis
    
    for species in colormap.keys():
        p.circle(iris_data[iris_data['species']==species][x_axis],iris_data[iris_data['species']==species][y_axis],color=colormap[species], fill_alpha=0.2,legend=species, size=5)
    p.legend.location='top_left'
    p.legend.label_text_font_size ='8pt'
    return p
# find the outliers for each category in the iris dataset for e.g. Itis-setosa
    # USed in the boxplot
def outliers(group, **kwargs):
    cat = group.name
    return group[(group[kwargs['args'][0]]> kwargs['args'][1].loc[cat][kwargs['args'][0]]) | (group[kwargs['args'][0]]< kwargs['args'][2].loc[cat][kwargs['args'][0]])][kwargs['args'][0]]
# generate the boxplots for each species
def plot_box(parameter):
    # find the quartiles and IQR for each category
    cats=species
    df=iris_data[[parameter,'species']]
    groups = df.groupby('species')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    
    # find the outliers for each category
    out = groups.apply(outliers, args=(parameter,upper,lower)).dropna()
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out.loc[keys[0]].loc[keys[1]])

    p = figure(plot_width = 550, plot_height = 300,background_fill_color="#efefef", x_range=cats)
    
    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper[parameter] = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,parameter]),upper[parameter])]
    lower[parameter] = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,parameter]),lower[parameter])]
    
    # stems
    p.segment(cats, upper[parameter], cats, q3[parameter], line_color="black")
    p.segment(cats, lower[parameter], cats, q1[parameter], line_color="black")
    
    # boxes
    p.vbar(cats, 0.7, q2[parameter], q3[parameter], fill_color="#5637f2", line_color="black")
    p.vbar(cats, 0.7, q1[parameter], q2[parameter], fill_color="#a0f7d3", line_color="black")
    
    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower[parameter], 0.2, 0.01, line_color="black")
    p.rect(cats, upper[parameter], 0.2, 0.01, line_color="black")
    
    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.4)
                 
    p.yaxis.axis_label = parameter
    
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"   
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size="8pt"
    
    return p
# Generates a heatmap for the correlation between the different features in the iris dataset
def create_corrheat(parameter):
    data=iris_data[iris_data['species']==parameter]
    corr_data=data.corr()
    x_range=corr_data.columns.values.tolist()
    
    corr_data = corr_data.stack().reset_index()
    corr_data.columns=['x','y','value']
    mapper = LinearColorMapper(palette=Viridis[256], low= corr_data['value'].min(), high=corr_data['value'].max())
    p = figure(title="Correlation Heatmap of {}".format(parameter),x_range=x_range, y_range=list(reversed(x_range)),\
               plot_width=550, plot_height=300)
    p.rect(x='x', y='y', width=1, height=1,fill_color={'field': 'value','transform': mapper},source=corr_data,line_color=None)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    return p
        

# Loading the iris data and keeping in memory for quick loading times
iris_data=load_data(rows='all')
# defining a principal component object in scikit-learn wth 2 principal components
pca = PCA(n_components=2)
# defining a standard scaler object in scikit-learn wth 2 principal components
scaler=StandardScaler()
y=iris_data['species']
x_scaler = scaler.fit(iris_data[['sepal length','sepal width', 'petal length', 'petal width']])
x_pca=pca.fit(x_scaler.transform(iris_data[['sepal length','sepal width', 'petal length', 'petal width']]))
# Loading the model already generated using a support vector classifier with 93% accuracy 
#model has been dumped using joblib
model=joblib.load('./data/models/iris_predictor')

# Generates predictions for the input features coming in fort he new case
def generate_predictions(params):
    x_scaled=x_scaler.transform(params)
    x_scaled_pca=x_pca.transform(x_scaled)
    prediction=model.predict(x_scaled_pca)
    return prediction
#--------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#
# Fetch the data from iris  dataset . Please install requests with python3.7
@app.route('/dashboard/getData',methods=['GET'])
def getData():
    data=load_data(rows='all')
    columns=data.iloc[:5].columns.values.tolist()
    iris_quality=data.describe()
    data=data.sample(5)
    quality_data=iris_quality.values.tolist()
    quality_columns=iris_quality.columns.values.tolist()
    quality_params=iris_quality.index.values.tolist()
    return jsonify({'data': data.values.tolist(),\
                  'columns':columns,
                  'data_name':'iris',
                  'categories':data['species'].unique().tolist(),
                  'quality_data': quality_data,
                  'quality_columns':quality_columns,
                  'quality_params':quality_params})
# fetches the scatter plot betwwen the various input features
@app.route('/dashboard/get_var', methods=['GET','POST'])
def get_var():
    # Determine the selected feature
    
    try:
        xaxis = request.get_json()['xaxis']
    except:
        
        xaxis='petal length'
    try:
        yaxis = request.get_json()['yaxis']
    except:
        
        yaxis='petal width'
    # Create the plot
    plot = plot_variation(xaxis,yaxis)
    return json.dumps(json_item(plot))
#fetches the box plots in application
@app.route('/dashboard/get_box', methods=['GET','POST'])
def get_box():
    # Determine the selected feature
    
    try:
        parameter = request.get_json()['parameter']
    except:
        
        parameter='petal length'
    # Create the plot
    plot = plot_box(parameter)
    return json.dumps(json_item(plot))
    

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# fetches the distribution histograms
@app.route('/dashboard/dist_hist', methods=['GET','POST'])
def dis_hist():
    # Determine the selected feature
    try:
        current_feature_name =  request.get_json()['parameter']
    except:
        current_feature_name = "sepal length"
    
        
    # Create the plot
    plot = create_histogram(current_feature_name, 10)
    return json.dumps(json_item(plot))
# fetches the distribution heatmaps
@app.route('/dashboard/get_corrheat', methods=['GET','POST'])
def corrheat():
    # Determine the selected feature
    try:
        current_feature_name =  request.get_json()['parameter']
    except:
        current_feature_name = "Iris-virginica"
    
        
    # Create the plot
    plot = create_corrheat(current_feature_name)
    return json.dumps(json_item(plot))
    
# fetches the whole dataset   
@app.route('/data')
def data():
    dataset=request.args.get('dataName')
    if dataset=='iris':
        data=load_data(rows='all')
    return render_template('data.html',data=data.values.tolist(), columns=data.columns.values.tolist())

# fetches the quality of the dataset
@app.route('/dashboard/get_quality', methods=['GET','POST'])
def get_quality():
    iris_quality=iris_data.describe()
    quality_data=iris_quality.values.tolist()
    quality_columns=iris_quality.columns.values.tolist()
    quality_params=iris_quality.index.values.tolist()
    return jsonify({'data': quality_data,\
                  'columns':quality_columns,
                  'params':quality_params})
    
# Fetches theclosest matching observations in the iris dataset 
    #The number of observations are decide by the user from the frontend    
@app.route('/dashboard/get_matches', methods=['GET','POST'])
def get_matches():
    # Determine the selected feature
    try:
        sepal_length =  float(request.get_json()['sepal_length'])
    except:
        sepal_length =5.8
    try:
        sepal_width =  float(request.get_json()['sepal_width'])
    except:
        sepal_width =3.05
    try:
        petal_length =  float(request.get_json()['petal_length'])
    except:
        petal_length =3.75
    try:
        petal_width =  float(request.get_json()['petal_width'])
    except:
        petal_width =1.19
    try:
        n=int(request.get_json()['n'])
    except:
        n=10
    iris_data_copy=iris_data.copy()
    #plot=create_pred_dist(iris_data_copy[iris_data_copy['species']])
    iris_data_copy['distance']=euclidean_distances(iris_data_copy[['sepal length','sepal width', 'petal length', 'petal width']]\
             ,[[sepal_length,sepal_width,petal_length,petal_width]])
        
    iris_data_copy=iris_data_copy.sort_values(by='distance',ascending=True).iloc[:n]
    plot=create_pred_dist(iris_data_copy)
    plot_pred_var=plot_pred_variation(iris_data_copy)
    prediction=generate_predictions([[sepal_length,sepal_width,petal_length,petal_width]])
    #print(json_item(plot))
    return jsonify({'data':iris_data_copy[['sepal length','sepal width', 'petal length', 'petal width','species']].values.tolist(),\
                    'columns':iris_data_copy[['sepal length','sepal width', 'petal length', 'petal width','species']].columns.values.tolist(),
                    'prediction':prediction[0], 'color':colormap[prediction[0]],'plot':json_item(plot),\
                    'plot_pred_var':json_item(plot_pred_var)})
#---------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#


# Or specify port manually:

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    #app.run(threaded=True,host='0.0.0.0', port=port)
    serve(app,host='0.0.0.0', port=5000)

