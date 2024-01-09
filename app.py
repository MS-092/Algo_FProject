# app.py
from dash import Dash, dcc, html, Input, Output, callback
from model_page import model_layout
import plotly.express as px
import plotly.graph_objects as go
import dash

app = Dash(__name__)

# Linear Regression
linear_regression_section = html.Div([
  html.H2('Linear Regression'),
  html.P('Linear regression is a statistical method that allows us to study relationships between two continuous (quantitative) variables.'
         ' One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable.'),
  html.P('Pros: Simple to implement and interpret, works well with small datasets.'),
  html.P('Cons: Assumes a linear relationship between variables, sensitive to outliers, assumes constant variance of residuals.'),
])

# Decision Tree
decision_tree_section = html.Div([
  html.H2('Decision Tree'),
  html.P('A decision tree is a flowchart-like structure in which each internal node represents feature(or attribute), each leaf node represents class label (decision taken after evaluating all features), and each branch represents a rule.'),
  html.P('Pros: Easy to understand and interpret, handles both categorical and numerical data, less prone to overfitting.'),
  html.P('Cons: Can create biased trees if some classes dominate, sensitive to small changes in data.'),
])

# Random Forest
random_forest_section = html.Div([
  html.H2('Random Forest'),
  html.P('Random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.'),
  html.P('Pros: Works well with both categorical and numerical data, less likely to overfit than single decision trees, provides feature importance.'),
  html.P('Cons: Less interpretable than decision trees, computationally expensive with large datasets.'),
])

# K-NN
knn_section = html.Div([
  html.H2('K-NN'),
  html.P('K-NN is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.'),
  html.P('Pros: Simple to understand and implement, works well with multi-class classification, less prone to overfitting.'),
  html.P('Cons: Computationally intensive with large datasets, sensitive to irrelevant attributes.'),
])

# Home Page Layout
home_page_layout = html.Div([
  html.H1('Welcome to Machine Learning Exploration'),
  linear_regression_section,
  decision_tree_section,
  random_forest_section,
  knn_section,
  dcc.Link('Go to Model Page', href='/model-page'),
])

app.layout = html.Div([
   dcc.Location(id='url', refresh=False),
   html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
           Input('url', 'pathname'))
def display_page(pathname):
   if pathname == '/model-page':
       return model_layout
   else:
       return home_page_layout

if __name__ == '__main__':
   app.run_server(debug=True)
