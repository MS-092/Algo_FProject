from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from sklearn.model_selection import train_test_split
# linear regression, decision tree, K-Neighbors
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import datetime
from sklearn.ensemble import RandomForestRegressor  # Random forest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Group, Scheme, Symbol

# Set up dash application
app = Dash(__name__)

# Set up the models that are going to be used
models = {
    'Regression': linear_model.LinearRegression,
    'Decision Tree': tree.DecisionTreeRegressor,
    'k-NN': neighbors.KNeighborsRegressor,
    'Random Forest': RandomForestRegressor
}

# Layout of the application
# HTML Components
# Layout of brief description of the algorithms used
# Dropdown menu for the models
# Upload button
# Graph display
# Evaluation metrics
# Prediction results

# Callbacks, preview and the ones that are decided by the user - based on upload
# Selected set of actions(model, graph, evaluation, etc) - based on group of children above
# Executes, when certain actions are met,
        # 1. Upload file
        # 2. Trained and selected a model to run
        # 3. Make predictions
        # 4. Updates graphs
        # 5. Shows evaluation results


# Components of algorithms description
# Model name
# Model description
# Pros&Cons

# Linear Regression
linear_regression_section = html.Div([
    html.H2('Linear Regression'),
    html.Img(src=app.get_asset_url('linear_regression.png'), style={'width': '20%', 'height': 'auto', 'objectFit': 'contain'}),
    html.P('Linear regression is a statistical method that allows us to study relationships between two continuous (quantitative) variables.'
           ' One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable.'),
    html.P('Pros: Simple to implement and interpret, works well with small datasets.'),
    html.P('Cons: Assumes a linear relationship between variables, sensitive to outliers, assumes constant variance of residuals.'),
])

# Decision Tree
decision_tree_section = html.Div([
    html.H2('Decision Tree'),
    html.Img(src=app.get_asset_url('decision_tree.png'), style={'width': '20%', 'height': 'auto', 'objectFit': 'contain'}),
    html.P('A decision tree is a flowchart-like structure in which each internal node represents feature(or attribute), each leaf node represents class label (decision taken after evaluating all features), and each branch represents a rule.'),
    html.P('Pros: Easy to understand and interpret, handles both categorical and numerical data, less prone to overfitting.'),
    html.P('Cons: Can create biased trees if some classes dominate, sensitive to small changes in data.'),
])

# Random Forest
random_forest_section = html.Div([
    html.H2('Random Forest'),
    html.Img(src=app.get_asset_url('random_forest.png'), style={'width': '20%', 'height': 'auto', 'objectFit': 'contain'}),
    html.P('Random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.'),
    html.P('Pros: Works well with both categorical and numerical data, less likely to overfit than single decision trees, provides feature importance.'),
    html.P('Cons: Less interpretable than decision trees, computationally expensive with large datasets.'),
])

# K-NN
knn_section = html.Div([
    html.H2('K-NN'),
    html.Img(src=app.get_asset_url('KNN.png'), style={'width': '20%', 'height': 'auto', 'objectFit': 'contain'}),
    html.P('K-NN is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.'),
    html.P('Pros: Simple to understand and implement, works well with multi-class classification, less prone to overfitting.'),
    html.P('Cons: Computationally intensive with large datasets, sensitive to irrelevant attributes.'),
])

app.layout = html.Div([
    html.H1('Welcome to Machine Learning Exploration', style={'textAlign': 'center'}),
    html.H3('Where you can start learning and getting your interests high to start your machine learning journey. With our playground that covers some basics about machine learning and what do they do', style={'textAlign': 'center'}),
    html.Div(linear_regression_section, style={'textAlign': 'center'}),
    html.Div(decision_tree_section, style={'textAlign': 'center'}),
    html.Div(random_forest_section, style={'textAlign': 'center'}),
    html.Div(knn_section, style={'textAlign': 'center'}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.A(html.Button("Check out the code in my github"),
           href="https://github.com/MS-092/Algo_FProject"),
    
    # Choosing model algorithms
    html.H2(
        'Now here I have provided you some model algorithms playground for you to play'),
    html.H3('Here you can choose any provided model algorithms, with your own custom dataset'),
    html.P("Select model:"),
    dcc.Dropdown(
        id='ml-regression-x-dropdown',
        options=["Regression", "Decision Tree", "k-NN", "Random Forest"],
        value='Decision Tree',
        clearable=False
    ),
    # Upload dataset - Parsing upload files
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',  # drag and drop area
            html.A('Select Files')  # upload button
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    # Set up the layout
    html.Div(id='upload-data-preview'),
    dcc.Graph(id="ml-regression-x-graph"),
    dcc.Markdown(id='evaluation-output'),
    dcc.Markdown(id='predicted-output'),
    html.Div(id='predicted-output-table')
])

# Parse upload files, getting the information of the data
# Based on the type of file being uploaded


def parse_contents(contents, filename, date):
    # This check ensures that there is a comma in the contents to split
    if ',' in contents:
        content_type, content_string = contents.split(',')
    else:
        # Handle the case where contents does not contain a comma
        return html.Div([
            'There was an error processing this file.'
        ])

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            page_action='native',
            page_size=10
        ),
        html.Hr(),
        html.Div('Raw Data'),
        html.Pre(contents[0:10] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(
    Output('upload-data-preview', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # Make sure all variables are lists to handle single or multiple file uploads
        if not isinstance(list_of_contents, list):
            list_of_contents = [list_of_contents]
        if not isinstance(list_of_names, list):
            list_of_names = [list_of_names]
        if not isinstance(list_of_dates, list):
            list_of_dates = [list_of_dates]
        
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children

@app.callback(
    Output("ml-regression-x-graph", "figure"),
    Output('evaluation-output', 'children'),
    Output('predicted-output-table', 'children'),  # Add this line for the table output
    Input('ml-regression-x-dropdown', "value"),
    Input('upload-data', 'contents'),
)
def train_and_display(name, contents):
    if contents is None:
        raise PreventUpdate
    else:
      content_type, content_string = contents.split(',')
      decoded = base64.b64decode(content_string)
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

      # Fetch the first two column names
      feature_col = df.columns[0]
      target_col = df.columns[1]

      X = df[feature_col].values[:, None]
      X_train, X_test, y_train, y_test = train_test_split(
          X, df[target_col], random_state=42)

      model = models[name]()
      model.fit(X_train, y_train)

      predictions = model.predict(X_test)

      x_range = np.linspace(X.min(), X.max(), 100)
      y_range = model.predict(x_range.reshape(-1, 1))

      fig = go.Figure([
          go.Scatter(x=X_test.squeeze(), y=y_test,
                      name='test', mode='markers'),
          go.Scatter(x=x_range, y=y_range,
                      name='prediction')
      ])

      mse = mean_squared_error(y_test, predictions)
      mae = mean_absolute_error(y_test, predictions)
      r2 = r2_score(y_test, predictions)
      corr, _ = pearsonr(y_test, predictions)

      eval_output = f"""
  **Model:** {name} \n
  **Mean Squared Error:** {mse} \n
  **Mean Absolute Error:** {mae} \n
  **R-squared:** {r2 * -1} \n
  **Correlation Coefficient:** {corr}
  """
      # average absolute difference between actual and predicted values - the lower the better value or going into 0%
      # how well regression model explained observed values/data - 0 until 1

      # Showing the list of the actual data and predicted output dataset      
      # Convert the actual and predicted values to a DataFrame
      pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

      # Create a DataTable to display the actual and predicted values
      table = dash_table.DataTable(
        data=pred_df.to_dict('records'),
        columns=[
            {'name': 'Actual', 'id': 'Actual', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
            {'name': 'Predicted', 'id': 'Predicted', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
        ],
        sort_action='native', # Enable sorting system in the table output
        style_table={'width': '50%'},  # Adjust the width of the table as needed
        style_cell={'textAlign': 'center'}
    )

      return fig, eval_output, table # Return statement and bringing out all of the output


# Main function to run the whole program process
if __name__ == "__main__":
    app.run_server(debug=True)
