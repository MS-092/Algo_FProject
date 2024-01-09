from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from dash.exceptions import PreventUpdate

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
"""
Dropdown menu for the models
Upload button
Graph display
component for evaluation metrics and predictions
"""
model_layout = html.Div([
 html.H1('Welcome to Model Exploration'),
 
# Choosing model algorithms
 html.P("Select model:"),
 dcc.Dropdown(
 id='ml-regression-x-dropdown',
 options=["Regression", "Decision Tree", "k-NN", "Random Forest"],
 value='Decision Tree',
 clearable=False
 ),
 # Upload dataset
 dcc.Upload(
 id='upload-data',
 children=html.Div([
 'Drag and Drop or ',
 html.A('Select Files')
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
 html.Div(id='upload-data-preview'),
 dcc.Graph(id="ml-regression-x-graph"),
 dcc.Markdown(id='evaluation-output'),
 dcc.Markdown(id='predicted-output'),
 html.Div(id='predicted-output-table')
])

# Parse upload files
# Based on the type of file being uploaded
def parse_contents(contents, filename, date):
 content_type, content_string = contents.split(',')
 decoded = base64.b64decode(content_string)

 try:
  if 'csv' in filename:
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
  elif 'xls' in filename:
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
  )
 ])

# Callbacks, preview and the ones that are decided by the user - based on upload
# Selected set of actions(model, graph, evaluation, etc) - based on group of children above
@app.callback(
 Output('upload-data-preview', 'children'),
 Input('upload-data', 'contents'),
 State('upload-data', 'filename'),
 State('upload-data', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
 if list_of_contents is not None:
  children = [
      parse_contents(c, n, d) for c, n, d in
      zip(list_of_contents, list_of_names, list_of_dates)
  ]
  return children

@app.callback(
 Output("ml-regression-x-graph", "figure"), 
 Output('evaluation-output', 'children'),
 Output('predicted-output', 'children'),
 Input('ml-regression-x-dropdown', "value"),
 Input('upload-data', 'contents')
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
    **Model:** {name}
    **Mean Squared Error:** {mse}
    **Mean Absolute Error:** {mae}
    **R-squared:** {r2}
    **Correlation Coefficient:** {corr}
    """

    pred_output = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    return fig, eval_output, pred_output.to_markdown(index=False)

if __name__ == "__main__":
 app.run_server(debug=True)
