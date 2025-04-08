import os, base64, dash, dash_bootstrap_components as dbc, plotly.express as px
import pandas as pd, numpy as np
from dash import dcc, html, dash_table, Input, Output
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from pipeline_local import run_pipeline

# ------------------------------------------------------------------ #
#  Data & Model
# ------------------------------------------------------------------ #
X_tr, y_tr, X_val, y_val, X_te, y_te = run_pipeline()
model = XGBClassifier(
    max_depth=5, learning_rate=0.1, gamma=4, min_child_weight=6,
    n_estimators=200, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="auc", use_label_encoder=False,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

y_prob = model.predict_proba(X_te)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
metrics = {
    "AUC": roc_auc_score(y_te, y_prob),
    "Accuracy": accuracy_score(y_te, y_pred),
    "Precision": precision_score(y_te, y_pred),
    "Recall": recall_score(y_te, y_pred),
    "F1": f1_score(y_te, y_pred),
}
cm = confusion_matrix(y_te, y_pred)
expr = "Expression" if "Expression" in X_tr.columns else X_tr.columns[0]

# ------------------------------------------------------------------ #
#  Figures
# ------------------------------------------------------------------ #
hist_fig = px.histogram(X_tr, x=expr, nbins=40,
                        title="Expression Distribution – Training Set")
target_fig = px.bar(x=["Class 0", "Class 1"], y=np.bincount(y_tr),
                    title="Target Balance After SMOTE",
                    labels={"x": "Class", "y": "Count"})
cm_fig = px.imshow(cm, text_auto=True,
                   x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                   title="Confusion Matrix – Test Set")
fi = pd.Series(model.feature_importances_, index=X_tr.columns) \
        .sort_values(ascending=False).head(20)
fi_fig = px.bar(fi[::-1], orientation="h",
                title="Top‑20 Feature Importances",
                labels={"value": "Importance", "index": "Feature"})

# ------------------------------------------------------------------ #
#  PDF‑derived narrative (condensed)
# ------------------------------------------------------------------ #
ABSTRACT = (
    "AdvanceHC Solution is developing an AI‑powered diagnostic tool to detect "
    "breast cancer early using machine‑learning models. Our challenge: the "
    "model is in trial phase, so we must prove reliability, address regulatory "
    "concerns and build trust with healthcare providers."
)
GOALS = [
    "Develop and test a high‑accuracy ML model for breast‑cancer detection.",
    "Validate against radiologist diagnoses.",
    "Ensure secure, compliant data storage & processing.",
    "Demonstrate transparent, explainable AI decisions.",
    "Forge partnerships with hospitals & research institutions.",
]
AUTHORS = "Arjun Venkatesh • Darren Chen • Vinh Dao"
COMPANY = "AdvanceHC Solution  |  Healthcare AI Startup"

# ------------------------------------------------------------------ #
#  Utility – encode screenshots
# ------------------------------------------------------------------ #
IMG_FILES = [
    "Screenshot 2025-04-07 at 5.26.26 PM.png",
    "Screenshot 2025-04-07 at 5.26.36 PM.png",
    "Screenshot 2025-04-07 at 5.26.47 PM.png",
    "Screenshot 2025-04-07 at 5.26.57 PM.png",
]
def _b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

images = [html.Img(src=_b64(os.path.join(os.path.dirname(__file__), p)),
                   style={"maxWidth": "100%", "marginBottom": "1rem"})
          for p in IMG_FILES if os.path.exists(p)]

# ------------------------------------------------------------------ #
#  Dash Layout
# ------------------------------------------------------------------ #
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
server = app.server

def kpi_card(k, v):
    return dbc.Col(dbc.Card(
        [dbc.CardHeader(k), dbc.CardBody(html.H4(f"{v:.3f}", className="card-title"))],
        className="shadow-sm"), md=2)

overview_tab = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Project Overview", className="text-primary")),
        dbc.Col(html.H6(AUTHORS, className="text-end text-muted"), width="auto"),
    ], align="center", justify="between"),
    html.Hr(),
    dbc.Row([dbc.Col(html.P(ABSTRACT))]),
    dbc.Row([dbc.Col(html.Ul([html.Li(g) for g in GOALS]))]),
    html.Hr(),
    dbc.Row([kpi_card(k, v) for k, v in metrics.items()], className="mb-4"),
    dbc.Row([
        dbc.Col(html.P(f"Training samples (after SMOTE): {len(X_tr)}")),
        dbc.Col(html.P(f"Validation samples: {len(X_val)}")),
        dbc.Col(html.P(f"Test samples: {len(X_te)}")),
    ], className="mb-2"),
], fluid=True)

dist_tab = dbc.Container([
    dbc.Row(dbc.Col(html.H4("Data Distributions", className="text-primary"))),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=hist_fig), md=6),
        dbc.Col(dcc.Graph(figure=target_fig), md=6),
    ]),
    dbc.Alert(
        "Left: log2‑ratio proteomic expression values show near‑Gaussian spread "
        "after scaling. Right: SMOTE has balanced the outcome variable, "
        "preventing model bias toward the majority class.",
        color="info", className="mt-3"),
], fluid=True)

perf_tab = dbc.Container([
    dbc.Row(dbc.Col(html.H4("Model Performance", className="text-primary"))),
    dbc.Row(dbc.Col(
        dcc.Slider(id="thr", min=0.05, max=0.95, step=0.05, value=0.5,
                   marks={i/100: str(i) for i in range(5, 100, 10)}),
        width=8)),
    dbc.Row(dbc.Col(dcc.Graph(id="cm-graph", figure=cm_fig))),
    dbc.Alert(
        "Use the slider to adjust the probability threshold. Lowering the "
        "threshold increases recall (catching more potential cancer cases) at "
        "the cost of precision.", color="info"),
], fluid=True)

fi_tab = dbc.Container([
    dbc.Row(dbc.Col(html.H4("Explainability – Feature Importance",
                            className="text-primary"))),
    dbc.Row(dbc.Col(dcc.Graph(figure=fi_fig))),
    dbc.Alert(
        "XGBoost's gain‑based importance shows which proteins and clinical "
        "features most influence predictions, supporting transparent decision‑"
        "making for clinicians.", color="info"),
], fluid=True)

about_tab = dbc.Container([
    dbc.Row(dbc.Col(html.H4("About AdvanceHC Solution", className="text-primary"))),
    dbc.Row(dbc.Col(html.P(COMPANY))),
    dbc.Row(dbc.Col(html.P(ABSTRACT))),
    dbc.Row(dbc.Col(html.H5("Screenshots & Artefacts", className="mt-4"))),
    dbc.Row([dbc.Col(img, md=6) for img in images]),
], fluid=True)

tabs = dbc.Tabs([
    dbc.Tab(overview_tab, label="Overview"),
    dbc.Tab(dist_tab, label="Distributions"),
    dbc.Tab(perf_tab, label="Performance"),
    dbc.Tab(fi_tab, label="Feature Importance"),
    dbc.Tab(about_tab, label="About"),
])

app.layout = dbc.Container([
    dbc.NavbarSimple(brand="Breast‑Cancer Proteomics Dashboard",
                     brand_href="#", color="primary", dark=True,
                     children=[dbc.NavItem(dbc.NavLink(COMPANY, href="#"))]),
    tabs,
], fluid=True, className="p-0")

# ------------------------------------------------------------------ #
#  Callbacks
# ------------------------------------------------------------------ #
@app.callback(Output("cm-graph", "figure"), Input("thr", "value"))
def _update_cm(th):
    cm_ = confusion_matrix(y_te, (y_prob >= th).astype(int))
    return px.imshow(cm_, text_auto=True,
                     x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                     title=f"Confusion Matrix – Threshold {th:.2f}")

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)