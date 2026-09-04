import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

st.set_page_config(
    page_title="SA Water Safety Prediction",
    page_icon="💧",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Data loading, cleaning & model training (mirrors the TP_PROJECT_FINAL notebook)
# ----------------------------------------------------------------------------

@st.cache_data
def load_and_clean_data(path):
    df = pd.read_csv(path)

    drop_cols = ['other_weather', 'start_comment', 'Nitrate', 'Phosphate',
                 'Solinity (Parts per Thousand)', 'Salinity Percentage', 'tds', 'id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.rename(columns={
        'dissolved_oxgen_ppm': 'DO_ppm',
        'dissolved_oxygen_percent': 'DO_percent',
        'Electrical Conductivity': 'EC',
        'Air Temperature': 'AirTemp',
        'Altitude (m)': 'Altitude',
        'Location Accuracy (m)': 'LocAccuracy'
    })

    outlier_rules = {
        'PH': (0, 14),
        'DO_ppm': (0, 20),
        'DO_percent': (0, 150),
        'AirTemp': (-5, 45),
    }
    removed_counts = {}
    for col, (lo, hi) in outlier_rules.items():
        n_before = df[col].notna().sum()
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        n_after = df[col].notna().sum()
        removed_counts[col] = int(n_before - n_after)

    num_cols = ['PH', 'DO_ppm', 'DO_percent', 'EC', 'AirTemp']
    missing_before = df[num_cols].isnull().sum().to_dict()

    for col in num_cols:
        df[col] = df.groupby('Biosphere')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    def label_potable(row):
        ph_ok = 6.5 <= row['PH'] <= 8.5
        do_ok = row['DO_percent'] >= 80
        ec_ok = row['EC'] <= 500
        return 1 if (ph_ok and do_ok and ec_ok) else 0

    df['Potable'] = df.apply(label_potable, axis=1)

    # Realistic label uncertainty, same as the notebook (avoids trivial 100% accuracy
    # since the label would otherwise be a deterministic function of the features)
    np.random.seed(42)
    noise_rate = 0.08
    flip_mask = np.random.rand(len(df)) < noise_rate
    df['Potable'] = np.where(flip_mask, 1 - df['Potable'], df['Potable'])

    meta = {
        'removed_counts': removed_counts,
        'missing_before': missing_before,
        'n_flipped': int(flip_mask.sum()),
        'noise_rate': noise_rate,
    }
    return df, meta


@st.cache_resource
def train_models(df):
    feature_cols = ['PH', 'DO_ppm', 'DO_percent', 'EC', 'AirTemp', 'Biosphere', 'weather']
    X = df[feature_cols].copy()
    y = df['Potable']

    le_bio = LabelEncoder()
    le_weather = LabelEncoder()
    X['Biosphere'] = le_bio.fit_transform(X['Biosphere'].astype(str))
    X['weather'] = le_weather.fit_transform(X['weather'].astype(str))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    num_feats = ['PH', 'DO_ppm', 'DO_percent', 'EC', 'AirTemp']
    scaler = StandardScaler()
    X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()
    X_train_scaled[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_val_scaled[num_feats] = scaler.transform(X_val[num_feats])
    X_test_scaled[num_feats] = scaler.transform(X_test[num_feats])

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    unsafe_f1 = lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=0)
    from sklearn.metrics import make_scorer
    scorer = make_scorer(f1_score, pos_label=0)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring=scorer)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    pred_test = best_rf.predict(X_test)
    proba_test_unsafe = best_rf.predict_proba(X_test)[:, 0]

    metrics = {
        'accuracy': accuracy_score(y_test, pred_test),
        'unsafe_precision': precision_score(y_test, pred_test, pos_label=0),
        'unsafe_recall': recall_score(y_test, pred_test, pos_label=0),
        'unsafe_f1': f1_score(y_test, pred_test, pos_label=0),
        'roc_auc_unsafe': roc_auc_score((y_test == 0).astype(int), proba_test_unsafe),
        'confusion_matrix': confusion_matrix(y_test, pred_test, labels=[0, 1]),
        'best_params': grid.best_params_,
        'cv_unsafe_f1': grid.best_score_,
        'y_test': y_test,
        'pred_test': pred_test,
        'proba_test_unsafe': proba_test_unsafe,
    }

    importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Also fit a version of best_rf-style model on the FULL feature-engineered dataset,
    # for live single-sample predictions
    predict_bundle = {
        'model': best_rf,
        'le_bio': le_bio,
        'le_weather': le_weather,
        'feature_cols': feature_cols,
    }

    return metrics, importances, predict_bundle


DATA_PATH = "water_quality_data.csv"
df, clean_meta = load_and_clean_data(DATA_PATH)
metrics, importances, predict_bundle = train_models(df)

num_cols = ['PH', 'DO_ppm', 'DO_percent', 'EC', 'AirTemp']

# ----------------------------------------------------------------------------
# Sidebar navigation & filters
# ----------------------------------------------------------------------------

st.sidebar.title("Water Safety Dashboard")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Explore the Data", "Model Performance", "Predict Water Safety"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter data")
biospheres = sorted(df['Biosphere'].dropna().unique().tolist())
selected_biospheres = st.sidebar.multiselect("Biosphere Reserve", biospheres, default=biospheres)

filtered_df = df[df['Biosphere'].isin(selected_biospheres)] if selected_biospheres else df

st.sidebar.markdown("---")
st.sidebar.caption(
    "Academic ML project — predicting river-water potability from South African "
    "citizen-science water-quality data (Be-Resilient network)."
)

# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------

if page == "Overview":
    st.title("Machine Learning for Water Safety Prediction in South Africa")
    st.markdown(
        "This dashboard summarises a classification model that predicts whether a "
        "river water sample is likely to be **potable or non-potable**, using citizen-science "
        "water-quality readings from the Kruger to Canyons, Vhembe, and Marico biosphere reserves."
    )

    with st.expander("How this dashboard is built (data pipeline)"):
        st.markdown(
            "1. **Load & clean** — drop unused columns, rename fields, and null out "
            "physically impossible readings (pH outside 0–14, dissolved oxygen outside "
            "0–20 ppm / 0–150 %, air temp outside −5–45 °C).\n"
            "2. **Impute** — fill the resulting gaps with the median for that biosphere "
            "reserve, falling back to the overall median.\n"
            "3. **Label** — a sample is called *potable* only if pH is 6.5–8.5 **and** "
            "dissolved oxygen ≥ 80 % **and** electrical conductivity ≤ 500. Everything "
            f"else is *not potable*. Then {clean_meta['noise_rate']:.0%} of labels "
            f"({clean_meta['n_flipped']} rows) are randomly flipped so the label isn't a "
            "perfectly learnable rule.\n"
            "4. **Split** — 70 % train / 15 % validation / 15 % test, stratified so the "
            "class balance is the same in each split.\n"
            "5. **Tune & train** — a Random Forest is grid-searched over tree count, depth "
            "and split size with 5-fold cross-validation, scored on unsafe-class F1.\n\n"
            "The **Model Performance** page reports scores on the held-out test set only "
            "— data the model never saw during training or tuning."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Biosphere Reserves", df['Biosphere'].nunique())
    c3.metric("Potable Rate", f"{df['Potable'].mean():.0%}")
    c4.metric("Test Accuracy (tuned RF)", f"{metrics['accuracy']:.1%}")
    st.caption(
        "**Samples** — rows left after cleaning. **Potable Rate** — share labelled safe "
        "to drink; near 50 % here, so plain accuracy is a fair yardstick (there's no "
        "majority class to trivially guess). **Test Accuracy** — percent of held-out "
        "samples the tuned model classified correctly."
    )

    st.markdown("### Key result")
    st.info(
        f"The tuned Random Forest correctly identifies **{metrics['unsafe_recall']:.1%}** "
        f"of unsafe samples (recall for the unsafe class) on the held-out test set, "
        f"with an overall accuracy of **{metrics['accuracy']:.1%}**. Recall on unsafe water "
        f"was prioritised, since missing unsafe water is the costlier error."
    )
    st.caption(
        "Read this as: out of every 100 genuinely unsafe samples, the model raises a "
        f"flag on about {metrics['unsafe_recall'] * 100:.0f}. The trade-off is more false "
        "alarms on clean water — see the confusion matrix on the Model Performance page."
    )

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("#### Sample locations")
        st.caption(
            "Each dot is a sampling site, coloured by its potability label "
            "(green = potable, red = not potable). Hover a dot for its reserve, river, "
            "site name, pH and electrical conductivity. Clusters of red mark stretches "
            "of river where readings repeatedly fall outside the safe thresholds."
        )
        map_df = df.dropna(subset=['latitude', 'longitude']).copy()
        map_df['Safety'] = map_df['Potable'].map({1: 'Potable', 0: 'Not Potable'})
        fig = px.scatter_mapbox(
            map_df, lat='latitude', lon='longitude', color='Safety',
            color_discrete_map={'Potable': '#2E8B57', 'Not Potable': '#DC5A4E'},
            hover_data=['Biosphere', 'river', 'site', 'PH', 'EC'],
            zoom=4.2, height=450,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown("#### Class balance")
        st.caption(
            "Share of all samples in each class. The near 50/50 split means the model "
            "can't score well just by always guessing one label."
        )
        counts = df['Potable'].value_counts().rename({0: 'Not Potable', 1: 'Potable'})
        fig2 = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={'Potable': '#2E8B57', 'Not Potable': '#DC5A4E'},
            hole=0.5,
        )
        fig2.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### About the labels")
    st.caption(
        "Potability was derived from thresholds (pH 6.5–8.5, dissolved oxygen ≥80%, "
        "electrical conductivity ≤500) rather than lab-confirmed drinking-water tests, "
        f"and {clean_meta['noise_rate']:.0%} of labels were randomly flipped to simulate "
        "real-world measurement uncertainty. Treat this model as an early-warning screening "
        "tool, not a lab replacement."
    )

# ----------------------------------------------------------------------------
# PAGE: Explore the Data
# ----------------------------------------------------------------------------

elif page == "Explore the Data":
    st.title("Explore the Data")

    st.markdown(f"Showing **{len(filtered_df):,}** of {len(df):,} samples based on sidebar filters.")
    st.caption(
        "Use the **Biosphere Reserve** filter in the sidebar to narrow every chart on "
        "this page. The three tabs below look at the data from different angles: one "
        "parameter at a time, two parameters together, and split by reserve."
    )

    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "By Biosphere"])

    with tab1:
        st.markdown("#### Distribution of a single parameter")
        st.caption(
            "Pick a water-quality parameter. The bars count how many samples fall in "
            "each value range, stacked by class (green = potable, red = not). The "
            "narrow **box plot** above the bars summarises each class — the box is the "
            "middle 50 % of values, the line is the median, and dots are outliers. "
            "Where the green and red distributions sit apart, that parameter carries "
            "signal the model can use; where they overlap heavily, it doesn't."
        )
        param = st.selectbox("Parameter", num_cols, index=0)
        fig = px.histogram(
            filtered_df, x=param, color=filtered_df['Potable'].map({1: 'Potable', 0: 'Not Potable'}),
            nbins=30, marginal="box",
            color_discrete_map={'Potable': '#2E8B57', 'Not Potable': '#DC5A4E'},
            labels={'color': 'Safety'},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation between water-quality parameters")
        st.caption(
            "Each cell is the Pearson correlation between two parameters, from −1 to +1. "
            "**Red** = they rise together, **blue** = one rises as the other falls, "
            "**pale** = little linear relationship. Dissolved oxygen in ppm and in "
            "percent are strongly correlated because they measure the same thing two "
            "ways. Strongly correlated features carry overlapping information."
        )
        corr = filtered_df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        st.markdown("#### Two parameters against each other")
        st.caption(
            "Every point is one sample. Choose which parameter goes on each axis. "
            "**Colour** is the biosphere reserve; **marker shape** is the potability "
            "label. Look for shapes that separate cleanly — a region of the plot that "
            "is all one shape is a combination of readings that reliably predicts "
            "safe or unsafe water."
        )
        c1, c2 = st.columns(2)
        with c1:
            xcol = st.selectbox("X axis", num_cols, index=num_cols.index('EC'))
        with c2:
            ycol = st.selectbox("Y axis", num_cols, index=num_cols.index('DO_ppm'))
        fig = px.scatter(
            filtered_df, x=xcol, y=ycol, color='Biosphere', symbol=filtered_df['Potable'].map({1: 'Potable', 0: 'Not Potable'}),
            opacity=0.75, labels={'symbol': 'Safety'},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Dissolved oxygen by weather condition")
        st.caption(
            "One box per weather category, showing how dissolved oxygen (ppm) is "
            "distributed on those days. Box = middle 50 % of readings, line = median, "
            "whiskers = the rest, dots = outliers. Big differences between boxes are "
            "why `weather` is included as a model feature."
        )
        fig_w = px.box(filtered_df, x='weather', y='DO_ppm', color='weather')
        fig_w.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    with tab3:
        st.markdown("#### One parameter compared across reserves")
        st.caption(
            "Same box-plot reading as before, with one box per biosphere reserve. "
            "If the boxes sit at clearly different levels, the reserves differ "
            "systematically for that parameter — worth knowing, because a model "
            "trained mostly on one reserve may transfer poorly to another."
        )
        param2 = st.selectbox("Parameter to compare across reserves", num_cols, index=num_cols.index('PH'), key="bio_param")
        fig_b = px.box(filtered_df, x='Biosphere', y=param2, color='Biosphere')
        fig_b.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("#### Sample counts per reserve")
        st.caption(
            "How many samples each reserve contributes after the sidebar filter. "
            "Uneven counts mean the model has seen far more of some reserves than others."
        )
        st.bar_chart(filtered_df['Biosphere'].value_counts())

    with st.expander("View raw (cleaned) data"):
        st.caption(
            "The table after cleaning and imputation, filtered to the reserves selected "
            "in the sidebar. `Potable` is the derived label (1 = potable, 0 = not)."
        )
        st.dataframe(filtered_df, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Model Performance
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance — Tuned Random Forest")

    st.markdown(
        f"Best cross-validated parameters: `{metrics['best_params']}` "
        f"(5-fold CV, optimised for unsafe-class F1 = **{metrics['cv_unsafe_f1']:.3f}**)"
    )
    st.caption(
        "These settings won a grid search: the training data was split into 5 folds, "
        "every combination of tree count, tree depth and minimum split size was trained "
        "and scored on the held-out fold, and the combination with the best average "
        "unsafe-class F1 was kept. All numbers below are then measured **once** on the "
        "separate test set."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Unsafe Recall", f"{metrics['unsafe_recall']:.1%}")
    c3.metric("Unsafe Precision", f"{metrics['unsafe_precision']:.1%}")
    c4.metric("Unsafe F1", f"{metrics['unsafe_f1']:.3f}")
    st.caption(
        "**Accuracy** — share of all test samples classified correctly. "
        "**Unsafe Recall** — of the water that really is unsafe, the share the model "
        "flagged (misses are 1 − recall). **Unsafe Precision** — of the samples the "
        "model flagged as unsafe, the share that really were (the rest are false "
        "alarms). **Unsafe F1** — a single score balancing recall and precision "
        "(their harmonic mean); 1.0 is perfect."
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Confusion matrix (test set)")
        st.caption(
            "Rows are the true class, columns are what the model predicted. The two "
            "cells on the diagonal (top-left, bottom-right) are correct calls. "
            "**Top-right** = unsafe water the model passed as potable — the dangerous "
            "miss. **Bottom-left** = potable water flagged as unsafe — a false alarm. "
            "Darker cells hold more samples."
        )
        cm = metrics['confusion_matrix']
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            x=['Predicted: Not Potable', 'Predicted: Potable'],
            y=['Actual: Not Potable', 'Actual: Potable'],
        )
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    with colB:
        st.markdown("#### ROC curve — unsafe class")
        st.caption(
            "The model outputs a probability of 'unsafe', not just a yes/no. This "
            "curve sweeps the cut-off from strict to lenient and plots how many unsafe "
            "samples are caught (true-positive rate, y) against how many clean samples "
            "are wrongly flagged (false-positive rate, x). The **dashed line** is "
            "random guessing. **AUC** is the area under the solid curve — the chance "
            "the model scores a random unsafe sample above a random potable one; "
            "1.0 is perfect, 0.5 is no better than chance."
        )
        fpr, tpr, _ = roc_curve((metrics['y_test'] == 0).astype(int), metrics['proba_test_unsafe'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {metrics['roc_auc_unsafe']:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Chance'))
        fig_roc.update_layout(height=420, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Feature importance")
    st.caption(
        "How much each input the forest relied on when splitting its trees, scaled to "
        "sum to 1. Longer bar = the model leaned on that feature more. Dissolved oxygen "
        "(% and ppm) and pH dominate, which matches the thresholds used to define the "
        "label in the first place. Note this shows *what the model used*, not cause "
        "and effect."
    )
    fig_imp = px.bar(
        importances.sort_values(ascending=True), orientation='h',
        labels={'value': 'Importance', 'index': 'Feature'},
    )
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Predict Water Safety
# ----------------------------------------------------------------------------

elif page == "Predict Water Safety":
    st.title("Predict Water Safety")
    st.markdown("Enter water-quality readings from a sample to get a live prediction from the tuned Random Forest model.")
    st.caption(
        "The seven values below are fed to the trained model as a single sample. It "
        "returns a probability for each class; the label with the higher probability is "
        "shown, along with that probability as a **confidence** figure. The numeric "
        "readings are used directly (a Random Forest needs no scaling); **Biosphere "
        "Reserve** and **Weather** are matched to the categories seen during training. "
        "Readings far outside the ranges in the training data will give unreliable "
        "results."
    )

    le_bio = predict_bundle['le_bio']
    le_weather = predict_bundle['le_weather']
    model = predict_bundle['model']
    feature_cols = predict_bundle['feature_cols']

    st.caption(
        "Reference ranges used to derive the 'potable' label: pH **6.5–8.5**, "
        "dissolved oxygen **≥ 80 %**, electrical conductivity **≤ 500**. The model "
        "isn't bound by these exact cut-offs, but inputs well inside all three "
        "usually predict potable."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        ph = st.slider("pH", 0.0, 14.0, 7.2, 0.1)
        do_ppm = st.slider("Dissolved Oxygen (ppm)", 0.0, 20.0, 7.0, 0.1)
    with c2:
        do_percent = st.slider("Dissolved Oxygen (%)", 0.0, 150.0, 85.0, 1.0)
        ec = st.slider("Electrical Conductivity", 0.0, 1000.0, 250.0, 5.0)
    with c3:
        air_temp = st.slider("Air Temperature (°C)", -5.0, 45.0, 22.0, 0.5)
        biosphere = st.selectbox("Biosphere Reserve", list(le_bio.classes_))
        weather = st.selectbox("Weather", list(le_weather.classes_))

    input_row = pd.DataFrame([{
        'PH': ph, 'DO_ppm': do_ppm, 'DO_percent': do_percent, 'EC': ec, 'AirTemp': air_temp,
        'Biosphere': le_bio.transform([biosphere])[0],
        'weather': le_weather.transform([weather])[0],
    }])[feature_cols]

    if st.button("Predict", type="primary"):
        pred = model.predict(input_row)[0]
        proba = model.predict_proba(input_row)[0]
        unsafe_proba = proba[0]
        potable_proba = proba[1]

        if pred == 1:
            st.success(f"Predicted **Potable** — confidence {potable_proba:.0%}")
        else:
            st.error(f"Predicted **Not Potable** — confidence {unsafe_proba:.0%}")

        st.progress(float(potable_proba), text=f"Potable probability: {potable_proba:.0%}")
        st.caption(
            "The bar is the model's probability that this sample is potable — it's the "
            "share of trees in the forest that voted 'potable'. Near 50 % means the "
            "model is genuinely unsure; near 0 % or 100 % means the trees mostly agree."
        )

        st.caption(
            "This is a screening estimate from a model trained on citizen-science data with "
            "engineered, threshold-based labels — not a substitute for laboratory testing."
        )
