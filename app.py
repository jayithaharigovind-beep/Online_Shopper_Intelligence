import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Shopper Intelligence",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #1f4e79 0%,
            #2878b5 50%,
            #5b9bd5 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #dfe6ee;
        margin-top: 1rem;
    }

    .probability {
        font-size: 2.7rem;
        font-weight: 800;
    }

    .small-note {
        color: #667085;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")


try:
    df = load_data()
except Exception as e:
    st.error(
        "Unable to load the main dataset. Please check that "
        "`data/online_shoppers_intention.csv` exists."
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "online_shoppers_rf_pipeline.pkl"
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the Random Forest model. "
        "Please check that the model exists at "
        "`models/online_shoppers_rf_pipeline.pkl`."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🛒 Online Shopper Intelligence")

st.write(
    "Predict visitor purchase intention using machine learning "
    "and understand the behavioural signals behind online conversion."
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Executive Overview",
        "🔮 Purchase Prediction",
        "📂 Dataset Diagnostics",
        "🤖 Model Performance",
        "🔍 Model Explainability",
        "💡 Business Insights"
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tab1:

    st.subheader("📊 Executive Overview")
    st.write(
        "A high-level view of the dataset, conversion behaviour and "
        "the purpose of the prediction model."
    )

    total_sessions = len(df)
    purchases = int(df["Revenue"].sum())
    non_purchases = total_sessions - purchases
    purchase_rate = df["Revenue"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Sessions", f"{total_sessions:,}")

    with c2:
        st.metric("Purchases", f"{purchases:,}")

    with c3:
        st.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    with c4:
        st.metric("Non-Purchase Rate", f"{100-purchase_rate:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Purchase Distribution")
        distribution = pd.Series(
            {
                "Non-Purchase": non_purchases,
                "Purchase": purchases
            }
        )
        st.bar_chart(distribution)

    with col2:
        st.markdown("### Dataset Snapshot")
        snapshot = pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Missing Values",
                    "Duplicate Rows",
                    "Target Variable"
                ],
                "Value": [
                    f"{len(df):,}",
                    f"{df.shape[1]}",
                    f"{int(df.isnull().sum().sum()):,}",
                    f"{int(df.duplicated().sum()):,}",
                    "Revenue"
                ]
            }
        )
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Purchase Rate by Visitor Type")
        visitor_rate = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        st.bar_chart(visitor_rate)

    with col2:
        st.markdown("### Purchase Rate by Month")
        month_rate = (
            df.groupby("Month")["Revenue"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        st.bar_chart(month_rate)

    st.info(
        f"""
        **Project objective:** predict whether an online visitor is likely
        to make a purchase based on browsing behaviour, session context and
        visitor characteristics.

        The dataset contains **{total_sessions:,} sessions**, but only
        **{purchase_rate:.2f}%** result in a purchase. This class imbalance
        makes recall and F1-score important in addition to accuracy.
        """
    )


# ============================================================
# TAB 2 — PURCHASE PREDICTION
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Visitor Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        VisitorType = st.selectbox(
            "Visitor Type",
            [
                "Returning_Visitor",
                "New_Visitor",
                "Other"
            ]
        )

    with col2:

        Month = st.selectbox(
            "Month",
            [
                "Feb",
                "Mar",
                "May",
                "June",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec"
            ]
        )

    with col3:

        Weekend = st.selectbox(
            "Weekend Visit",
            [
                False,
                True
            ],
            format_func=lambda x:
                "Yes" if x else "No"
        )


    # --------------------------------------------------------
    # BROWSING BEHAVIOUR
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Browsing Behaviour</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        Administrative = st.number_input(
            "Administrative Pages",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    with col2:

        Informational = st.number_input(
            "Informational Pages",
            min_value=0,
            max_value=100,
            value=2,
            step=1
        )

    with col3:

        ProductRelated = st.number_input(
            "Product-Related Pages",
            min_value=0,
            max_value=200,
            value=20,
            step=1
        )


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    TotalPages = (
        Administrative
        + Informational
        + ProductRelated
    )

    st.info(
        f"📌 **Total Pages Viewed:** {TotalPages} "
        "(calculated automatically from the three page categories)"
    )


    # --------------------------------------------------------
    # TIME SPENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Time Spent on Website</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        Administrative_Duration = st.number_input(
            "Administrative Time",
            min_value=0.0,
            value=70.0,
            step=10.0
        )

    with col2:

        Informational_Duration = st.number_input(
            "Informational Time",
            min_value=0.0,
            value=20.0,
            step=10.0
        )

    with col3:

        ProductRelated_Duration = st.number_input(
            "Product-Related Time",
            min_value=0.0,
            value=500.0,
            step=50.0
        )


    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Engagement Metrics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        BounceRates = st.number_input(
            "Bounce Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01
        )

    with col2:

        ExitRates = st.number_input(
            "Exit Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01
        )

    with col3:

        SpecialDay = st.selectbox(
            "Special Day",
            [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0
            ]
        )


    # --------------------------------------------------------
    # TECHNICAL / TRAFFIC INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Traffic & Technology</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        OperatingSystems = st.selectbox(
            "Operating System",
            list(range(1, 9))
        )

    with col2:

        Browser = st.selectbox(
            "Browser",
            list(range(1, 14))
        )

    with col3:

        Region = st.selectbox(
            "Region",
            list(range(1, 10))
        )

    with col4:

        TrafficType = st.selectbox(
            "Traffic Type",
            list(range(1, 21))
        )


    st.markdown("---")


    # ========================================================
    # PREDICTION
    # ========================================================

    predict_button = st.button(
        "🔮 Predict Purchase Probability",
        use_container_width=True,
        type="primary"
    )


    if predict_button:

        input_data = pd.DataFrame(
            {
                "Administrative":
                    [Administrative],

                "Administrative_Duration":
                    [Administrative_Duration],

                "Informational":
                    [Informational],

                "Informational_Duration":
                    [Informational_Duration],

                "ProductRelated":
                    [ProductRelated],

                "ProductRelated_Duration":
                    [ProductRelated_Duration],

                "TotalPages":
                    [TotalPages],

                "BounceRates":
                    [BounceRates],

                "ExitRates":
                    [ExitRates],

                "SpecialDay":
                    [SpecialDay],

                "Month":
                    [Month],

                "OperatingSystems":
                    [OperatingSystems],

                "Browser":
                    [Browser],

                "Region":
                    [Region],

                "TrafficType":
                    [TrafficType],

                "VisitorType":
                    [VisitorType],

                "Weekend":
                    [Weekend]
            }
        )


        try:

            probability = (
                model.predict_proba(input_data)[0][1]
            )

            # Use probability bands for the dashboard classification so
            # the main result and the business interpretation are consistent.
            if probability >= 0.70:
                intent_label = "HIGH PURCHASE POTENTIAL"
                intent_icon = "🟢"
            elif probability >= 0.40:
                intent_label = "MODERATE PURCHASE POTENTIAL"
                intent_icon = "🟡"
            else:
                intent_label = "LOW PURCHASE POTENTIAL"
                intent_icon = "🔴"


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Prediction Result"
            )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Purchase Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:

                if probability >= 0.70:
                    st.success(f"{intent_icon} {intent_label}")
                elif probability >= 0.40:
                    st.info(f"{intent_icon} {intent_label}")
                else:
                    st.warning(f"{intent_icon} {intent_label}")


            st.progress(
                float(probability)
            )


            st.write(
                f"The visitor shows **{intent_label.lower()}** based on "
                "the estimated purchase probability."
            )


            # ------------------------------------------------
            # BUSINESS INTERPRETATION
            # ------------------------------------------------

            st.markdown(
                "### 💡 Business Interpretation"
            )

            if probability >= 0.70:

                st.success(
                    "High purchase intent. "
                    "This visitor could be prioritised for "
                    "conversion-oriented communication."
                )

            elif probability >= 0.40:

                st.info(
                    "Moderate purchase intent. "
                    "This visitor may benefit from additional "
                    "product information, offers or retargeting."
                )

            else:

                st.warning(
                    "Low purchase intent. "
                    "The visitor may require stronger engagement "
                    "before being targeted for conversion."
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# TAB 3 — DATASET DIAGNOSTICS
# ============================================================

with tab3:

    st.subheader(
        "📊 Upload a Dataset for Diagnostics"
    )

    st.write(
        "Upload a CSV file to examine its structure, "
        "basic statistics and purchase behaviour."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )


            st.success(
                "Dataset uploaded successfully."
            )


            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Rows",
                    uploaded_df.shape[0]
                )

            with col2:

                st.metric(
                    "Columns",
                    uploaded_df.shape[1]
                )

            with col3:

                st.metric(
                    "Missing Values",
                    int(uploaded_df.isnull().sum().sum())
                )

            with col4:

                duplicate_count = (
                    uploaded_df.duplicated().sum()
                )

                st.metric(
                    "Duplicate Rows",
                    int(duplicate_count)
                )


            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )


            # ------------------------------------------------
            # DATA TYPES
            # ------------------------------------------------

            st.subheader(
                "Data Types"
            )

            dtype_df = pd.DataFrame(
                {
                    "Feature":
                        uploaded_df.columns,

                    "Data Type":
                        uploaded_df.dtypes.astype(str).values,

                    "Missing Values":
                        uploaded_df.isnull().sum().values
                }
            )

            st.dataframe(
                dtype_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # NUMERICAL SUMMARY
            # ------------------------------------------------

            st.subheader(
                "Numerical Summary"
            )

            st.dataframe(
                uploaded_df.describe().T,
                use_container_width=True
            )


            # ------------------------------------------------
            # ENGINEER TOTALPAGES IF POSSIBLE
            # ------------------------------------------------

            page_columns = [
                "Administrative",
                "Informational",
                "ProductRelated"
            ]

            if all(
                col in uploaded_df.columns
                for col in page_columns
            ):

                uploaded_df["TotalPages"] = (
                    uploaded_df["Administrative"]
                    + uploaded_df["Informational"]
                    + uploaded_df["ProductRelated"]
                )

                st.subheader(
                    "Feature Engineering"
                )

                st.success(
                    "TotalPages was calculated automatically "
                    "from Administrative + Informational + ProductRelated."
                )

                st.metric(
                    "Average Total Pages",
                    f"{uploaded_df['TotalPages'].mean():.2f}"
                )


            # ------------------------------------------------
            # TARGET ANALYSIS
            # ------------------------------------------------

            if "Revenue" in uploaded_df.columns:

                st.subheader(
                    "Purchase Outcome Distribution"
                )

                revenue_counts = (
                    uploaded_df["Revenue"]
                    .value_counts()
                )

                st.bar_chart(
                    revenue_counts
                )


                purchase_rate = (
                    uploaded_df["Revenue"]
                    .astype(int)
                    .mean()
                    * 100
                )


                st.metric(
                    "Purchase Rate",
                    f"{purchase_rate:.2f}%"
                )


            else:

                st.info(
                    "Revenue column was not found, "
                    "so purchase-outcome analysis is unavailable."
                )


        except Exception as e:

            st.error(
                f"Unable to process the uploaded dataset: {e}"
            )


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab4:

    st.subheader(
        "🤖 Model Comparison"
    )

    st.write(
        "Performance of the classification models evaluated "
        "for online purchase intention prediction."
    )


    model_results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "AdaBoost",
                "Gradient Boosting",
                "Random Forest"
            ],

            "Accuracy": [
                0.6298,
                0.6395,
                0.8447,
                0.8427,
                0.7064
            ],

            "Precision": [
                0.2593,
                0.2593,
                0.0000,
                0.4483,
                0.2950
            ],

            "Recall": [
                0.7487,
                0.7147,
                0.0000,
                0.0681,
                0.6440
            ],

            "F1 Score": [
                0.3852,
                0.3805,
                0.0000,
                0.1182,
                0.4046
            ],

            "ROC-AUC": [
                0.7347,
                0.7319,
                0.7345,
                0.7595,
                0.7522
            ]
        }
    )


    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "ROC-AUC": "{:.2%}"
            }
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # MODEL METRIC CHART
    # --------------------------------------------------------

    st.subheader(
        "Performance Overview"
    )

    chart_df = model_results.set_index(
        "Model"
    )

    st.bar_chart(
        chart_df[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
        ]
    )


    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "⭐ Selected Model: Random Forest"
    )

    st.write(
        """
        Random Forest was selected as the final model because it
        provides a stronger balance between identifying purchasers
        and controlling false predictions.

        While AdaBoost and Gradient Boosting achieve higher accuracy,
        their purchaser recall is extremely low. Random Forest achieves
        a recall of 64.40% and the highest F1 score among the evaluated
        models, making it more suitable for identifying potential
        purchasers.
        """
    )


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Feature Engineering"
    )

    st.write(
        """
        A new engineered feature called **TotalPages** was created:

        **TotalPages = Administrative + Informational + ProductRelated**

        This combines the three page-count variables into a single
        measure of overall browsing activity.

        The Random Forest was retrained using this feature. The
        resulting model achieved approximately 70.64% accuracy,
        29.50% precision, 64.40% recall, 40.46% F1 score and
        75.22% ROC-AUC.
        """
    )


    # --------------------------------------------------------
    # BUSINESS TAKEAWAY
    # --------------------------------------------------------

    st.subheader(
        "💼 Business Takeaway"
    )

    st.info(
        """
        The objective is not simply to maximise accuracy.
        For an e-commerce business, identifying potential purchasers
        is important because these visitors can be prioritised for
        targeted communication, personalised offers and retargeting.

        Therefore, recall and F1 score are important alongside
        accuracy and ROC-AUC.
        """
    )

# ============================================================
# TAB 5 — MODEL EXPLAINABILITY
# ============================================================

with tab5:

    st.markdown("## 🔍 Model Explainability")

    st.caption(
        "Understanding which variables are most influential in the "
        "Random Forest model."
    )

    try:

        rf_classifier = model.named_steps["classifier"]
        preprocessor_fitted = model.named_steps["preprocessor"]

        feature_names = (
            preprocessor_fitted.get_feature_names_out()
        )

        importances = rf_classifier.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(
            by="Importance",
            ascending=False
        )

        feature_importance_df["Display Feature"] = (
            feature_importance_df["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        top10 = feature_importance_df.head(10).copy()

        top10["Importance %"] = (
            top10["Importance"] * 100
        ).round(2)

        # TOP THREE
        c1, c2, c3 = st.columns(3)

        for i, column in enumerate([c1, c2, c3]):

            if i < len(top10):

                row = top10.iloc[i]

                with column:

                    st.metric(
                        f"#{i + 1} {row['Display Feature']}",
                        f"{row['Importance %']:.2f}%"
                    )

        st.divider()

        st.markdown("### Top 10 Predictive Features")

        chart_data = (
            top10[
                ["Display Feature", "Importance"]
            ]
            .set_index("Display Feature")
            .sort_values("Importance")
        )

        st.bar_chart(chart_data)

        st.markdown("### Feature Importance Ranking")

        ranking_table = top10[
            ["Display Feature", "Importance %"]
        ].rename(
            columns={
                "Display Feature": "Feature",
                "Importance %": "Importance"
            }
        )

        st.dataframe(
            ranking_table,
            use_container_width=True,
            hide_index=True
        )

        totalpages_row = feature_importance_df[
            feature_importance_df["Display Feature"]
            .str.lower()
            .eq("totalpages")
        ]

        if not totalpages_row.empty:

            totalpages_importance = (
                float(
                    totalpages_row.iloc[0]["Importance"]
                ) * 100
            )

            totalpages_rank = (
                feature_importance_df
                .reset_index(drop=True)
                .index[
                    feature_importance_df[
                        "Display Feature"
                    ].str.lower()
                    .eq("totalpages")
                ][0]
                + 1
            )

            st.success(
                f"""
                **Feature Engineering Result:** TotalPages has an
                importance of **{totalpages_importance:.2f}%** and ranks
                **#{totalpages_rank}** among the model's predictors.

                TotalPages combines Administrative, Informational and
                ProductRelated page counts into one measure of overall
                browsing activity.
                """
            )

        st.markdown("### 🧠 Interpretation")

        st.info(
            """
            The most influential variables are primarily related to
            visitor engagement and browsing behaviour.

            **ExitRates** is the strongest predictor in this model,
            followed by **ProductRelated_Duration** and **BounceRates**.

            The engineered **TotalPages** feature is also among the
            strongest predictors.

            Feature importance indicates predictive influence; it does
            not establish that a feature causes a purchase.
            """
        )

    except Exception as e:

        st.error(
            "Feature importance could not be generated from the saved model."
        )

        st.code(str(e))


# ============================================================
# TAB 6 — BUSINESS INSIGHTS
# ============================================================

with tab6:

    st.markdown("## 💡 Business Insights")

    st.caption(
        "Translating behavioural patterns and model outputs into "
        "marketing actions."
    )

    st.markdown("### 👤 Purchase Rate by Visitor Type")

    visitor_insight = (
        df.groupby("VisitorType")["Revenue"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(visitor_insight)

    st.markdown("### 📅 Purchase Rate by Month")

    monthly_insight = (
        df.groupby("Month")["Revenue"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(monthly_insight)

    st.markdown("### 🗓️ Weekday vs Weekend Purchase Rate")

    weekend_insight = (
        df.groupby("Weekend")["Revenue"]
        .mean()
        .mul(100)
    )

    weekend_insight.index = [
        "Weekend" if value else "Weekday"
        for value in weekend_insight.index
    ]

    st.bar_chart(weekend_insight)

    st.divider()

    st.markdown("### 💼 Key Business Findings")

    findings = [
        (
            "Conversion opportunity",
            f"Only {df['Revenue'].mean():.2%} of sessions result "
            "in a purchase, leaving a large pool of non-converting visitors."
        ),
        (
            "Browsing behaviour matters",
            "Page activity and time spent on product-related content "
            "provide useful signals of purchase intention."
        ),
        (
            "Exit behaviour matters",
            "ExitRates is the strongest feature in the Random Forest, "
            "making visitor drop-off an important behavioural signal."
        ),
        (
            "Feature engineering adds value",
            "TotalPages provides a consolidated measure of browsing "
            "activity and ranks among the model's strongest predictors."
        ),
        (
            "Visitor segments differ",
            "Purchase rates vary across visitor types, supporting "
            "differentiated engagement strategies."
        ),
        (
            "Probability enables prioritisation",
            "A purchase-probability score allows marketers to rank "
            "visitors rather than relying only on a binary prediction."
        )
    ]

    for title, description in findings:

        st.markdown(f"**{title}**")
        st.write(description)

    st.divider()

    st.info(
        """
        **Overall business implication:** The model can support an
        e-commerce business by helping prioritise relatively high-potential
        visitors for targeted engagement while reducing unnecessary
        conversion-focused spending on low-potential sessions.
        """
    )
