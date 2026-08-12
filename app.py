import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

with open("deploy_pipeline.pkl", "rb") as f:
    artifacts = pickle.load(f)


# ============================================================
# PREPROCESS ONE SAMPLE
# ============================================================

def preprocess_single_sample(raw_data):

    # --------------------------------------------------------
    # 1. RAW INPUT
    # --------------------------------------------------------

    df = pd.DataFrame([raw_data])

    print("\n================ RAW INPUT ================")
    print(df)
    print("===========================================\n")


    # --------------------------------------------------------
    # 2. LOG TRANSFORMATION
    # SAME AS TRAINING
    # --------------------------------------------------------

    skewed_cols = artifacts["skewed_cols"]

    for col in skewed_cols:

        if col in df.columns:

            df[col] = np.log1p(
                df[col].clip(lower=0)
            )


    # --------------------------------------------------------
    # 3. IQR CLIPPING
    # SAME BOUNDS FROM TRAINING
    # --------------------------------------------------------

    iqr_bounds = artifacts["iqr_bounds"]

    for col, bounds in iqr_bounds.items():

        if col in df.columns:

            lower_bound, upper_bound = bounds

            df[col] = df[col].clip(
                lower=lower_bound,
                upper=upper_bound
            )


    # --------------------------------------------------------
    # 4. TARGET ENCODING
    # --------------------------------------------------------

    target_maps = artifacts["target_maps"]

    for col, mapping in target_maps.items():

        if col in df.columns:

            df[f"mean_{col}"] = (
                df[col].map(mapping)
            )

            # Unknown category
            if df[f"mean_{col}"].isna().any():

                values = list(mapping.values())

                fallback = (
                    np.mean(values)
                    if len(values) > 0
                    else 0.5
                )

                df[f"mean_{col}"] = (
                    df[f"mean_{col}"]
                    .fillna(fallback)
                )


    # --------------------------------------------------------
    # 5. FREQUENCY ENCODING
    # --------------------------------------------------------

    freq_maps = artifacts["freq_maps"]

    for col, mapping in freq_maps.items():

        if col in df.columns:

            df[f"{col}_freq"] = (
                df[col].map(mapping)
            )

            # Unknown value
            if df[f"{col}_freq"].isna().any():

                values = list(mapping.values())

                fallback = (
                    np.mean(values)
                    if len(values) > 0
                    else 0
                )

                df[f"{col}_freq"] = (
                    df[f"{col}_freq"]
                    .fillna(fallback)
                )


    # --------------------------------------------------------
    # 6. QUANTILE BINNING
    # EXACT TRAINING BIN EDGES
    # --------------------------------------------------------

    bin_edges = artifacts["bin_edges"]

    for col, q_dict in bin_edges.items():

        if col not in df.columns:
            continue

        for q, bins in q_dict.items():

            feature_name = f"{col}_bin{q}"

            if bins is None:

                df[feature_name] = 0

            else:

                df[feature_name] = pd.cut(
                    df[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )

                df[feature_name] = (
                    df[feature_name]
                    .fillna(0)
                )


    # --------------------------------------------------------
    # 7. FEATURE ENGINEERING
    # EXACT SAME AS TRAINING
    # --------------------------------------------------------

    df["loan_to_income"] = (
        df["loan_amount"] /
        (df["annual_income"] + 1)
    )

    df["total_debt"] = (
        df["debt_to_income_ratio"] *
        df["annual_income"]
    )

    df["available_income"] = (
        df["annual_income"] *
        (1 - df["debt_to_income_ratio"])
    )

    df["affordability"] = (
        df["available_income"] /
        (df["loan_amount"] + 1)
    )

    df["monthly_payment"] = (
        df["loan_amount"] *
        (1 + df["interest_rate"] / 100) /
        12
    )

    df["payment_to_income"] = (
        df["monthly_payment"] /
        (df["annual_income"] / 12 + 1)
    )

    df["risk_score"] = (
        df["debt_to_income_ratio"] * 40
        +
        (1 - df["credit_score"] / 850) * 30
        +
        df["interest_rate"] * 2
    )

    df["credit_interest"] = (
        df["credit_score"] *
        df["interest_rate"] / 100
    )

    df["income_credit"] = (
        np.log1p(df["annual_income"]) *
        df["credit_score"] / 1000
    )

    df["debt_loan"] = (
        df["debt_to_income_ratio"] *
        np.log1p(df["loan_amount"])
    )

    df["log_income"] = np.log1p(
        df["annual_income"]
    )

    df["log_loan"] = np.log1p(
        df["loan_amount"]
    )

    df["income_to_dti"] = (
        df["annual_income"] /
        (1 + df["debt_to_income_ratio"])
    )

    df["interest_to_score"] = (
        df["interest_rate"] /
        df["credit_score"]
    )

    df["loan_per_score"] = (
        df["loan_amount"] /
        df["credit_score"]
    )

    df["loan_to_dti"] = (
        df["loan_amount"] /
        (1 + df["debt_to_income_ratio"])
    )


    # --------------------------------------------------------
    # 8. CREDIT SCORE TIERS
    # --------------------------------------------------------

    def map_fico_tier(score):

        if score >= 800:
            return "Exceptional"

        elif score >= 740:
            return "Very Good"

        elif score >= 670:
            return "Good"

        elif score >= 580:
            return "Fair"

        else:
            return "Poor"


    def map_vantage_tier(score):

        if score >= 781:
            return "Excellent"

        elif score >= 661:
            return "Good"

        elif score >= 601:
            return "Fair"

        elif score >= 500:
            return "Poor"

        else:
            return "Very Poor"


    df["credit_score_FICO_tier"] = (
        df["credit_score"].apply(map_fico_tier)
    )

    df["credit_score_Vantage_tier"] = (
        df["credit_score"].apply(map_vantage_tier)
    )


    # --------------------------------------------------------
    # 9. GRADE / SUBGRADE
    # SAME AS TRAINING
    # --------------------------------------------------------

    if "grade_subgrade" in df.columns:

        df["grade"] = (
            df["grade_subgrade"]
            .str[0]
        )

        df["subgrade"] = (
            df["grade_subgrade"]
            .str[1:]
            .astype(int)
        )

        grade_order = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6
        }

        df["grade"] = (
            df["grade"].map(grade_order)
        )

        df["credit_rank"] = (
            df["grade"] * 10 +
            df["subgrade"]
        )

        df.drop(
            columns=["grade_subgrade"],
            inplace=True
        )


    # --------------------------------------------------------
    # 10. ONE-HOT ENCODING
    # --------------------------------------------------------

    ohe = artifacts["ohe"]

    onehot_cols = artifacts["onehot_cols"]

    for col in onehot_cols:

        if col not in df.columns:
            df[col] = np.nan


    encoded = ohe.transform(
        df[onehot_cols]
    )


    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(
            onehot_cols
        ),
        index=df.index
    )


    df = df.drop(
        columns=onehot_cols,
        errors="ignore"
    )


    df = pd.concat(
        [df, encoded_df],
        axis=1
    )


    # --------------------------------------------------------
    # 11. EXACT FEATURE ORDER
    # --------------------------------------------------------

    feature_names = artifacts["feature_names"]

    df = df.reindex(
        columns=feature_names
    )


    # --------------------------------------------------------
    # 12. NaN / INF
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("\n================ PROCESSED INPUT ================")

    print("Shape:", df.shape)

    print(
        "Expected features:",
        len(feature_names)
    )

    print(
        "Actual features:",
        len(df.columns)
    )

    missing_features = (
        set(feature_names) - set(df.columns)
    )

    extra_features = (
        set(df.columns) - set(feature_names)
    )

    print(
        "Missing features:",
        missing_features
    )

    print(
        "Extra features:",
        extra_features
    )

    print(
        "NaN count:",
        df.isna().sum().sum()
    )

    print("=================================================\n")


    return df


# ============================================================
# PREDICTION
# ============================================================

def predict_single_sample(raw_data):

    df_proc = preprocess_single_sample(
        raw_data
    )


    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    lgb_predictions = []

    for model in artifacts["lgb_models"]:

        probability = (
            model.predict_proba(df_proc)[0, 1]
        )

        lgb_predictions.append(probability)


    lgb_p = np.mean(
        lgb_predictions
    )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_predictions = []

    for model in artifacts["xgb_models"]:

        probability = (
            model.predict_proba(df_proc)[0, 1]
        )

        xgb_predictions.append(probability)


    xgb_p = np.mean(
        xgb_predictions
    )


    # --------------------------------------------------------
    # CATBOOST
    # --------------------------------------------------------

    cat_predictions = []

    for model in artifacts["cat_models"]:

        probability = (
            model.predict_proba(df_proc)[0, 1]
        )

        cat_predictions.append(probability)


    cat_p = np.mean(
        cat_predictions
    )


    # --------------------------------------------------------
    # BLEND
    # --------------------------------------------------------

    w_lgb, w_xgb, w_cat = (
        artifacts["weights"]
    )

    final_prob = (
        w_lgb * lgb_p
        +
        w_xgb * xgb_p
        +
        w_cat * cat_p
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = artifacts[
        "best_threshold"
    ]

    prediction = int(
        final_prob >= threshold
    )


    return {

        "probability": round(
            float(final_prob),
            5
        ),

        "prediction": prediction,

        "threshold_used": round(
            float(threshold),
            5
        ),

        "model_probabilities": {

            "lightgbm": round(
                float(lgb_p),
                5
            ),

            "xgboost": round(
                float(xgb_p),
                5
            ),

            "catboost": round(
                float(cat_p),
                5
            )
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Loan prediction API is running"
    })


# ============================================================
# PREDICT API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON payload provided"
            }), 400


        result = predict_single_sample(
            data
        )


        return jsonify({

            "success": True,

            "result": result

        })


    except Exception as e:

        print("\nERROR:")
        print(str(e))

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )