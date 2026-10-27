import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="BeverageHealth AI", page_icon="🥤", layout="wide")

st.title("🥤 BeverageHealth AI")
st.caption("Beverage ingredient and nutrition analysis using Machine Learning")
st.warning(
    "Educational prototype only. The included sample dataset uses synthetic, rule-derived labels. "
    "This is not a medical diagnosis or a clinically validated prediction."
)

@st.cache_data
def load_data():
    return pd.read_csv("beverage_dataset.csv")

@st.cache_resource
def build_model():
    df = load_data()
    features = ["sugar_g", "caffeine_mg", "calories_kcal", "sodium_mg"]
    X = df[features]
    y = df["nutrition_concern_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), y_test, preds, features

try:
    df = load_data()
    model, accuracy, y_test, preds, features = build_model()
except Exception as e:
    st.error(f"Could not load/train the model: {e}")
    st.stop()

st.sidebar.header("Beverage details")
beverage = st.sidebar.selectbox(
    "Beverage type",
    ["Soft drink", "Fruit juice", "Energy drink", "Sports drink", "Tea", "Coffee", "Other"]
)
serving = st.sidebar.number_input("Serving size (ml)", min_value=50, max_value=1500, value=250, step=50)
sugar = st.sidebar.number_input("Sugar (g per serving)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
caffeine = st.sidebar.number_input("Caffeine (mg per serving)", min_value=0.0, max_value=1000.0, value=0.0, step=5.0)
calories = st.sidebar.number_input("Calories (kcal per serving)", min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
sodium = st.sidebar.number_input("Sodium (mg per serving)", min_value=0.0, max_value=5000.0, value=50.0, step=5.0)

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Your beverage")
    st.write(f"**Type:** {beverage} · **Serving:** {serving} ml")
    metrics = st.columns(4)
    metrics[0].metric("Sugar", f"{sugar:g} g")
    metrics[1].metric("Caffeine", f"{caffeine:g} mg")
    metrics[2].metric("Calories", f"{calories:g} kcal")
    metrics[3].metric("Sodium", f"{sodium:g} mg")

    if st.button("Analyse beverage", type="primary", use_container_width=True):
        sample = pd.DataFrame([[sugar, caffeine, calories, sodium]], columns=features)
        category = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        confidence = max(probabilities) * 100

        st.subheader("ML model result")
        if category == "High":
            st.error("Predicted nutrition-concern category: HIGH")
        elif category == "Moderate":
            st.warning("Predicted nutrition-concern category: MODERATE")
        else:
            st.success("Predicted nutrition-concern category: LOW")
        st.write(f"Model confidence for this category: **{confidence:.1f}%**")
        st.caption("Confidence is relative to this demo model and synthetic labels; it is not a health-risk probability.")

        st.subheader("General nutrition notes")
        notes = []
        if sugar > 25:
            notes.append("Sugar is above 25 g per entered serving. Check added sugar and serving size.")
        if caffeine > 150:
            notes.append("Caffeine is above 150 mg per entered serving. Consider total daily caffeine from all sources.")
        if calories > 200:
            notes.append("Compare calories per serving with alternatives and consider the amount consumed.")
        if sodium > 300:
            notes.append("Compare sodium content with other beverages and your dietary needs.")
        if not notes:
            notes.append("Compare the complete nutrition label and serving size before choosing a beverage.")
        for note in notes:
            st.write("• " + note)
        st.info("These notes are general educational information, not personalised medical advice.")

with right:
    st.subheader("Nutrition profile")
    chart_df = pd.DataFrame({
        "Nutrient": ["Sugar (g)", "Caffeine (mg)", "Calories (kcal)", "Sodium (mg)"],
        "Amount": [sugar, caffeine, calories, sodium]
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(chart_df["Nutrient"], chart_df["Amount"])
    ax.set_ylabel("Entered amount (units differ)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Values have different units, so compare each nutrient with its own label/reference rather than comparing bar heights directly.")

with st.expander("About the Machine Learning model"):
    st.write("Algorithm: Random Forest Classifier")
    st.write(f"Test accuracy against the demo dataset's labels: **{accuracy*100:.1f}%**")
    st.write("The labels in the included dataset are generated from transparent example rules, not from observed medical outcomes.")
    st.write(f"Dataset size: {len(df)} records")
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Project testing details"):
    st.write("Test-set classification report:")
    st.code(classification_report(y_test, preds, zero_division=0))
    st.write("Confusion matrix (actual rows × predicted columns; class order: High, Low, Moderate):")
    labels = ["High", "Low", "Moderate"]
    cm = confusion_matrix(y_test, preds, labels=labels)
    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

st.divider()
st.subheader("Project objectives")
st.write("- Accept nutrition values for a beverage.")
st.write("- Train a Random Forest model using a CSV dataset.")
st.write("- Predict a demo nutrition-concern category.")
st.write("- Display nutrition values and general educational notes.")
st.caption("Future improvement: replace the synthetic sample data with a verified, real beverage nutrition dataset and document its source.")
