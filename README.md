# Career Compass

**Career Compass** is an interactive web application that helps users identify the countries that best match their career and lifestyle preferences.

By combining user-defined priorities with socio-economic indicators and data-driven methods, the app produces a personalized ranking of countries along with clear visual explanations.

This web app was developed for the Data Science Lab course.
---

## Project Goals

Young professionals often struggle to compare countries based on multiple dimensions such as income, job opportunities, cost of living, and quality of life.

Career Compass aims to:
- Translate user preferences into a numerical profile
- Compare this profile with real socio-economic data
- Generate a transparent and explainable ranking of countries
- Provide an intuitive and modern user experience

---

## How It Works

### 1. User Preferences
The user rates **7 key indicators** on a scale from **1 (not important)** to **5 (very important)**:

- **Income & Economy**
- **Employment**
- **Cost of Living**
- **Health & Environment**
- **Safety**
- **Social Life**
- **Mobility**

These values are normalized and converted into a **user preference vector**.

---

### 2. Recommendation System
The recommendation score is computed using **cosine similarity** between:
- the user preference vector
- each country’s indicator vector

The result is a **Match Score** (0–100%) for every country.

---

### 3. Clustering
Countries are grouped using **K-Means clustering** on the same indicators.
- The number of clusters is chosen automatically using **silhouette score**
- Each country displays its cluster membership to provide contextual insight

---

### 4. Results & Explainability
For the **Top 5 recommended countries**, the app shows:
- Match score with dynamic color coding
- Ranking position
- Cluster membership
- Expandable details including:
  - **Current Levels** (blue bars)
  - **Growth Trends** (green bars)

---

## User Interface Features

- Responsive layout
- Interactive sliders with real-time feedback
- Expandable country cards
- Clear visual distinction between current performance and trends

---

## Data Sources & Structure

The application relies on two preprocessed datasets:

- `df_levels_latest.csv`  
  Contains normalized latest year indicators (2024) per country

- `df_trends.csv`  
  Contains normalized (computed) trend indicators (`trend_*` columns)

Both datasets are assumed to be **pre-cleaned and normalized** in a separate data preparation notebook.

---

## Technologies Used

- **Python**
- **Streamlit** (web app framework)
- **Pandas / NumPy** (data handling)
- **Scikit-learn**
  - Cosine Similarity
  - K-Means
  - Silhouette Score
- **PyArrow** (Parquet support)
- **Custom CSS** for UI styling

---

## Running the App Locally

To run the application locally, make sure you have the following files in the **same folder**:

- `app.py`
- `df_levels_latest.csv`
- `df_trends.csv`

### 1. Open a terminal and move to the project folder
Navigate to the directory that contains the three files listed above.

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn pyarrow

streamlit run app.py
