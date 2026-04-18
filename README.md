# 🤖 AI Data Assistant

A modern desktop application that combines **Machine Learning + Conversational UI** to help users analyze datasets, train models, and generate predictions — all through an interactive chat interface.

---

## 🚀 Overview

AI Data Assistant is an intelligent system that allows users to:

* Upload a dataset (CSV)
* Interact with the system via chat
* Automatically train ML models
* Generate predictions
* Visualize results using graphs
* Get simple explanations powered by LLM

The application is designed with a **modern UI**, featuring:

* Chat-based workflow
* Analytics dashboard
* Session state panel
* Smooth interactions and structured layout

---

## 🧠 Key Features

### 💬 Chat-Based ML Workflow

* Step-by-step guided interaction
* Automatically:

  * Detects target column
  * Suggests features
  * Handles missing values
* Asks only **important questions** before prediction 

---

### 📊 Machine Learning Engine

Supports both:

* **Classification**
* **Regression**

Includes 8 models:

* Random Forest
* Decision Tree
* Gradient Boosting
* Extra Trees
* AdaBoost
* KNN
* Logistic / Ridge
* SVM 

---

### 📈 Analytics Dashboard

* Graph-based insights after prediction
* Includes:

  * Feature relationships
  * Trends
  * Statistical summary
* Scrollable and structured UI 

---

### 🧾 Session State Tracking

Displays:

* Selected model
* Target column
* Task type
* Metrics (Accuracy, RMSE, R², etc.)
* Feature importance 

---

### 🤖 LLM Integration (Optional)

* Uses local LLM (Ollama)
* Provides:

  * Dataset summary
  * Prediction explanation in simple language 

---

### 🎨 Modern UI

* Chat bubbles with typing animation
* Sidebar navigation (Chat / Graphs)
* Top bar with workflow steps
* Glass-style cards and panels
* Smooth scrolling and transitions

---

## 🏗️ Project Structure

```
AI_Data_Assistant/
│
├── main.py                # Main application (UI + logic)
├── Controller            # Chat & workflow logic
├── Session               # Stores session state
├── MLEngine              # ML model training & prediction
├── GraphsPanel           # Analytics dashboard
├── LLM Integration       # Dataset summary & explanation
```

---

## ⚙️ How It Works

1. Upload a CSV file
2. System analyzes dataset
3. Chat guides user through:

   * Target selection
   * Feature filtering
   * Missing value handling
   * Model selection
4. Model is trained automatically
5. User provides inputs
6. Prediction is generated
7. Graphs & insights are displayed

---

## 🖥️ Installation

### 1. Clone the repository

```bash
[git clone https://github.com/githubmofo/ML-Chatbot.git]
cd ML-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python main.py
```

---

## 🧪 Requirements

* Python 3.8+
* Works on Windows/Linux/Mac

---

## 🔌 Optional Setup (LLM)

Install and run Ollama:

```bash
ollama run qwen2.5
```

If not running, app will still work (without explanations).

---

## 📊 Example Use Case

* Upload dataset (e.g., house prices)
* Select target → price
* Choose model → Random Forest
* Enter feature values
* Get:

  * Prediction
  * Metrics
  * Feature importance
  * Graphs
  * Explanation

---

## ⚠️ Limitations

* Tkinter-based UI (limited advanced rendering)
* Large datasets are sampled for performance 
* LLM requires local setup (Ollama)

---

## 🚀 Future Improvements

* Web-based UI (React + Vite)
* More advanced visualizations
* Model comparison
* Export reports
* Cloud LLM support

---

## **👨‍💻 Author**

---
By Jenish Lad
By Parv Garara


