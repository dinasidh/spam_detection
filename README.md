To run the Email Spam Detector project, please follow these steps:

This project consists of two main parts: a Python Flask backend for the machine learning model and a React frontend for the user interface.

**Project Structure:**

```
Spam_detector/
├── backend/
│   ├── data/
│   │   └── spam.csv
│   ├── preprocessing/
│   │   └── text_preprocessor.py
│   ├── models/
│   │   └── spam_classifier.py
│   ├── utils/
│   │   └── evaluation.py
│   └── app.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── SpamDetector.css
│   │   │   └── SpamDetector.js
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── index.css
│   │   └── index.js
│   └── package.json
├── requirements.txt
└── README.md
```

### 1. Backend Setup and Run

1.  **Install Python Dependencies:**
    Open your terminal or command prompt, navigate to the project root directory (`c:\Users\kushw\OneDrive\Desktop\Spam_detector`), and install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to download NLTK data. The script `backend/preprocessing/text_preprocessor.py` attempts to do this automatically, but if it fails, you can run these commands in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

2.  **Run the Flask Backend:**
    Navigate to the `backend` directory and run the Flask application:
    ```bash
    cd backend
    python app.py
    ```
    The backend server will start, typically on `http://127.0.0.1:5000/`.

### 2. Frontend Setup and Run

1.  **Install Node.js Dependencies:**
    Open a **new** terminal or command prompt, navigate to the `frontend` directory:
    ```bash
    cd frontend
    npm install
    ```

2.  **Run the React Frontend:**
    From the `frontend` directory, start the React development server:
    ```bash
    npm start
    ```

3.  **Access the GUI:**
    After running `npm start`, your web browser should automatically open a new tab displaying the React application, usually at `http://localhost:3000/`. If it doesn't, open your browser and go to this address.

### 3. Use the Spam Detector

-   Ensure both the Flask backend and the React frontend are running simultaneously.
-   In the React interface, you will see a text area where you can enter or paste an email message.
-   Click the **"Predict"** button to classify the email as Spam or Ham.
-   The prediction result will be displayed on the main page.
-   Model evaluation metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix) will also be displayed.
