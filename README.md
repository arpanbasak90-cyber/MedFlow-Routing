# 🏥 MedFlow-Routing — AI-Powered Hospital Routing System

An intelligent ambulance routing system that uses AI to recommend the best hospital for a patient based on their medical condition, real-time hospital capacity, and distance.

---

## 📌 Features

- 🤖 AI-based hospital recommendation using patient symptoms and condition
- 🗺️ Online and offline routing support
- 📊 Real-time hospital capacity and specialty matching
- ⚡ Fast response time for emergency scenarios
- 🌐 Web interface for dispatchers and medical staff

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI / Uvicorn
- **AI/ML:** Scikit-learn, Supabase
- **Frontend:** HTML, CSS, JavaScript
- **Routing:** Custom online/offline routing engine
- **Database:** Supabase

---

## 📁 Project Structure

```
Hospital_ai_project/
│
├── main.py                  # Entry point
├── app.py                   # FastAPI app setup
├── backend.py               # Core backend logic
├── ai_interface.py          # AI model interface
├── hospital_selector.py     # Hospital selection logic
├── routing_engine.py        # Routing algorithms
├── online_router.py         # Online routing
├── offline_router.py        # Offline routing fallback
├── graph_cache_manager.py   # Graph caching
├── utils.py                 # Utility functions
├── frontend.html            # Main UI
├── data/                    # Dataset files
├── models/                  # Trained ML models
└── requirements.txt         # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/arpanbasak90-cyber/MedFlow-Routing.git
cd MedFlow-Routing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 4. Run the application

```bash
python main.py
```

Then open your browser and go to `http://localhost:8000`

---

## 👥 Contributors

- [@arpanbasak90-cyber](https://github.com/arpanbasak90-cyber)
- [@Priyasmit-work](https://github.com/Priyasmit-work)

---

## 📄 License

This project is for educational and research purposes.
