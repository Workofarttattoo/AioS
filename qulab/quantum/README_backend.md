# Backend
cd qulab/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend (new terminal)
cd ../frontend
npm i
npm run dev