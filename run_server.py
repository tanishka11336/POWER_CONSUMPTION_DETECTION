import uvicorn
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 65)
    print(" TETOUAN SMART GRID DEEP LEARNING POWER FORECASTING SYSTEM")
    print(" Starting FastAPI Server & Neural Inference Engine...")
    print(" Web Dashboard: http://localhost:8000")
    print(" API Documentation: http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
