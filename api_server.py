"""
FastAPI Server for  Emotional Understanding System
Lightweight local API for predictions and recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json

from pipeline import Pipeline
from conversational_model import LightweightConversationalModel

# Initialize app and models
app = FastAPI(
    title=" Emotional Understanding API",
    description="Real-time emotional state prediction and guidance",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded once at startup)
pipeline = None
conversational_model = None

# Request/Response models
class PredictionRequest(BaseModel):
    journal_text: str
    ambience_type: Optional[str] = "forest"
    stress_level: int = 3  # 1-5
    energy_level: int = 3  # 1-5
    sleep_hours: float = 6.0
    duration_min: int = 15
    time_of_day: Optional[str] = "afternoon"
    previous_day_mood: Optional[str] = "calm"
    face_emotion_hint: Optional[str] = "none"
    
    def validate_inputs(self):
        """Validate request parameters"""
        if not self.journal_text or len(self.journal_text.strip()) == 0:
            raise ValueError("journal_text cannot be empty")
        if self.stress_level < 1 or self.stress_level > 5:
            raise ValueError("stress_level must be between 1 and 5")
        if self.energy_level < 1 or self.energy_level > 5:
            raise ValueError("energy_level must be between 1 and 5")
        if self.sleep_hours < 0 or self.sleep_hours > 24:
            raise ValueError("sleep_hours must be between 0 and 24")
        return True
    reflection_quality: Optional[str] = "clear"  # clear, vague, or conflicted

class PredictionResponse(BaseModel):
    id: Optional[int] = None
    predicted_state: str
    predicted_intensity: int
    confidence: float
    uncertain_flag: int
    what_to_do: str
    when_to_do: str
    supportive_message: str
    reasoning: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global pipeline, conversational_model
    
    print("\n[API] Loading  models...")
    try:
        # Initialize pipeline and load data for feature engineering
        pipeline = Pipeline()
        print("[API] ✓ Pipeline initialized")
        
        # Load data to prepare feature engineer
        try:
            pipeline.load_data()
            print("[API] ✓ Data loaded")
        except Exception as e:
            print(f"[API] ! Could not load data for feature engineering: {e}")
            # Create a minimal feature engineer if data doesn't exist
            from feature_engineering import FeatureEngineer
            pipeline.feature_engineer = FeatureEngineer()
            print("[API] ✓ Feature engineer created (minimal mode)")
        
        # Try to load pre-trained models if available
        try:
            pipeline.state_clf.load(str(Path('models') / 'state_classifier'))
            pipeline.intensity_pred.load(str(Path('models') / 'intensity_predictor'))
            print("[API] ✓ Pre-trained models loaded")
        except:
            print("[API] ! Pre-trained models not found, training minimal models...")
            try:
                # Try to prepare and train if possible
                if pipeline.train_data is not None:
                    X_train, X_test, y_state, y_intensity = pipeline.prepare_features()
                    pipeline.train_models(X_train, y_state, y_intensity)
                    print("[API] ✓ Models trained on available data")
                else:
                    print("[API] ! Skipping model training - no training data available")
            except Exception as train_err:
                print(f"[API] ! Could not train models: {train_err}")
        
        conversational_model = LightweightConversationalModel()
        print("[API] ✓ Conversational model initialized")
        print("[API] ✓ Server ready for predictions!")
    except Exception as e:
        print(f"[API] ✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main HTML interface"""
    html_path = Path("ui") / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    
    # Fallback: return basic info
    return """
    <html>
        <head><title> API</title></head>
        <body>
            <h1> Emotional Understanding API</h1>
            <p>API is running. See documentation at <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": pipeline is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction on user input
    
    Args:
        journal_text: User's reflection text
        stress_level: 1-5 (1=low, 5=high)
        energy_level: 1-5 (1=low, 5=high)
        sleep_hours: Hours of sleep
        time_of_day: morning/afternoon/evening/night
        Other metadata...
    
    Returns:
        Prediction with state, intensity, confidence, and recommendation
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Create feature DataFrame
        input_df = pd.DataFrame({
            'id': [0],
            'journal_text': [request.journal_text],
            'ambience_type': [request.ambience_type],
            'duration_min': [request.duration_min],
            'sleep_hours': [request.sleep_hours],
            'energy_level': [request.energy_level],
            'stress_level': [request.stress_level],
            'time_of_day': [request.time_of_day],
            'previous_day_mood': [request.previous_day_mood or 'calm'],
            'face_emotion_hint': [request.face_emotion_hint or 'none'],
            'reflection_quality': [request.reflection_quality],
        })
        
        # Feature engineering
        X = pipeline.feature_engineer.transform(input_df)
        
        # Predict state and intensity
        state = pipeline.state_clf.predict(X)[0]
        intensity = int(pipeline.intensity_pred.predict(X)[0])
        confidence_state = pipeline.state_clf.get_confidence_scores(X)[0]
        confidence_intensity = pipeline.intensity_pred.get_confidence_scores(X)[0]
        combined_confidence = (confidence_state * 0.6 + confidence_intensity * 0.4)
        
        # Uncertainty flag
        uncertain_flag = 1 if combined_confidence < 0.6 else 0
        
        # Apply decision engine
        from decision_engine import DecisionEngine
        engine = DecisionEngine()
        decision = engine.decide(
            emotional_state=state,
            intensity=intensity,
            stress_level=request.stress_level,
            energy_level=request.energy_level,
            time_of_day=request.time_of_day,
            sleep_hours=request.sleep_hours,
        )
        
        # Generate conversational message (override simple one)
        conversational_msg = conversational_model.generate_message(
            state=state,
            intensity=intensity,
            stress=request.stress_level,
            energy=request.energy_level,
            time_of_day=request.time_of_day
        )
        
        reasoning = f"State confidence: {confidence_state:.2f}. Intensity confidence: {confidence_intensity:.2f}. "
        reasoning += f"Stress/Energy ratio suggests need for {decision['what_to_do']}."
        
        return PredictionResponse(
            predicted_state=state,
            predicted_intensity=intensity,
            confidence=float(combined_confidence),
            uncertain_flag=uncertain_flag,
            what_to_do=decision['what_to_do'],
            when_to_do=decision['when_to_do'],
            supportive_message=conversational_msg,
            reasoning=reasoning,
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(file_path: str):
    """
    Batch predict on CSV file
    
    Args:
        file_path: Path to CSV with multiple samples
    
    Returns:
        JSON with predictions for all samples
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        df = pd.read_csv(file_path)
        predictions = []
        
        for idx, row in df.iterrows():
            request = PredictionRequest(
                journal_text=row.get('journal_text', ''),
                stress_level=int(row.get('stress_level', 3)),
                energy_level=int(row.get('energy_level', 3)),
                sleep_hours=float(row.get('sleep_hours', 6)),
                time_of_day=row.get('time_of_day', 'afternoon'),
            )
            
            pred = await predict(request)
            predictions.append(pred.dict())
        
        return {
            "total": len(predictions),
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get aggregated statistics about predictions"""
    # In production, this would come from a database
    # For now, return hardcoded example statistics
    return {
        "total_predictions": 120,
        "state_distribution": {
            "calm": 15,
            "focused": 22,
            "anxious": 18,
            "overwhelmed": 12,
            "tired": 15,
            "neutral": 38,
        },
        "average_confidence": 0.72,
        "uncertain_rate": 0.15,
        "top_recommendations": {
            "deep_work": 25,
            "rest": 22,
            "grounding": 20,
            "pause": 28,
            "yoga": 6,
        }
    }


@app.get("/info")
async def get_info():
    """Get system information"""
    return {
        "name": " Emotional Understanding System",
        "tagline": "From Understanding Humans → To Guiding Them",
        "version": "1.0.0",
        "features": [
            "Emotional state prediction",
            "Intensity estimation",
            "Decision guidance (what + when)",
            "Uncertainty quantification",
            "Conversational messages",
            "Label noise handling"
        ],
        "endpoints": {
            "predict": "POST /predict - Single prediction",
            "batch": "POST /batch-predict - Batch predictions",
            "health": "GET /health - Health check",
            "stats": "GET /stats - Aggregated statistics",
            "ui": "GET / - Web interface"
        }
    }


# Include API documentation
@app.get("/docs", response_class=HTMLResponse)
async def docs():
    """API documentation page"""
    return """
    <html>
    <head>
        <title> API Documentation</title>
        <style>
            body { font-family: Arial; margin: 20px; max-width: 1000px; }
            .endpoint { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; }
            code { background: #f4f4f4; padding: 2px 5px; }
        </style>
    </head>
    <body>
        <h1> API Documentation</h1>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Make a single prediction on user input</p>
            <p><strong>Request:</strong></p>
            <code>{
    "journal_text": "I'm feeling anxious",
    "stress_level": 4,
    "energy_level": 2,
    "sleep_hours": 5,
    "time_of_day": "morning"
}</code>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health and model status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /stats</h3>
            <p>Get aggregated prediction statistics</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /info</h3>
            <p>Get system information and available endpoints</p>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        Emotional Understanding API Server           ║
    ║       http://localhost:8000                               ║
    ║       API Docs: http://localhost:8000/docs                ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
