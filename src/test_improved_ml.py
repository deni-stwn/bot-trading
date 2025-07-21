# filepath: c:\Users\ACER\Desktop\Bangden\trading-bot\src\test_improved_ml.py

# from ml.predictor import ImprovedMLPredictor
from ml.predictor_v2 import ImprovedMLPredictorV2
import pandas as pd
import numpy as np

def test_model():
    """Test the improved model"""
    print("🧪 Testing Improved ML Model")
    print("=" * 40)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
    test_data = []
    
    base_price = 50000
    for i, date in enumerate(dates):
        price = base_price + i * 10 + np.random.normal(0, 100)
        test_data.append({
            'time': date,
            'open': price,
            'high': price * 1.001,
            'low': price * 0.999,
            'close': price,
            'tick_volume': 100
        })
    
    test_df = pd.DataFrame(test_data)
    
    # Test prediction
    predictor = ImprovedMLPredictorV2()
    
    if predictor.model is None:
        print("❌ Model not loaded. Please train first:")
        print("   python improved_train_ml.py")
        return
    
    result = predictor.predict_signal(test_df)
    
    print("📊 Prediction Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_model()