import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class MLFoodScale:
    def __init__(self, food_data_path: str = 'food.csv', model_path: str = 'food_scale_model.joblib'):
        """Initialize the ML-enhanced food scale system.
        
        Args:
            food_data_path (str): Path to the food database CSV file
            model_path (str): Path to save/load the trained model
        """
        self.food_data_path = food_data_path
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'Data.Protein', 'Data.Carbohydrate', 'Data.Fat.Total Lipid', 
            'Data.Fiber', 'Data.Energy', 'Data.Water'
        ]
        self.target_columns = [
            'Data.Protein', 'Data.Carbohydrate', 'Data.Fat.Total Lipid', 
            'Data.Fiber', 'Data.Energy'
        ]
        
        # Load data
        try:
            self.food_database = pd.read_csv(food_data_path)
            self.current_weight = 0.0
            self.current_food = None
            self.measurement_history = []
            
            # Check if model exists, if not train a new one
            if os.path.exists(model_path):
                self.load_model()
            else:
                self.train_model()
                
        except Exception as e:
            raise Exception(f"Error initializing ML Food Scale: {str(e)}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training the model.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (targets) for training
        """
        # Select features and targets
        X = self.food_database[self.feature_columns].copy()
        y = self.food_database[self.target_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def train_model(self) -> None:
        """Train the machine learning model."""
        print("Training machine learning model...")
        
        # Prepare data
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save model
        self.save_model()
        
        # Plot feature importance
        self.plot_feature_importance()
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_columns = saved_data['feature_columns']
            self.target_columns = saved_data['target_columns']
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Training a new model instead...")
            self.train_model()
    
    def plot_feature_importance(self) -> None:
        """Plot feature importance for the trained model."""
        if self.model is None:
            return
            
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance for Nutrient Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
    
    def set_weight(self, weight_grams: float) -> None:
        """Set the current weight reading from the scale.
        
        Args:
            weight_grams (float): Weight in grams
        """
        self.current_weight = weight_grams
        
    def identify_food(self, food_name: str) -> bool:
        """Identify the food item from the database.
        
        Args:
            food_name (str): Name of the food to identify
            
        Returns:
            bool: True if food was found, False otherwise
        """
        # Search for the food in the database
        food_match = self.food_database[
            self.food_database['Description'].str.contains(food_name, case=False, na=False)
        ]
        
        if not food_match.empty:
            self.current_food = food_match.iloc[0]
            return True
        return False
    
    def predict_nutrients(self) -> Optional[Dict[str, float]]:
        """Predict nutrients using the ML model.
        
        Returns:
            Optional[Dict[str, float]]: Predicted nutrients or None if prediction fails
        """
        if self.current_food is None or self.current_weight <= 0 or self.model is None:
            return None
            
        try:
            # Prepare features for prediction
            features = self.current_food[self.feature_columns].values.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Scale prediction to current weight
            nutrients = {}
            for i, nutrient in enumerate(self.target_columns):
                nutrient_name = nutrient.split('.')[-1].lower()
                nutrients[nutrient_name] = prediction[i] * (self.current_weight / 100)
            
            return nutrients
        except Exception as e:
            print(f"Error predicting nutrients: {str(e)}")
            return None
    
    def calculate_nutrients(self) -> Optional[Dict[str, float]]:
        """Calculate nutrients based on current weight and food.
        
        Returns:
            Optional[Dict[str, float]]: Dictionary of nutrients and their amounts,
                                      or None if no food is identified
        """
        if self.current_food is None or self.current_weight <= 0:
            return None
            
        # Calculate nutrients per 100g
        nutrients = {
            'protein': self.current_food['Data.Protein'] * (self.current_weight / 100),
            'carbohydrates': self.current_food['Data.Carbohydrate'] * (self.current_weight / 100),
            'fat': self.current_food['Data.Fat.Total Lipid'] * (self.current_weight / 100),
            'fiber': self.current_food['Data.Fiber'] * (self.current_weight / 100),
            'calories': self.current_food['Data.Energy'] * (self.current_weight / 100)
        }
        
        return nutrients
    
    def get_food_info(self, use_ml: bool = True) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """Get current food information including name, weight, and nutrients.
        
        Args:
            use_ml (bool): Whether to use ML predictions or traditional calculations
            
        Returns:
            Optional[Tuple[str, float, Dict[str, float]]]: Food name, weight, and nutrients,
                                                         or None if no food is identified
        """
        if self.current_food is None:
            return None
            
        # Get nutrients using ML or traditional method
        if use_ml:
            nutrients = self.predict_nutrients()
        else:
            nutrients = self.calculate_nutrients()
            
        if nutrients is None:
            return None
            
        return (
            self.current_food['Description'],
            self.current_weight,
            nutrients
        )

    def calculate_accuracy(self, expected_weight: float, expected_nutrients: Dict[str, float], use_ml: bool = True) -> Dict[str, float]:
        """Calculate the accuracy of the measurements.
        
        Args:
            expected_weight (float): Expected weight in grams
            expected_nutrients (Dict[str, float]): Expected nutrient values
            use_ml (bool): Whether to use ML predictions or traditional calculations
            
        Returns:
            Dict[str, float]: Accuracy metrics for each measurement
        """
        if self.current_food is None or self.current_weight <= 0:
            return {}
            
        # Calculate weight accuracy
        weight_accuracy = 1 - abs(self.current_weight - expected_weight) / expected_weight
        
        # Get nutrients using ML or traditional method
        if use_ml:
            current_nutrients = self.predict_nutrients()
        else:
            current_nutrients = self.calculate_nutrients()
            
        if current_nutrients is None:
            return {'weight': weight_accuracy}
            
        # Calculate nutrient accuracies
        nutrient_accuracies = {}
        
        for nutrient, expected_value in expected_nutrients.items():
            if nutrient in current_nutrients and expected_value > 0:
                accuracy = 1 - abs(current_nutrients[nutrient] - expected_value) / expected_value
                nutrient_accuracies[nutrient] = max(0, min(1, accuracy))  # Clamp between 0 and 1
                
        # Calculate overall accuracy
        accuracies = {
            'weight': weight_accuracy,
            **nutrient_accuracies
        }
        
        # Store measurement in history
        self.measurement_history.append({
            'food': self.current_food['Description'],
            'expected_weight': expected_weight,
            'measured_weight': self.current_weight,
            'expected_nutrients': expected_nutrients,
            'measured_nutrients': current_nutrients,
            'accuracies': accuracies,
            'method': 'ML' if use_ml else 'Traditional'
        })
        
        return accuracies

    def get_average_accuracy(self, use_ml: bool = True) -> Dict[str, float]:
        """Calculate average accuracy across all measurements.
        
        Args:
            use_ml (bool): Whether to filter for ML predictions or traditional calculations
            
        Returns:
            Dict[str, float]: Average accuracy for each metric
        """
        if not self.measurement_history:
            return {}
            
        # Filter measurements by method
        filtered_history = [
            m for m in self.measurement_history 
            if (m['method'] == 'ML') == use_ml
        ]
        
        if not filtered_history:
            return {}
            
        # Initialize accumulators
        total_accuracies = {}
        count = len(filtered_history)
        
        # Sum up all accuracies
        for measurement in filtered_history:
            for metric, accuracy in measurement['accuracies'].items():
                if metric not in total_accuracies:
                    total_accuracies[metric] = 0
                total_accuracies[metric] += accuracy
                
        # Calculate averages
        return {metric: acc/count for metric, acc in total_accuracies.items()}

def main():
    # Initialize the ML-enhanced food scale
    scale = MLFoodScale()
    
    # Example usage
    print("\nML-Enhanced Smart Food Scale System")
    print("----------------------------------")
    
    # Get food name from user
    food_name = input("Enter food name: ")
    
    # Try to identify the food
    if scale.identify_food(food_name):
        # Get weight from user
        try:
            weight = float(input("Enter weight in grams: "))
            scale.set_weight(weight)
            
            # Get food information using both methods
            traditional_info = scale.get_food_info(use_ml=False)
            ml_info = scale.get_food_info(use_ml=True)
            
            if traditional_info and ml_info:
                food_name, weight, traditional_nutrients = traditional_info
                _, _, ml_nutrients = ml_info
                
                print("\nFood Information:")
                print(f"Food: {food_name}")
                print(f"Weight: {weight:.1f}g")
                
                # Display traditional calculations
                print("\nTraditional Calculations:")
                print(f"Calories: {traditional_nutrients['calories']:.1f} kcal")
                print(f"Protein: {traditional_nutrients['protein']:.1f}g")
                print(f"Carbohydrates: {traditional_nutrients['carbohydrates']:.1f}g")
                print(f"Fat: {traditional_nutrients['fat']:.1f}g")
                print(f"Fiber: {traditional_nutrients['fiber']:.1f}g")
                
                # Display ML predictions
                print("\nML Predictions:")
                print(f"Calories: {ml_nutrients['calories']:.1f} kcal")
                print(f"Protein: {ml_nutrients['protein']:.1f}g")
                print(f"Carbohydrates: {ml_nutrients['carbohydrates']:.1f}g")
                print(f"Fat: {ml_nutrients['fat']:.1f}g")
                print(f"Fiber: {ml_nutrients['fiber']:.1f}g")
                
                # Calculate accuracy (example values)
                expected_nutrients = {
                    'calories': traditional_nutrients['calories'] * 0.95,  # Example: 95% of calculated value
                    'protein': traditional_nutrients['protein'] * 0.98,    # Example: 98% of calculated value
                    'carbohydrates': traditional_nutrients['carbohydrates'] * 0.97,  # Example: 97% of calculated value
                    'fat': traditional_nutrients['fat'] * 0.96,           # Example: 96% of calculated value
                    'fiber': traditional_nutrients['fiber'] * 0.99        # Example: 99% of calculated value
                }
                
                # Calculate accuracy for both methods
                traditional_accuracies = scale.calculate_accuracy(weight, expected_nutrients, use_ml=False)
                ml_accuracies = scale.calculate_accuracy(weight, expected_nutrients, use_ml=True)
                
                # Display traditional accuracy
                print("\nTraditional Method Accuracy:")
                for metric, accuracy in traditional_accuracies.items():
                    print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                
                # Display ML accuracy
                print("\nML Method Accuracy:")
                for metric, accuracy in ml_accuracies.items():
                    print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                
                # Show average accuracy if multiple measurements
                if len(scale.measurement_history) > 1:
                    traditional_avg = scale.get_average_accuracy(use_ml=False)
                    ml_avg = scale.get_average_accuracy(use_ml=True)
                    
                    print("\nAverage Traditional Accuracy:")
                    for metric, accuracy in traditional_avg.items():
                        print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                    
                    print("\nAverage ML Accuracy:")
                    for metric, accuracy in ml_avg.items():
                        print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                        
        except ValueError:
            print("Error: Please enter a valid weight number")
    else:
        print(f"Error: Food '{food_name}' not found in database")

if __name__ == "__main__":
    main() 