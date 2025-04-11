import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import os

class SmartFoodScale:
    def __init__(self, food_data_path: str = 'food.csv'):
        """Initialize the smart food scale system.
        
        Args:
            food_data_path (str): Path to the food database CSV file
        """
        try:
            self.food_database = pd.read_csv(food_data_path)
            self.current_weight = 0.0
            self.current_food = None
            self.measurement_history = []
        except Exception as e:
            raise Exception(f"Error loading food database: {str(e)}")
        
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
            'calories': self._calculate_calories() * (self.current_weight / 100)
        }
        
        return nutrients
    
    def _calculate_calories(self) -> float:
        """Calculate calories per 100g using macronutrients.
        
        Returns:
            float: Calories per 100g
        """
        # Standard conversion: 4 cal/g for protein and carbs, 9 cal/g for fat
        protein_cals = self.current_food['Data.Protein'] * 4
        carb_cals = self.current_food['Data.Carbohydrate'] * 4
        fat_cals = self.current_food['Data.Fat.Total Lipid'] * 9
        
        return protein_cals + carb_cals + fat_cals
    
    def get_food_info(self) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """Get current food information including name, weight, and nutrients.
        
        Returns:
            Optional[Tuple[str, float, Dict[str, float]]]: Food name, weight, and nutrients,
                                                         or None if no food is identified
        """
        if self.current_food is None:
            return None
            
        nutrients = self.calculate_nutrients()
        if nutrients is None:
            return None
            
        return (
            self.current_food['Description'],
            self.current_weight,
            nutrients
        )

    def calculate_accuracy(self, expected_weight: float, expected_nutrients: Dict[str, float]) -> Dict[str, float]:
        """Calculate the accuracy of the measurements.
        
        Args:
            expected_weight (float): Expected weight in grams
            expected_nutrients (Dict[str, float]): Expected nutrient values
            
        Returns:
            Dict[str, float]: Accuracy metrics for each measurement
        """
        if self.current_food is None or self.current_weight <= 0:
            return {}
            
        # Calculate weight accuracy
        weight_accuracy = 1 - abs(self.current_weight - expected_weight) / expected_weight
        
        # Calculate nutrient accuracies
        current_nutrients = self.calculate_nutrients()
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
            'accuracies': accuracies
        })
        
        return accuracies

    def get_average_accuracy(self) -> Dict[str, float]:
        """Calculate average accuracy across all measurements.
        
        Returns:
            Dict[str, float]: Average accuracy for each metric
        """
        if not self.measurement_history:
            return {}
            
        # Initialize accumulators
        total_accuracies = {}
        count = len(self.measurement_history)
        
        # Sum up all accuracies
        for measurement in self.measurement_history:
            for metric, accuracy in measurement['accuracies'].items():
                if metric not in total_accuracies:
                    total_accuracies[metric] = 0
                total_accuracies[metric] += accuracy
                
        # Calculate averages
        return {metric: acc/count for metric, acc in total_accuracies.items()}

def main():
    # Initialize the smart food scale
    scale = SmartFoodScale()
    
    # Example usage
    print("Smart Food Scale System")
    print("----------------------")
    
    # Get food name from user
    food_name = input("Enter food name: ")
    
    # Try to identify the food
    if scale.identify_food(food_name):
        # Get weight from user
        try:
            weight = float(input("Enter weight in grams: "))
            scale.set_weight(weight)
            
            # Get food information
            food_info = scale.get_food_info()
            if food_info:
                food_name, weight, nutrients = food_info
                
                print("\nFood Information:")
                print(f"Food: {food_name}")
                print(f"Weight: {weight:.1f}g")
                print("\nNutrients:")
                print(f"Calories: {nutrients['calories']:.1f} kcal")
                print(f"Protein: {nutrients['protein']:.1f}g")
                print(f"Carbohydrates: {nutrients['carbohydrates']:.1f}g")
                print(f"Fat: {nutrients['fat']:.1f}g")
                print(f"Fiber: {nutrients['fiber']:.1f}g")
                
                # Calculate accuracy (example values)
                expected_nutrients = {
                    'calories': nutrients['calories'] * 0.95,  # Example: 95% of calculated value
                    'protein': nutrients['protein'] * 0.98,    # Example: 98% of calculated value
                    'carbohydrates': nutrients['carbohydrates'] * 0.97,  # Example: 97% of calculated value
                    'fat': nutrients['fat'] * 0.96,           # Example: 96% of calculated value
                    'fiber': nutrients['fiber'] * 0.99        # Example: 99% of calculated value
                }
                
                accuracies = scale.calculate_accuracy(weight, expected_nutrients)
                
                print("\nAccuracy Metrics:")
                for metric, accuracy in accuracies.items():
                    print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                
                # Show average accuracy if multiple measurements
                if len(scale.measurement_history) > 1:
                    avg_accuracies = scale.get_average_accuracy()
                    print("\nAverage Accuracy Across All Measurements:")
                    for metric, accuracy in avg_accuracies.items():
                        print(f"{metric.capitalize()}: {accuracy*100:.1f}%")
                        
        except ValueError:
            print("Error: Please enter a valid weight number")
    else:
        print(f"Error: Food '{food_name}' not found in database")

if __name__ == "__main__":
    main() 