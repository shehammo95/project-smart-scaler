import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import cv2
import pytesseract
from PIL import Image
import pyzbar.pyzbar as pyzbar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
from fpdf import FPDF
import qrcode
from dataclasses import dataclass
from enum import Enum
import logging
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FoodScaleError(Exception):
    """Custom exception for food scale errors"""
    pass

class DatabaseError(FoodScaleError):
    """Exception for database-related errors"""
    pass

class FoodNotFoundError(FoodScaleError):
    """Exception for when food is not found in database"""
    pass

class InvalidWeightError(FoodScaleError):
    """Exception for invalid weight values"""
    pass

class DietaryRestriction(Enum):
    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    KETO = "keto"
    PALEO = "paleo"

@dataclass
class NutritionalGoal:
    calories: float
    protein: float
    carbohydrates: float
    fat: float
    fiber: float

@dataclass
class UserProfile:
    name: str
    age: int
    weight: float
    height: float
    activity_level: str
    dietary_restrictions: List[DietaryRestriction]
    nutritional_goals: NutritionalGoal

class AdvancedFoodScale:
    def __init__(self, food_data_path: str = 'food.csv', db_path: str = 'food_scale.db'):
        """Initialize the advanced food scale system.
        
        Args:
            food_data_path (str): Path to the food database CSV file
            db_path (str): Path to the SQLite database file
        """
        self.food_data_path = food_data_path
        self.db_path = db_path
        self.food_database = pd.read_csv(food_data_path)
        self.current_weight = 0.0
        self.current_food = None
        self.users = {}
        self.measurement_history = []
        self.recipes = []
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            profile_data TEXT
        )
        ''')
        
        # Create measurements table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            food_name TEXT,
            weight REAL,
            nutrients TEXT,
            timestamp DATETIME,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create recipes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            ingredients TEXT,
            instructions TEXT,
            nutritional_info TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_user(self, profile: UserProfile) -> int:
        """Add a new user to the system.
        
        Args:
            profile (UserProfile): User profile information
            
        Returns:
            int: User ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        profile_data = {
            'age': profile.age,
            'weight': profile.weight,
            'height': profile.height,
            'activity_level': profile.activity_level,
            'dietary_restrictions': [r.value for r in profile.dietary_restrictions],
            'nutritional_goals': {
                'calories': profile.nutritional_goals.calories,
                'protein': profile.nutritional_goals.protein,
                'carbohydrates': profile.nutritional_goals.carbohydrates,
                'fat': profile.nutritional_goals.fat,
                'fiber': profile.nutritional_goals.fiber
            }
        }
        
        cursor.execute(
            'INSERT INTO users (name, profile_data) VALUES (?, ?)',
            (profile.name, json.dumps(profile_data))
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
        
    def identify_food_from_image(self, image_path: str) -> Optional[str]:
        """Identify food from an image using computer vision.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[str]: Identified food name or None if not found
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text from image (for labels)
            text = pytesseract.image_to_string(image_rgb)
            
            # Find food items in database that match the extracted text
            matches = []
            for food in self.food_database['Description']:
                if food.lower() in text.lower():
                    matches.append(food)
                    
            if matches:
                return matches[0]  # Return the first match
                
            # If no text match, use color and shape analysis
            # This is a simplified version - in a real system, you'd use a trained model
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze shape and color characteristics
            # This is where you'd implement more sophisticated food recognition
            # For now, return None if no match found
            return None
            
        except Exception as e:
            logger.error(f"Error in food image recognition: {str(e)}")
            return None
            
    def scan_barcode(self, image_path: str) -> Optional[str]:
        """Scan barcode from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[str]: Product name or None if not found
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Decode barcodes
            decoded_objects = pyzbar.decode(image)
            
            for obj in decoded_objects:
                barcode = obj.data.decode('utf-8')
                # In a real system, you'd query a product database with the barcode
                # For now, return None
                return None
                
            return None
            
        except Exception as e:
            logger.error(f"Error in barcode scanning: {str(e)}")
            return None
            
    def estimate_portion_size(self, image_path: str) -> Optional[float]:
        """Estimate portion size from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[float]: Estimated weight in grams or None if estimation fails
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Find the largest contour (assumed to be the food item)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate area
            area = cv2.contourArea(largest_contour)
            
            # In a real system, you'd calibrate this conversion based on:
            # 1. Camera distance
            # 2. Reference object size
            # 3. Food density
            # For now, use a simple conversion factor
            estimated_weight = area * 0.1  # Example conversion factor
            
            return estimated_weight
            
        except Exception as e:
            logger.error(f"Error in portion size estimation: {str(e)}")
            return None
            
    def suggest_meals(self, user_id: int, target_calories: float) -> List[Dict]:
        """Suggest meal combinations based on nutritional goals.
        
        Args:
            user_id (int): User ID
            target_calories (float): Target calories for the meal
            
        Returns:
            List[Dict]: List of suggested meal combinations
        """
        try:
            # Get user profile
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT profile_data FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return []
                
            profile_data = json.loads(result[0])
            dietary_restrictions = profile_data['dietary_restrictions']
            
            # Filter foods based on dietary restrictions
            available_foods = self.food_database.copy()
            if DietaryRestriction.VEGETARIAN.value in dietary_restrictions:
                available_foods = available_foods[available_foods['Data.Protein'] < 20]
            if DietaryRestriction.VEGAN.value in dietary_restrictions:
                available_foods = available_foods[available_foods['Data.Protein'] < 20]
            if DietaryRestriction.GLUTEN_FREE.value in dietary_restrictions:
                available_foods = available_foods[~available_foods['Description'].str.contains('wheat|bread|pasta', case=False)]
                
            # Generate meal combinations
            suggestions = []
            for _ in range(5):  # Generate 5 suggestions
                meal = {
                    'foods': [],
                    'total_calories': 0,
                    'total_protein': 0,
                    'total_carbs': 0,
                    'total_fat': 0
                }
                
                remaining_calories = target_calories
                while remaining_calories > 0 and len(meal['foods']) < 5:
                    # Randomly select a food
                    food = available_foods.sample(n=1).iloc[0]
                    food_calories = food['Data.Energy']
                    
                    if food_calories <= remaining_calories:
                        meal['foods'].append({
                            'name': food['Description'],
                            'calories': food_calories,
                            'protein': food['Data.Protein'],
                            'carbs': food['Data.Carbohydrate'],
                            'fat': food['Data.Fat.Total Lipid']
                        })
                        
                        meal['total_calories'] += food_calories
                        meal['total_protein'] += food['Data.Protein']
                        meal['total_carbs'] += food['Data.Carbohydrate']
                        meal['total_fat'] += food['Data.Fat.Total Lipid']
                        
                        remaining_calories -= food_calories
                        
                suggestions.append(meal)
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Error in meal suggestion: {str(e)}")
            return []
            
    def track_nutritional_goals(self, user_id: int, date: datetime = None) -> Dict:
        """Track nutritional goals and progress.
        
        Args:
            user_id (int): User ID
            date (datetime): Date to track (default: today)
            
        Returns:
            Dict: Nutritional progress information
        """
        try:
            if date is None:
                date = datetime.now()
                
            # Get user profile
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT profile_data FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            
            if not result:
                return {}
                
            profile_data = json.loads(result[0])
            goals = profile_data['nutritional_goals']
            
            # Get measurements for the specified date
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            cursor.execute('''
                SELECT nutrients FROM measurements 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
            ''', (user_id, start_date, end_date))
            
            measurements = cursor.fetchall()
            conn.close()
            
            # Calculate totals
            totals = {
                'calories': 0,
                'protein': 0,
                'carbohydrates': 0,
                'fat': 0,
                'fiber': 0
            }
            
            for measurement in measurements:
                nutrients = json.loads(measurement[0])
                for nutrient, value in nutrients.items():
                    totals[nutrient] += value
                    
            # Calculate progress
            progress = {
                'goals': goals,
                'current': totals,
                'remaining': {
                    nutrient: max(0, goals[nutrient] - totals[nutrient])
                    for nutrient in goals
                },
                'percentage': {
                    nutrient: min(100, (totals[nutrient] / goals[nutrient]) * 100)
                    for nutrient in goals
                }
            }
            
            return progress
            
        except Exception as e:
            logger.error(f"Error in nutritional goals tracking: {str(e)}")
            return {}
            
    def suggest_recipes(self, available_ingredients: List[str]) -> List[Dict]:
        """Suggest recipes based on available ingredients.
        
        Args:
            available_ingredients (List[str]): List of available ingredients
            
        Returns:
            List[Dict]: List of suggested recipes
        """
        # TODO: Implement recipe suggestion logic
        return []
            
    def analyze_eating_patterns(self, user_id: int, days: int = 30) -> Dict:
        """Analyze eating patterns and provide insights.
        
        Args:
            user_id (int): User ID
            days (int): Number of days to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Get measurements for the specified period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT food_name, weight, nutrients, timestamp 
                FROM measurements 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
            ''', (user_id, start_date, end_date))
            
            measurements = cursor.fetchall()
            conn.close()
            
            if not measurements:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(measurements, columns=['food_name', 'weight', 'nutrients', 'timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['nutrients'] = df['nutrients'].apply(json.loads)
            
            # Calculate daily totals
            daily_totals = df.groupby(df['timestamp'].dt.date).apply(
                lambda x: pd.Series({
                    'calories': sum(n['calories'] for n in x['nutrients']),
                    'protein': sum(n['protein'] for n in x['nutrients']),
                    'carbohydrates': sum(n['carbohydrates'] for n in x['nutrients']),
                    'fat': sum(n['fat'] for n in x['nutrients']),
                    'fiber': sum(n['fiber'] for n in x['nutrients'])
                })
            )
            
            # Calculate statistics
            analysis = {
                'daily_averages': daily_totals.mean().to_dict(),
                'daily_std': daily_totals.std().to_dict(),
                'most_common_foods': df['food_name'].value_counts().head(5).to_dict(),
                'total_measurements': len(df),
                'average_weight': df['weight'].mean(),
                'weight_std': df['weight'].std()
            }
            
            # Generate visualizations
            plt.figure(figsize=(12, 6))
            daily_totals['calories'].plot()
            plt.title('Daily Calorie Intake')
            plt.savefig('calorie_trend.png')
            plt.close()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in eating pattern analysis: {str(e)}")
            return {}
            
    def export_data(self, user_id: int, format: str = 'csv', start_date: datetime = None, end_date: datetime = None) -> str:
        """Export nutritional data to various formats.
        
        Args:
            user_id (int): User ID
            format (str): Export format ('csv' or 'pdf')
            start_date (datetime): Start date for export
            end_date (datetime): End date for export
            
        Returns:
            str: Path to the exported file
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
                
            # Get measurements
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT food_name, weight, nutrients, timestamp 
                FROM measurements 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
            ''', (user_id, start_date, end_date))
            
            measurements = cursor.fetchall()
            conn.close()
            
            if not measurements:
                return ""
                
            # Convert to DataFrame
            df = pd.DataFrame(measurements, columns=['food_name', 'weight', 'nutrients', 'timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['nutrients'] = df['nutrients'].apply(json.loads)
            
            # Export based on format
            if format.lower() == 'csv':
                output_path = f'nutritional_data_{user_id}_{datetime.now().strftime("%Y%m%d")}.csv'
                df.to_csv(output_path, index=False)
            elif format.lower() == 'pdf':
                output_path = f'nutritional_data_{user_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, 'Nutritional Data Report', ln=True)
                pdf.set_font('Arial', '', 12)
                
                for _, row in df.iterrows():
                    pdf.cell(0, 10, f"Food: {row['food_name']}", ln=True)
                    pdf.cell(0, 10, f"Weight: {row['weight']}g", ln=True)
                    pdf.cell(0, 10, f"Timestamp: {row['timestamp']}", ln=True)
                    for nutrient, value in row['nutrients'].items():
                        pdf.cell(0, 10, f"{nutrient.capitalize()}: {value:.1f}", ln=True)
                    pdf.ln(10)
                    
                pdf.output(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error in data export: {str(e)}")
            return ""
            
    def generate_qr_code(self, data: Dict) -> str:
        """Generate QR code for food information.
        
        Args:
            data (Dict): Food information to encode
            
        Returns:
            str: Path to the generated QR code image
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(json_data)
            qr.make(fit=True)
            
            # Create image
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # Save image
            output_path = f"food_qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            qr_image.save(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in QR code generation: {str(e)}")
            return ""

    def identify_food(self, food_name: str) -> bool:
        """Identify food from name.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            bool: True if food is found, False otherwise
        """
        try:
            matching_foods = self.food_database[
                self.food_database['Description'].str.contains(food_name, case=False)
            ]
            
            if matching_foods.empty:
                raise FoodNotFoundError(f"Food '{food_name}' not found in database")
                
            self.current_food = matching_foods.iloc[0]
            return True
            
        except Exception as e:
            logger.error(f"Error identifying food: {str(e)}")
            raise FoodScaleError(f"Failed to identify food: {str(e)}")

    def set_weight(self, weight: float) -> None:
        """Set the current weight.
        
        Args:
            weight (float): Weight in grams
            
        Raises:
            InvalidWeightError: If weight is not positive
        """
        if weight <= 0:
            raise InvalidWeightError("Weight must be positive")
        self.current_weight = weight

def main():
    # Initialize the advanced food scale
    scale = AdvancedFoodScale()
    
    while True:
        print("\nAdvanced Food Scale System")
        print("-------------------------")
        print("1. Add User Profile")
        print("2. Identify Food from Image")
        print("3. Scan Barcode")
        print("4. Estimate Portion Size")
        print("5. Suggest Meals")
        print("6. Track Nutritional Goals")
        print("7. Suggest Recipes")
        print("8. Analyze Eating Patterns")
        print("9. Export Data")
        print("10. Generate QR Code")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-10): ")
        
        if choice == '0':
            break
            
        elif choice == '1':
            # Add user profile
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            weight = float(input("Enter weight (kg): "))
            height = float(input("Enter height (cm): "))
            activity_level = input("Enter activity level (low/medium/high): ")
            
            print("\nDietary Restrictions:")
            print("1. None")
            print("2. Vegetarian")
            print("3. Vegan")
            print("4. Gluten-free")
            print("5. Dairy-free")
            print("6. Keto")
            print("7. Paleo")
            
            restrictions = []
            while True:
                restriction = input("Enter restriction number (or 'done' to finish): ")
                if restriction.lower() == 'done':
                    break
                try:
                    restriction_num = int(restriction)
                    if 1 <= restriction_num <= 7:
                        restrictions.append(DietaryRestriction(restriction_num - 1))
                except ValueError:
                    print("Invalid input")
                    
            print("\nNutritional Goals:")
            calories = float(input("Enter target calories: "))
            protein = float(input("Enter target protein (g): "))
            carbs = float(input("Enter target carbohydrates (g): "))
            fat = float(input("Enter target fat (g): "))
            fiber = float(input("Enter target fiber (g): "))
            
            profile = UserProfile(
                name=name,
                age=age,
                weight=weight,
                height=height,
                activity_level=activity_level,
                dietary_restrictions=restrictions,
                nutritional_goals=NutritionalGoal(
                    calories=calories,
                    protein=protein,
                    carbohydrates=carbs,
                    fat=fat,
                    fiber=fiber
                )
            )
            
            user_id = scale.add_user(profile)
            print(f"\nUser profile added successfully! User ID: {user_id}")
            
        elif choice == '2':
            # Identify food from image
            image_path = input("Enter image path: ")
            food_name = scale.identify_food_from_image(image_path)
            if food_name:
                print(f"\nIdentified food: {food_name}")
            else:
                print("\nCould not identify food from image")
                
        elif choice == '3':
            # Scan barcode
            image_path = input("Enter image path: ")
            product_name = scale.scan_barcode(image_path)
            if product_name:
                print(f"\nScanned product: {product_name}")
            else:
                print("\nCould not scan barcode")
                
        elif choice == '4':
            # Estimate portion size
            image_path = input("Enter image path: ")
            weight = scale.estimate_portion_size(image_path)
            if weight:
                print(f"\nEstimated weight: {weight:.1f}g")
            else:
                print("\nCould not estimate portion size")
                
        elif choice == '5':
            # Suggest meals
            user_id = int(input("Enter user ID: "))
            target_calories = float(input("Enter target calories: "))
            suggestions = scale.suggest_meals(user_id, target_calories)
            
            if suggestions:
                print("\nMeal Suggestions:")
                for i, meal in enumerate(suggestions, 1):
                    print(f"\nMeal {i}:")
                    print(f"Total Calories: {meal['total_calories']:.1f}")
                    print(f"Total Protein: {meal['total_protein']:.1f}g")
                    print(f"Total Carbs: {meal['total_carbs']:.1f}g")
                    print(f"Total Fat: {meal['total_fat']:.1f}g")
                    print("\nFoods:")
                    for food in meal['foods']:
                        print(f"- {food['name']}")
            else:
                print("\nNo meal suggestions available")
                
        elif choice == '6':
            # Track nutritional goals
            user_id = int(input("Enter user ID: "))
            progress = scale.track_nutritional_goals(user_id)
            
            if progress:
                print("\nNutritional Progress:")
                print(f"Goals:")
                for nutrient, value in progress['goals'].items():
                    print(f"{nutrient.capitalize()}: {value:.1f}")
                print(f"\nCurrent:")
                for nutrient, value in progress['current'].items():
                    print(f"{nutrient.capitalize()}: {value:.1f}")
                print(f"\nRemaining:")
                for nutrient, value in progress['remaining'].items():
                    print(f"{nutrient.capitalize()}: {value:.1f}")
                print(f"\nPercentage of Goals:")
                for nutrient, value in progress['percentage'].items():
                    print(f"{nutrient.capitalize()}: {value:.1f}%")
            else:
                print("\nNo nutritional data available")
                
        elif choice == '7':
            # Suggest recipes
            ingredients = input("Enter available ingredients (comma-separated): ").split(',')
            suggestions = scale.suggest_recipes(ingredients)
            
            if suggestions:
                print("\nRecipe Suggestions:")
                for i, recipe in enumerate(suggestions, 1):
                    print(f"\nRecipe {i}: {recipe['name']}")
                    print("Ingredients:")
                    for ingredient in recipe['ingredients']:
                        print(f"- {ingredient}")
                    print("\nInstructions:")
                    print(recipe['instructions'])
                    print("\nNutritional Information:")
                    for nutrient, value in recipe['nutritional_info'].items():
                        print(f"{nutrient.capitalize()}: {value:.1f}")
            else:
                print("\nNo recipe suggestions available")
                
        elif choice == '8':
            # Analyze eating patterns
            user_id = int(input("Enter user ID: "))
            days = int(input("Enter number of days to analyze: "))
            analysis = scale.analyze_eating_patterns(user_id, days)
            
            if analysis:
                print("\nEating Pattern Analysis:")
                print("\nDaily Averages:")
                for nutrient, value in analysis['daily_averages'].items():
                    print(f"{nutrient.capitalize()}: {value:.1f}")
                print("\nMost Common Foods:")
                for food, count in analysis['most_common_foods'].items():
                    print(f"{food}: {count} times")
                print(f"\nTotal Measurements: {analysis['total_measurements']}")
                print(f"Average Weight: {analysis['average_weight']:.1f}g")
                print(f"Weight Standard Deviation: {analysis['weight_std']:.1f}g")
                print("\nCalorie trend plot saved as 'calorie_trend.png'")
            else:
                print("\nNo eating pattern data available")
                
        elif choice == '9':
            # Export data
            user_id = int(input("Enter user ID: "))
            format = input("Enter export format (csv/pdf): ")
            output_path = scale.export_data(user_id, format)
            
            if output_path:
                print(f"\nData exported successfully to: {output_path}")
            else:
                print("\nExport failed")
                
        elif choice == '10':
            # Generate QR code
            food_name = input("Enter food name: ")
            weight = float(input("Enter weight (g): "))
            
            data = {
                'food_name': food_name,
                'weight': weight,
                'timestamp': datetime.now().isoformat()
            }
            
            output_path = scale.generate_qr_code(data)
            if output_path:
                print(f"\nQR code generated successfully: {output_path}")
            else:
                print("\nQR code generation failed")
                
        else:
            print("\nInvalid choice")

if __name__ == "__main__":
    main() 