import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum
import json
import os

class DietaryGoal(Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    HEALTH_IMPROVEMENT = "health_improvement"
    ENERGY_BOOST = "energy_boost"
    BALANCED_DIET = "balanced_diet"

@dataclass
class NutritionalValues:
    calories: float
    protein: float
    carbohydrates: float
    fat: float
    fiber: float
    sugar: float
    sodium: float

class NutritionAssistant:
    def __init__(self, food_data_path: str = 'data/food.csv'):
        """Initialize the nutrition assistant.
        
        Args:
            food_data_path (str): Path to the food database CSV file
        """
        self.food_database = pd.read_csv(food_data_path)
        self.pairings = self._load_pairings()
        self.substitutions = self._load_substitutions()
        
    def _load_pairings(self) -> Dict[str, List[str]]:
        """Load food pairings from a JSON file or create default ones.
        
        Returns:
            Dict[str, List[str]]: Dictionary of food pairings
        """
        pairings_file = 'data/food_pairings.json'
        
        if os.path.exists(pairings_file):
            with open(pairings_file, 'r') as f:
                return json.load(f)
        
        # Default pairings if file doesn't exist
        default_pairings = {
            "apple": ["almonds", "peanut butter", "cheese", "yogurt"],
            "banana": ["honey", "cinnamon", "nuts", "yogurt"],
            "chicken": ["rice", "vegetables", "quinoa", "sweet potato"],
            "salmon": ["asparagus", "rice", "quinoa", "avocado"],
            "rice": ["beans", "vegetables", "chicken", "fish"],
            "quinoa": ["chicken", "vegetables", "beans", "avocado"],
            "avocado": ["toast", "eggs", "salmon", "tomato"],
            "eggs": ["toast", "avocado", "spinach", "cheese"],
            "spinach": ["eggs", "chicken", "salmon", "quinoa"],
            "broccoli": ["chicken", "salmon", "rice", "quinoa"]
        }
        
        # Save default pairings
        os.makedirs('data', exist_ok=True)
        with open(pairings_file, 'w') as f:
            json.dump(default_pairings, f, indent=4)
            
        return default_pairings
    
    def _load_substitutions(self) -> Dict[str, List[str]]:
        """Load food substitutions from a JSON file or create default ones.
        
        Returns:
            Dict[str, List[str]]: Dictionary of food substitutions
        """
        substitutions_file = 'data/food_substitutions.json'
        
        if os.path.exists(substitutions_file):
            with open(substitutions_file, 'r') as f:
                return json.load(f)
        
        # Default substitutions if file doesn't exist
        default_substitutions = {
            "rice": ["quinoa", "cauliflower rice", "barley", "brown rice"],
            "pasta": ["zucchini noodles", "spaghetti squash", "rice noodles", "chickpea pasta"],
            "potato": ["sweet potato", "cauliflower", "turnips", "parsnips"],
            "bread": ["lettuce wraps", "collard wraps", "portobello mushrooms", "sweet potato"],
            "sugar": ["honey", "maple syrup", "stevia", "monk fruit sweetener"],
            "oil": ["avocado", "applesauce", "banana", "yogurt"],
            "milk": ["almond milk", "oat milk", "coconut milk", "soy milk"],
            "butter": ["avocado", "olive oil", "coconut oil", "applesauce"],
            "meat": ["tofu", "tempeh", "seitan", "legumes"],
            "cheese": ["nutritional yeast", "cashew cheese", "tofu", "avocado"]
        }
        
        # Save default substitutions
        os.makedirs('data', exist_ok=True)
        with open(substitutions_file, 'w') as f:
            json.dump(default_substitutions, f, indent=4)
            
        return default_substitutions
    
    def get_nutritional_values(self, food_name: str, weight_grams: float) -> Optional[NutritionalValues]:
        """Get nutritional values for a food item adjusted by weight.
        
        Args:
            food_name (str): Name of the food
            weight_grams (float): Weight in grams
            
        Returns:
            Optional[NutritionalValues]: Nutritional values or None if food not found
        """
        # Search for the food in the database
        food_match = self.food_database[
            self.food_database['Description'].str.contains(food_name, case=False, na=False)
        ]
        
        if food_match.empty:
            return None
            
        food = food_match.iloc[0]
        
        # Calculate nutritional values per 100g
        factor = weight_grams / 100
        
        return NutritionalValues(
            calories=food['Data.Energy'] * factor,
            protein=food['Data.Protein'] * factor,
            carbohydrates=food['Data.Carbohydrate'] * factor,
            fat=food['Data.Fat.Total Lipid'] * factor,
            fiber=food['Data.Fiber'] * factor,
            sugar=food.get('Data.Sugars', 0) * factor,
            sodium=food.get('Data.Sodium', 0) * factor
        )
    
    def get_goal_fit(self, food_name: str, weight_grams: float, goal: DietaryGoal) -> Dict[str, str]:
        """Assess how well a food fits with a dietary goal.
        
        Args:
            food_name (str): Name of the food
            weight_grams (float): Weight in grams
            goal (DietaryGoal): User's dietary goal
            
        Returns:
            Dict[str, str]: Assessment of fit with goal
        """
        nutritional_values = self.get_nutritional_values(food_name, weight_grams)
        
        if nutritional_values is None:
            return {
                "fit_score": "unknown",
                "explanation": "Food not found in database"
            }
        
        # Calculate macronutrient ratios
        total_macros = nutritional_values.protein + nutritional_values.carbohydrates + nutritional_values.fat
        if total_macros > 0:
            protein_ratio = nutritional_values.protein / total_macros
            carb_ratio = nutritional_values.carbohydrates / total_macros
            fat_ratio = nutritional_values.fat / total_macros
        else:
            protein_ratio = carb_ratio = fat_ratio = 0
        
        # Assess fit based on goal
        if goal == DietaryGoal.WEIGHT_LOSS:
            if nutritional_values.calories < 100 and nutritional_values.fiber > 2:
                return {
                    "fit_score": "excellent",
                    "explanation": "Low in calories and high in fiber, perfect for weight loss"
                }
            elif nutritional_values.calories < 200 and nutritional_values.protein > 5:
                return {
                    "fit_score": "good",
                    "explanation": "Moderate calories with protein to help maintain muscle during weight loss"
                }
            elif nutritional_values.calories > 300:
                return {
                    "fit_score": "poor",
                    "explanation": "High in calories, consider a smaller portion or a different food"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for weight loss, watch portion size"
                }
                
        elif goal == DietaryGoal.MUSCLE_GAIN:
            if nutritional_values.protein > 20 and nutritional_values.calories > 200:
                return {
                    "fit_score": "excellent",
                    "explanation": "High in protein and calories, great for muscle gain"
                }
            elif nutritional_values.protein > 10 and protein_ratio > 0.2:
                return {
                    "fit_score": "good",
                    "explanation": "Good protein content to support muscle growth"
                }
            elif nutritional_values.protein < 5:
                return {
                    "fit_score": "poor",
                    "explanation": "Low in protein, not ideal for muscle gain"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for muscle gain, consider adding a protein source"
                }
                
        elif goal == DietaryGoal.MAINTENANCE:
            if 0.1 <= protein_ratio <= 0.3 and 0.4 <= carb_ratio <= 0.6 and 0.2 <= fat_ratio <= 0.4:
                return {
                    "fit_score": "excellent",
                    "explanation": "Well-balanced macronutrients, perfect for maintenance"
                }
            elif nutritional_values.calories < 300 and nutritional_values.fiber > 2:
                return {
                    "fit_score": "good",
                    "explanation": "Moderate calories with fiber, good for maintenance"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for maintenance, consider portion size"
                }
                
        elif goal == DietaryGoal.HEALTH_IMPROVEMENT:
            if nutritional_values.fiber > 5 and nutritional_values.sugar < 5:
                return {
                    "fit_score": "excellent",
                    "explanation": "High in fiber and low in sugar, excellent for health"
                }
            elif nutritional_values.fiber > 3 and nutritional_values.sodium < 200:
                return {
                    "fit_score": "good",
                    "explanation": "Good fiber content and moderate sodium, good for health"
                }
            elif nutritional_values.sugar > 10 or nutritional_values.sodium > 500:
                return {
                    "fit_score": "poor",
                    "explanation": "High in sugar or sodium, not ideal for health improvement"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for health improvement"
                }
                
        elif goal == DietaryGoal.ENERGY_BOOST:
            if nutritional_values.carbohydrates > 20 and nutritional_values.calories > 150:
                return {
                    "fit_score": "excellent",
                    "explanation": "High in carbs and calories, great for energy boost"
                }
            elif nutritional_values.carbohydrates > 10 and nutritional_values.fiber > 2:
                return {
                    "fit_score": "good",
                    "explanation": "Good carb content with fiber for sustained energy"
                }
            elif nutritional_values.carbohydrates < 5:
                return {
                    "fit_score": "poor",
                    "explanation": "Low in carbs, not ideal for energy boost"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for energy boost"
                }
                
        elif goal == DietaryGoal.BALANCED_DIET:
            if 0.1 <= protein_ratio <= 0.3 and 0.4 <= carb_ratio <= 0.6 and 0.2 <= fat_ratio <= 0.4:
                return {
                    "fit_score": "excellent",
                    "explanation": "Well-balanced macronutrients, perfect for a balanced diet"
                }
            elif nutritional_values.fiber > 3 and nutritional_values.protein > 5:
                return {
                    "fit_score": "good",
                    "explanation": "Good fiber and protein content, good for a balanced diet"
                }
            else:
                return {
                    "fit_score": "moderate",
                    "explanation": "Moderate fit for a balanced diet"
                }
        
        # Default case
        return {
            "fit_score": "unknown",
            "explanation": "Unable to assess fit for this goal"
        }
    
    def get_pairing(self, food_name: str) -> Optional[str]:
        """Get a healthy pairing for a food item.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            Optional[str]: Suggested pairing or None if no pairing found
        """
        # Find the food in the pairings dictionary
        for food, pairings in self.pairings.items():
            if food.lower() in food_name.lower():
                if pairings:
                    return random.choice(pairings)
        
        # If no specific pairing found, suggest a generic one
        generic_pairings = ["vegetables", "fruits", "nuts", "seeds", "whole grains", "lean protein"]
        return random.choice(generic_pairings)
    
    def get_substitution(self, food_name: str) -> Optional[str]:
        """Get a healthy substitution for a food item.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            Optional[str]: Suggested substitution or None if no substitution found
        """
        # Find the food in the substitutions dictionary
        for food, substitutions in self.substitutions.items():
            if food.lower() in food_name.lower():
                if substitutions:
                    return random.choice(substitutions)
        
        return None
    
    def get_summary(self, food_name: str, weight_grams: float, goal: DietaryGoal) -> Dict[str, str]:
        """Get a friendly summary with emojis and brief notes.
        
        Args:
            food_name (str): Name of the food
            weight_grams (float): Weight in grams
            goal (DietaryGoal): User's dietary goal
            
        Returns:
            Dict[str, str]: Summary with emojis and notes
        """
        nutritional_values = self.get_nutritional_values(food_name, weight_grams)
        
        if nutritional_values is None:
            return {
                "title": "❓ Food Not Found",
                "summary": "I couldn't find this food in my database. Please try a different name or check your spelling."
            }
        
        goal_fit = self.get_goal_fit(food_name, weight_grams, goal)
        pairing = self.get_pairing(food_name)
        substitution = self.get_substitution(food_name)
        
        # Emoji mapping for fit scores
        fit_emojis = {
            "excellent": "🌟",
            "good": "👍",
            "moderate": "😐",
            "poor": "⚠️",
            "unknown": "❓"
        }
        
        # Emoji mapping for goals
        goal_emojis = {
            DietaryGoal.WEIGHT_LOSS: "⚖️",
            DietaryGoal.MUSCLE_GAIN: "💪",
            DietaryGoal.MAINTENANCE: "🔄",
            DietaryGoal.HEALTH_IMPROVEMENT: "❤️",
            DietaryGoal.ENERGY_BOOST: "⚡",
            DietaryGoal.BALANCED_DIET: "🥗"
        }
        
        # Create title
        title = f"{fit_emojis.get(goal_fit['fit_score'], '❓')} {food_name.capitalize()} ({weight_grams}g) {goal_emojis.get(goal, '🎯')}"
        
        # Create summary
        summary = f"{goal_fit['explanation']}\n\n"
        
        # Add nutritional highlights
        if nutritional_values.protein > 10:
            summary += f"🥩 High in protein: {nutritional_values.protein:.1f}g\n"
        if nutritional_values.fiber > 5:
            summary += f"🌾 High in fiber: {nutritional_values.fiber:.1f}g\n"
        if nutritional_values.calories > 300:
            summary += f"🔥 High in calories: {nutritional_values.calories:.1f} kcal\n"
        if nutritional_values.sugar > 10:
            summary += f"🍬 High in sugar: {nutritional_values.sugar:.1f}g\n"
        if nutritional_values.sodium > 500:
            summary += f"🧂 High in sodium: {nutritional_values.sodium:.1f}mg\n"
        
        # Add pairing suggestion
        if pairing:
            summary += f"\n🍽️ Try pairing with: {pairing}\n"
        
        # Add substitution suggestion
        if substitution:
            summary += f"🔄 Consider substituting with: {substitution}\n"
        
        # Add goal-specific tips
        if goal == DietaryGoal.WEIGHT_LOSS:
            summary += "\n💡 Tip: Focus on portion control and high-fiber foods to feel full longer."
        elif goal == DietaryGoal.MUSCLE_GAIN:
            summary += "\n💡 Tip: Combine with a protein source if this food is low in protein."
        elif goal == DietaryGoal.MAINTENANCE:
            summary += "\n💡 Tip: Balance this food with other food groups throughout the day."
        elif goal == DietaryGoal.HEALTH_IMPROVEMENT:
            summary += "\n💡 Tip: Look for whole food versions with minimal processing."
        elif goal == DietaryGoal.ENERGY_BOOST:
            summary += "\n💡 Tip: Pair with protein to prevent energy crashes."
        elif goal == DietaryGoal.BALANCED_DIET:
            summary += "\n💡 Tip: Include a variety of food groups to ensure balanced nutrition."
        
        return {
            "title": title,
            "summary": summary
        }

def main():
    # Initialize the nutrition assistant
    assistant = NutritionAssistant()
    
    # Example usage
    print("\nSmart Nutrition Assistant")
    print("------------------------")
    
    # Get food name from user
    food_name = input("Enter food name: ")
    
    # Get weight from user
    try:
        weight = float(input("Enter weight in grams: "))
    except ValueError:
        print("Error: Please enter a valid weight number")
        return
    
    # Get goal from user
    print("\nDietary Goals:")
    print("1. Weight Loss")
    print("2. Muscle Gain")
    print("3. Maintenance")
    print("4. Health Improvement")
    print("5. Energy Boost")
    print("6. Balanced Diet")
    
    try:
        goal_choice = int(input("Enter goal number (1-6): "))
        if 1 <= goal_choice <= 6:
            goal = list(DietaryGoal)[goal_choice - 1]
        else:
            print("Invalid goal choice, using Balanced Diet")
            goal = DietaryGoal.BALANCED_DIET
    except ValueError:
        print("Invalid input, using Balanced Diet")
        goal = DietaryGoal.BALANCED_DIET
    
    # Get nutritional values
    nutritional_values = assistant.get_nutritional_values(food_name, weight)
    
    if nutritional_values:
        print("\nNutritional Values:")
        print(f"Calories: {nutritional_values.calories:.1f} kcal")
        print(f"Protein: {nutritional_values.protein:.1f}g")
        print(f"Carbohydrates: {nutritional_values.carbohydrates:.1f}g")
        print(f"Fat: {nutritional_values.fat:.1f}g")
        print(f"Fiber: {nutritional_values.fiber:.1f}g")
        if nutritional_values.sugar > 0:
            print(f"Sugar: {nutritional_values.sugar:.1f}g")
        if nutritional_values.sodium > 0:
            print(f"Sodium: {nutritional_values.sodium:.1f}mg")
    else:
        print(f"\nError: Food '{food_name}' not found in database")
        return
    
    # Get goal fit
    goal_fit = assistant.get_goal_fit(food_name, weight, goal)
    
    print(f"\nFit with {goal.value.replace('_', ' ').title()} Goal:")
    print(f"Score: {goal_fit['fit_score'].capitalize()}")
    print(f"Explanation: {goal_fit['explanation']}")
    
    # Get pairing
    pairing = assistant.get_pairing(food_name)
    if pairing:
        print(f"\nHealthy Pairing: {pairing}")
    
    # Get substitution
    substitution = assistant.get_substitution(food_name)
    if substitution:
        print(f"Healthy Substitution: {substitution}")
    
    # Get summary
    summary = assistant.get_summary(food_name, weight, goal)
    
    print("\n" + "=" * 50)
    print(summary["title"])
    print("=" * 50)
    print(summary["summary"])
    print("=" * 50)

if __name__ == "__main__":
    main() 