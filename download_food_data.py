import os
import json
import shutil
import pandas as pd
from datetime import datetime

def create_food_data_folder():
    # Create main folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_folder = f"food_data_{timestamp}"
    os.makedirs(main_folder, exist_ok=True)
    
    # Create subfolders
    subfolders = [
        "raw_data",
        "processed_data",
        "images",
        "reports",
        "api_data",
        "scale_data",
        "nutrition_data",
        "recipes"
    ]
    
    for folder in subfolders:
        os.makedirs(os.path.join(main_folder, folder), exist_ok=True)
    
    return main_folder

def process_food_csv(main_folder):
    # Read and process the food CSV
    if os.path.exists("data/food.csv"):
        df = pd.read_csv("data/food.csv")
        
        # Save processed versions
        processed_dir = os.path.join(main_folder, "processed_data")
        os.makedirs(processed_dir, exist_ok=True)
        
        df.to_csv(os.path.join(processed_dir, "food_processed.csv"), index=False)
        
        # Create category-based files
        categories = df['Category'].unique()
        for category in categories:
            category_df = df[df['Category'] == category]
            # Clean category name for filename
            clean_category = ''.join(c for c in category.lower() if c.isalnum() or c in (' ', '_')).strip()
            clean_category = clean_category.replace(' ', '_')
            category_file = f"food_{clean_category}.csv"
            category_df.to_csv(os.path.join(processed_dir, category_file), index=False)

def copy_data_files(main_folder):
    # Copy existing data files
    data_files = [
        "data/food_database.json",
        "data/food.csv",
        "data/food_substitutions.json",
        "data/food_pairings.json"
    ]
    
    raw_data_dir = os.path.join(main_folder, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    for file in data_files:
        if os.path.exists(file):
            shutil.copy2(file, raw_data_dir)
    
    # Process food CSV data
    process_food_csv(main_folder)
    
    # Create additional data files
    create_additional_data_files(main_folder)
    
    # Create a comprehensive README file
    create_readme(main_folder)

def create_additional_data_files(main_folder):
    # Create nutrition guidelines
    nutrition_guidelines = {
        "daily_values": {
            "calories": 2000,
            "protein": 50,
            "carbs": 275,
            "fat": 78,
            "fiber": 28,
            "sugar": 50,
            "sodium": 2300
        },
        "vitamins": {
            "vitamin_a": 900,
            "vitamin_c": 90,
            "vitamin_d": 20,
            "vitamin_e": 15,
            "vitamin_k": 120,
            "thiamin": 1.2,
            "riboflavin": 1.3,
            "niacin": 16,
            "vitamin_b6": 1.7,
            "folate": 400,
            "vitamin_b12": 2.4
        },
        "minerals": {
            "calcium": 1300,
            "iron": 18,
            "magnesium": 420,
            "phosphorus": 1250,
            "potassium": 4700,
            "sodium": 2300,
            "zinc": 11
        }
    }
    
    nutrition_dir = os.path.join(main_folder, "nutrition_data")
    os.makedirs(nutrition_dir, exist_ok=True)
    
    with open(os.path.join(nutrition_dir, "nutrition_guidelines.json"), "w") as f:
        json.dump(nutrition_guidelines, f, indent=4)

def create_readme(main_folder):
    readme_content = f"""Food Data Collection
===================

This folder contains comprehensive food-related data collected on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Folder Structure:
- raw_data/: Original data files
- processed_data/: Processed and cleaned data
- images/: Food images
- reports/: Analysis reports
- api_data/: API-related data
- scale_data/: Scale measurement data
- nutrition_data/: Nutritional guidelines and standards
- recipes/: Recipe data

Data Sources:
1. USDA Food Database
2. Custom Food Substitutions
3. Custom Food Pairings
4. Nutritional Guidelines
5. Scale Measurements

File Descriptions:
- food.csv: Raw USDA food database
- food_database.json: Combined food data in JSON format
- food_substitutions.json: Healthy food substitutions
- food_pairings.json: Food pairing recommendations
- nutrition_guidelines.json: Daily nutritional requirements

For more information, please refer to the documentation in each folder.
"""
    
    with open(os.path.join(main_folder, "README.md"), "w") as f:
        f.write(readme_content)

def main():
    print("Creating food data folder structure...")
    main_folder = create_food_data_folder()
    
    print("Copying and processing data files...")
    copy_data_files(main_folder)
    
    print(f"\nData has been downloaded to: {os.path.abspath(main_folder)}")
    print("\nFolder structure created:")
    for root, dirs, files in os.walk(main_folder):
        level = root.replace(main_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    main() 