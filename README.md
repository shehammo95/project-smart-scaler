# Food Data Collection System

A system for collecting and processing food data from various sources including USDA and OpenFoodFacts APIs.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
USDA_API_KEY=your_usda_api_key
OPENFOODFACTS_API_KEY=your_openfoodfacts_api_key
```

3. Initialize data directories:
```bash
python data_collection/setup.py
```

## Usage

To collect data for a list of food items, run:
```bash
python data_collection/main.py
```

The script will:
- Collect nutritional data from USDA and OpenFoodFacts
- Download and process food images
- Save all data in the `data` directory

## Directory Structure

- `data/food_data/`: Contains JSON files with nutritional information
- `data/food_images/`: Contains processed food images
- `data/logs/`: Contains log files

## Data Collection Process

1. For each food item:
   - Collect data from USDA API
   - Collect data from OpenFoodFacts API
   - Process and merge the data
   - Download and process images
   - Save all data to appropriate directories

## Error Handling

The system includes comprehensive error handling and logging. Check the `data/logs` directory for detailed logs of the collection process. 