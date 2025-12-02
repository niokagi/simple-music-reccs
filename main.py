import sys
from src.train_model import train
from src.recommender import MusicRecommender

def main():
    print("1. Train New Model (Retrain)")
    print("2. Generate Music Recommendations")
    
    choice = input("Select an option (1/2): ")
    
    if choice == '1':
        print("\nInitializing training pipeline...")
        try:
            train()
            print("\nSuccess: Model successfully trained and artifacts saved to 'models/' directory.")
        except Exception as e:
            print(f"\nCritical Error during training: {e}")
            
    elif choice == '2':
        try:
            # Initialize Recommender (Load model into RAM)
            rec_engine = MusicRecommender()
            
            while True:
                query = input("\nEnter song title (or type 'exit' to quit): ")
                if query.lower() == 'exit':
                    print("Exiting application. Goodbye!")
                    break
                
                print(f"Searching for matches for '{query}'...")
                result = rec_engine.recommend(query)
                print("-" * 60)
                print(result)
                print("-" * 60)
                
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Please execute the training phase (Option 1) first to generate necessary artifacts.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            
    else:
        print("Invalid selection. Please restart and choose 1 or 2.")

if __name__ == "__main__":
    main()