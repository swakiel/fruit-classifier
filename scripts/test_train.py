import setup_path
from train_model import train

def main():
    """
    Simple script to train a model on a selected dataset.
    """
    # Example: train FruitNet for 5 epochs
    print("Training FruitNet model...")
    train("fruitnet", epochs=5)

    # Example: train Fruits-360 for 5 epochs
    print("Training Fruits-360 model...")
    train("fruits360", epochs=5)

if __name__ == "__main__":
    main()