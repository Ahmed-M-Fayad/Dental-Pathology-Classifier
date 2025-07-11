import os
import matplotlib.pyplot as plt
import tensorflow as tf
import zipfile
from pathlib import Path


# Refined delivery pipeline
class DatasetPipeline:
    """
    A streamlined pipeline for teeth classification dataset processing.
    Save once with pickle, load later, and run with a single method.
    """

    def __init__(
        self,
        drive_dataset_path="/content/drive/MyDrive/Datasets/Teeth DataSet.zip",
        image_size=(256, 256),
        batch_size=32,
    ):
        """
        Initialize the dataset pipeline.

        Args:
            drive_dataset_path (str): Path to dataset ZIP file in Google Drive
            image_size (tuple): Target image dimensions (height, width)
            batch_size (int): Batch size for training
        """
        self.drive_dataset_path = drive_dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.main_dir = None
        self.class_names = None
        self.extraction_dir = "/content/drive/MyDrive/Datasets/Extracted Dataset"

        # Dataset objects and stats
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.stats = None

    def load_dataset(self):
        """Mount Google Drive and extract dataset from ZIP file."""
        # Mount Google Drive

        # Check if already mounted
        if not os.path.ismount("/content/drive"):
            print("Google Drive is not mounted.")
        else:
            print("Google Drive already mounted.")

        # Check if dataset ZIP exists
        if not os.path.exists(self.drive_dataset_path):
            raise FileNotFoundError(
                f"Dataset ZIP not found at: {self.drive_dataset_path}"
            )

        # Create extraction directory if it doesn't exist
        os.makedirs(self.extraction_dir, exist_ok=True)

        # Extract dataset
        print(f"Extracting dataset from {self.drive_dataset_path}...")
        with zipfile.ZipFile(self.drive_dataset_path, "r") as zip_ref:
            zip_ref.extractall(self.extraction_dir)

        # Expected structure: Teeth_Dataset/Training, Teeth_Dataset/Validation, Teeth_Dataset/Testing
        self.main_dir = os.path.join(self.extraction_dir, "Teeth_Dataset")

        if not os.path.exists(self.main_dir):
            raise FileNotFoundError(
                f"Expected 'Teeth_Dataset' directory not found at: {self.main_dir}"
            )

        if not self._validate_dataset_structure(self.main_dir):
            raise ValueError(
                f"Invalid dataset structure in {self.main_dir}. "
                "Expected subdirectories: Training, Validation, Testing"
            )

        print(f"Dataset successfully extracted to: {self.main_dir}")
        return self.main_dir

    def _validate_dataset_structure(self, directory):
        """Validate that directory contains required subdirectories."""
        required_dirs = ["Training", "Validation", "Testing"]
        return all(os.path.exists(os.path.join(directory, d)) for d in required_dirs)

    def analyze_dataset(self):
        """Analyze dataset structure and provide statistics."""
        if self.main_dir is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        train_dir = os.path.join(self.main_dir, "Training")
        val_dir = os.path.join(self.main_dir, "Validation")
        test_dir = os.path.join(self.main_dir, "Testing")

        # Count images efficiently
        def count_images(directory):
            return sum(
                1
                for f in Path(directory).rglob("*")
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
            )

        train_count = count_images(train_dir)
        val_count = count_images(val_dir)
        test_count = count_images(test_dir)

        # Get class information
        classes = sorted(
            [
                d
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        )
        self.class_names = classes

        self.stats = {
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "total_images": train_count + val_count + test_count,
            "num_classes": len(classes),
            "class_names": classes,
        }

        # Display statistics
        print("\n" + "=" * 50)
        print("DATASET ANALYSIS")
        print("=" * 50)
        print(f"Training images: {self.stats['train_images']}")
        print(f"Validation images: {self.stats['val_images']}")
        print(f"Test images: {self.stats['test_images']}")
        print(f"Total images: {self.stats['total_images']}")
        print(f"Number of classes: {self.stats['num_classes']}")
        print(f"Classes: {', '.join(classes)}")
        print("=" * 50)

        return self.stats

    def create_datasets(self):
        """Create optimized TensorFlow datasets for training, validation, and testing."""
        if self.main_dir is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Dataset directories
        train_dir = os.path.join(self.main_dir, "Training")
        val_dir = os.path.join(self.main_dir, "Validation")
        test_dir = os.path.join(self.main_dir, "Testing")

        # Create datasets with optimizations
        train_data = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'],
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'],
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_data = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'],
            image_size=(256, 256),
            batch_size=32,
            shuffle=False
        )
        # Store datasets in pipeline
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.class_names = train_data.class_names

        print(f"\nTensorFlow datasets created successfully!")
        print(f"Training batches: {len(train_data)}")
        print(f"Validation batches: {len(val_data)}")
        print(f"Test batches: {len(test_data)}")

        return train_data, val_data, test_data

    def visualize_samples(self, max_samples=12):
        """Visualize sample images from each class."""
        if self.train_data is None:
            raise ValueError("No training data available. Run the pipeline first.")

        plt.figure(figsize=(15, 10))

        # Collect one sample per class
        samples_collected = {}
        for image_batch, label_batch in self.train_data.take(5):  # Take a few batches
            for i in range(len(image_batch)):
                class_idx = tf.argmax(label_batch[i]).numpy()
                class_name = self.class_names[class_idx]

                if class_name not in samples_collected:
                    samples_collected[class_name] = (
                        image_batch[i].numpy().astype("uint8")
                    )

                if len(samples_collected) >= min(max_samples, len(self.class_names)):
                    break

            if len(samples_collected) >= min(max_samples, len(self.class_names)):
                break

        # Display samples
        num_samples = len(samples_collected)
        cols = 4
        rows = (num_samples + cols - 1) // cols

        for idx, (class_name, image) in enumerate(samples_collected.items()):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(image)
            plt.title(class_name, fontsize=12)
            plt.axis("off")

        plt.suptitle("Training Dataset Samples", fontsize=16)
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Execute the complete pipeline - the main method to run everything.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, stats)
        """
        print("Starting Dataset Pipeline...")

        # Step 1: Load dataset
        self.load_dataset()

        # Step 2: Analyze dataset
        self.analyze_dataset()

        # Step 3: Create TensorFlow datasets
        self.create_datasets()

        # Step 4: Visualize samples
        print("\nGenerating sample visualization...")
        self.visualize_samples()

        print("\nPipeline completed!")
        return self.train_data, self.val_data, self.test_data, self.stats

    def __getstate__(self):
        """Custom method for pickling - exclude TensorFlow datasets"""
        state = self.__dict__.copy()
        # Remove TensorFlow datasets as they can't be pickled
        state["train_data"] = None
        state["val_data"] = None
        state["test_data"] = None
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
