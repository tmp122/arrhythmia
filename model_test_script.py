import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Add Keras serialization decorator for the custom TransformerBlock layer
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

# Configuration
SAMPLE_LENGTH = 200
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=5):
    """Apply a bandpass filter to the signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess_signal(signal, sample_length=SAMPLE_LENGTH):
    """Preprocess a signal by filtering, normalizing, and segmenting"""
    # Apply bandpass filter
    filtered = butter_bandpass_filter(signal)
    
    # Normalize
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-7)
    
    # Ensure length is correct
    if len(filtered) < sample_length:
        # Pad if too short
        padding = sample_length - len(filtered)
        filtered = np.pad(filtered, (0, padding), 'constant')
    elif len(filtered) > sample_length:
        # Take center segment if too long
        start = (len(filtered) - sample_length) // 2
        filtered = filtered[start:start+sample_length]
    
    return filtered

def segment_signal(signal, sample_length=SAMPLE_LENGTH, step=50):
    """Segment a long signal into overlapping windows"""
    segments = []
    for i in range(0, max(1, len(signal) - sample_length + 1), step):
        segment = signal[i:i+sample_length]
        if len(segment) == sample_length:
            segments.append(segment)
    
    # If signal is shorter than sample_length, pad it
    if len(segments) == 0 and len(signal) > 0:
        padded = np.pad(signal, (0, sample_length - len(signal)), 'constant')
        segments.append(padded)
        
    return np.array(segments)

def load_models():
    """Load all trained models"""
    models = {}
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    try:
        for model_name in ['cnn', 'lstm', 'transformer']:
            model_path = f'{model_name}_model.keras'
            if os.path.exists(model_path):
                models[model_name] = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                print(f"Successfully loaded {model_name.upper()} model")
            else:
                print(f"Could not find {model_name.upper()} model at {model_path}")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
    
    return models

def ensemble_predict(models, X_data):
    """Get ensemble predictions from multiple models"""
    if not models:
        print("No models available for prediction!")
        return None
    
    # Collect predictions from each model
    all_preds = []
    for name, model in models.items():
        print(f"Getting predictions from {name.upper()} model...")
        y_pred = model.predict(X_data)
        all_preds.append(y_pred)
    
    # Average the predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds

def test_with_file(file_path, models):
    """Test models using data from a CSV file"""
    print(f"\nTesting with file: {file_path}")
    
    try:
        # Load the data from CSV
        if file_path.endswith(('.csv', '.txt')):
            df = pd.read_csv(file_path)
            
            # If there's only one column, use it directly
            if len(df.columns) == 1:
                signal = df.iloc[:, 0].values
            else:
                # Use the first non-time column
                signal_col = [col for col in df.columns if 'time' not in col.lower()][0]
                print(f"Using column: {signal_col}")
                signal = df[signal_col].values
        else:
            print(f"Unsupported file format: {file_path}")
            return
        
        # Visualize the signal
        plt.figure(figsize=(12, 4))
        plt.plot(signal[:1000])  # Plot first 1000 samples
        plt.title(f'Signal from {os.path.basename(file_path)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(f'test_result_{os.path.basename(file_path).split(".")[0]}_signal.png')
        plt.close()
        
        # Segment and preprocess the signal
        segments = segment_signal(signal)
        
        # Normalize each segment
        normalized_segments = np.array([
            (segment - np.mean(segment)) / (np.std(segment) + 1e-7)
            for segment in segments
        ])
        
        # Make predictions
        predictions = ensemble_predict(models, normalized_segments)
        
        if predictions is not None:
            # Get the predicted classes
            pred_classes = np.argmax(predictions, axis=1)
            
            # Count occurrences of each class
            class_counts = np.bincount(pred_classes, minlength=len(CLASS_NAMES))
            
            # Create a dataframe for the class distribution
            df_distribution = pd.DataFrame({
                'Arrhythmia Type': CLASS_NAMES,
                'Count': class_counts,
                'Percentage': class_counts / np.sum(class_counts) * 100
            })
            
            # Get the dominant class (mode)
            dominant_class = CLASS_NAMES[np.argmax(class_counts)]
            
            # Display summary information
            print(f"\nDominant arrhythmia type detected: {dominant_class}")
            print("\nDistribution of Detected Arrhythmia Types:")
            print(df_distribution)
            
            # Generate and save visualizations
            
            # Distribution as bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Arrhythmia Type', y='Percentage', data=df_distribution)
            plt.ylabel('Percentage %')
            plt.title(f'Distribution of Detected Arrhythmia Types in {os.path.basename(file_path)}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'test_result_{os.path.basename(file_path).split(".")[0]}_distribution.png')
            plt.close()
            
            # Signal with classification
            plt.figure(figsize=(15, 6))
            plt.plot(signal, color='gray', alpha=0.7)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
            segment_indices = list(range(0, len(segments)*50, 50))
            
            for i, (start_idx, pred_class) in enumerate(zip(segment_indices, pred_classes)):
                if i < len(segment_indices):
                    end_idx = min(start_idx + SAMPLE_LENGTH, len(signal))
                    plt.axvspan(start_idx, end_idx, alpha=0.3, color=colors[pred_class], 
                              label=f"{CLASS_NAMES[pred_class]}" if CLASS_NAMES[pred_class] not in plt.gca().get_legend_handles_labels()[1] else "")
            
            plt.title(f'ECG Signal with Classified Segments - {os.path.basename(file_path)}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'test_result_{os.path.basename(file_path).split(".")[0]}_classified.png')
            plt.close()
            
            return df_distribution
            
    except Exception as e:
        print(f"Error testing with {file_path}: {str(e)}")
        return None

def test_individual_models(file_path, models):
    """Test each model separately and compare their performance"""
    print(f"\nTesting individual models with file: {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            signal = df.iloc[:, 0].values
        else:
            signal_col = [col for col in df.columns if 'time' not in col.lower()][0]
            signal = df[signal_col].values
        
        # Segment and preprocess
        segments = segment_signal(signal)
        normalized_segments = np.array([
            (segment - np.mean(segment)) / (np.std(segment) + 1e-7)
            for segment in segments
        ])
        
        # Results dictionary
        results = {}
        
        # Test each model individually
        for name, model in models.items():
            print(f"\nTesting {name.upper()} model...")
            predictions = model.predict(normalized_segments)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Count occurrences of each class
            class_counts = np.bincount(pred_classes, minlength=len(CLASS_NAMES))
            
            # Store results
            results[name] = {
                'predictions': predictions,
                'pred_classes': pred_classes,
                'class_counts': class_counts,
                'dominant_class': CLASS_NAMES[np.argmax(class_counts)]
            }
            
            print(f"{name.upper()} dominant class: {results[name]['dominant_class']}")
        
        # Compare models
        plt.figure(figsize=(12, 8))
        
        for i, name in enumerate(models.keys()):
            # Calculate percentage distribution
            percentages = results[name]['class_counts'] / np.sum(results[name]['class_counts']) * 100
            
            plt.subplot(len(models), 1, i+1)
            sns.barplot(x=CLASS_NAMES, y=percentages)
            plt.title(f'{name.upper()} Model Predictions')
            plt.ylabel('Percentage %')
            plt.ylim(0, 100)
            
        plt.tight_layout()
        plt.savefig(f'test_result_{os.path.basename(file_path).split(".")[0]}_model_comparison.png')
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error in individual model testing: {str(e)}")
        return None

def main():
    # Load models
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models loaded. Please ensure model files are in the current directory.")
        return
    
    # Check for test data directory
    if not os.path.exists('test_data'):
        print("Test data directory not found. Running generate_test_data.py to create test data...")
        from generate_test_data import generate_test_files
        generate_test_files()
    
    # Test with each file in the test_data directory
    test_files = [f for f in os.listdir('test_data') if f.endswith(('.csv', '.txt'))]
    
    if not test_files:
        print("No test files found.")
        return
    
    print(f"Found {len(test_files)} test files.")
    
    # Test ensemble prediction with each file
    for file in test_files:
        file_path = os.path.join('test_data', file)
        test_with_file(file_path, models)
    
    # Choose one file for individual model comparison
    if test_files:
        comparison_file = os.path.join('test_data', test_files[0])
        print(f"\nComparing individual model performance on {comparison_file}...")
        test_individual_models(comparison_file, models)
    
    print("\nTesting completed. Results saved as PNG files.")

if __name__ == "__main__":
    main()
