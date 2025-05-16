import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
import os
import tempfile
import seaborn as sns
from scipy.signal import butter, filtfilt
import io
import base64
from PIL import Image

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

# Preprocessing functions
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
                st.success(f"Successfully loaded {model_name.upper()} model")
            else:
                st.warning(f"Could not find {model_name.upper()} model at {model_path}")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models

def ensemble_predict(models, X_data):
    """Get ensemble predictions from multiple models"""
    if not models:
        st.error("No models available for prediction!")
        return None
    
    # Collect predictions from each model
    all_preds = []
    for name, model in models.items():
        with st.spinner(f"Getting predictions from {name.upper()} model..."):
            y_pred = model.predict(X_data)
            all_preds.append(y_pred)
    
    # Average the predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds

def display_prediction_results(predictions, segment_indices=None, signal=None):
    """Display prediction results as charts and tables"""
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
    st.subheader("Classification Results")
    st.markdown(f"**Dominant arrhythmia type detected: {dominant_class}**")
    
    # Display distribution as bar chart
    st.subheader("Distribution of Detected Arrhythmia Types")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Arrhythmia Type', y='Percentage', data=df_distribution, ax=ax)
    ax.set_ylabel('Percentage %')
    ax.set_title('Distribution of Detected Arrhythmia Types')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Display as a table
    st.subheader("Detailed Classification Results")
    st.table(df_distribution)
    
    # If we have segments and original signal, show segment classifications
    if segment_indices is not None and signal is not None:
        st.subheader("Signal with Classified Segments")
        
        # Plot the signal with colored regions for each segment
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(signal, color='gray', alpha=0.7)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
        
        for i, (start_idx, pred_class) in enumerate(zip(segment_indices, pred_classes)):
            end_idx = start_idx + SAMPLE_LENGTH
            ax.axvspan(start_idx, end_idx, alpha=0.3, color=colors[pred_class], 
                     label=f"{CLASS_NAMES[pred_class]}" if i == 0 else "")
            
        ax.set_title('ECG Signal with Classified Segments')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        
        # Create custom legend entries
        handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.3) for i in range(len(CLASS_NAMES))]
        ax.legend(handles, CLASS_NAMES, loc='upper right')
        
        st.pyplot(fig)

def get_sample_ecg():
    """Return a sample ECG signal for demonstration"""
    # Generate a synthetic ECG-like signal
    t = np.linspace(0, 10, 3600)  # 10 seconds at 360 Hz
    
    # Base signal
    signal = np.zeros_like(t)
    
    # Add P waves
    for i in range(10):
        center = i + 0.2
        signal += 0.2 * np.exp(-((t - center) ** 2) / 0.01)
    
    # Add QRS complexes
    for i in range(10):
        center = i + 0.5
        signal += np.exp(-((t - center) ** 2) / 0.001)
        signal -= 0.3 * np.exp(-((t - center + 0.05) ** 2) / 0.002)
        signal += 0.5 * np.exp(-((t - center + 0.1) ** 2) / 0.003)
    
    # Add T waves
    for i in range(10):
        center = i + 0.8
        signal += 0.4 * np.exp(-((t - center) ** 2) / 0.02)
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(t))
    
    return signal

def parse_csv_data(content):
    """Parse CSV content into a numpy array"""
    try:
        # Try to read as a CSV with header
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # If there's only one column, use it directly
        if len(df.columns) == 1:
            signal = df.iloc[:, 0].values
        else:
            # Ask the user which column to use for the ECG signal
            st.info("Multiple columns detected in the CSV file.")
            selected_column = st.selectbox("Select the column containing the ECG signal:", df.columns)
            signal = df[selected_column].values
            
        return signal
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None

def parse_wfdb_data(record_path, annotation_path=None):
    """Parse WFDB format data"""
    try:
        # Read the record
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # Use the first channel
        
        # If annotation is provided, read it too
        annotations = None
        if annotation_path and os.path.exists(annotation_path):
            annotations = wfdb.rdann(annotation_path, 'atr')
        
        return signal, annotations
    except Exception as e:
        st.error(f"Error parsing WFDB data: {str(e)}")
        return None, None

# Main app
def main():
    st.set_page_config(
        page_title="ECG Arrhythmia Classifier",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("❤️ ECG Arrhythmia Classification App")
    st.markdown("""
    This application uses an ensemble of deep learning models (CNN, LSTM, and Transformer) 
    to classify ECG signals into five categories:
    - **Normal**: Normal sinus rhythm
    - **Supraventricular**: Supraventricular arrhythmias
    - **Ventricular**: Ventricular arrhythmias
    - **Fusion**: Fusion beats
    - **Unknown**: Unknown beats
    """)
    
    # Initialize session state
    if 'models' not in st.session_state:
        st.session_state.models = None
    
    # Load models button
    if st.button("Load Models") or st.session_state.models is not None:
        if st.session_state.models is None:
            st.session_state.models = load_models()
    
    # Check if models are loaded
    if st.session_state.models is None or len(st.session_state.models) == 0:
        st.warning("Please load the models first")
    else:
        st.success(f"Models loaded: {', '.join([name.upper() for name in st.session_state.models.keys()])}")
        
        # Input selection
        st.subheader("Select Input Data")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload ECG data", "Use sample data"]
        )
        
        signal = None
        
        if input_method == "Upload ECG data":
            st.write("Upload an ECG signal file (CSV, TXT, or WFDB format)")
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "dat", "hea"])
            
            if uploaded_file is not None:
                # Process based on file type
                if uploaded_file.name.endswith(('.csv', '.txt')):
                    # Read as CSV/TXT
                    content = uploaded_file.read()
                    signal = parse_csv_data(content)
                elif uploaded_file.name.endswith(('.dat', '.hea')):
                    # Read as WFDB
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Save the uploaded file
                        file_path = os.path.join(tmpdirname, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.read())
                            
                        # If it's a header file, look for corresponding dat file
                        base_name = os.path.splitext(file_path)[0]
                        signal, _ = parse_wfdb_data(base_name)
        
        else:  # Use sample data
            signal = get_sample_ecg()
            st.info("Using sample ECG data")
        
        if signal is not None:
            # Show original signal
            st.subheader("Original ECG Signal")
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(signal)
            ax.set_title('Original ECG Signal')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig)
            
            # Preprocess and classify
            if st.button("Analyze ECG"):
                with st.spinner("Preprocessing signal..."):
                    # Segment the signal
                    segment_indices = list(range(0, len(signal) - SAMPLE_LENGTH + 1, 50))
                    segments = []
                    
                    for idx in segment_indices:
                        segment = signal[idx:idx+SAMPLE_LENGTH]
                        # Normalize
                        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-7)
                        segments.append(segment)
                    
                    segments = np.array(segments)
                    
                    # If no segments created (signal too short), process the whole signal
                    if len(segments) == 0:
                        segments = np.array([preprocess_signal(signal)])
                        segment_indices = [0]
                
                with st.spinner("Classifying ECG segments..."):
                    # Make predictions
                    predictions = ensemble_predict(st.session_state.models, segments)
                    
                    if predictions is not None:
                        # Display results
                        display_prediction_results(predictions, segment_indices, signal)
                    else:
                        st.error("Could not generate predictions. Please check the models and input data.")

if __name__ == "__main__":
    main()
