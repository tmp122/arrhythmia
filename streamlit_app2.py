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
from scipy.signal import butter, filtfilt, resample
import io
import base64

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
SAMPLING_RATE = 360  # MIT-BIH standard sampling rate

# Preprocessing functions
def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=5):
    """Apply a bandpass filter to the signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_signal(signal):
    """Normalize signal to have zero mean and unit variance"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-7)

def preprocess_signal(signal, sample_length=SAMPLE_LENGTH):
    """Preprocess a signal by filtering and normalizing"""
    # Apply bandpass filter
    filtered = butter_bandpass_filter(signal)
    
    # Normalize
    filtered = normalize_signal(filtered)
    
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
    segment_indices = []
    
    for i in range(0, max(1, len(signal) - sample_length + 1), step):
        segment = signal[i:i+sample_length]
        if len(segment) == sample_length:
            # Apply preprocessing to each segment
            processed_segment = butter_bandpass_filter(segment)
            processed_segment = normalize_signal(processed_segment)
            
            segments.append(processed_segment)
            segment_indices.append(i)
    
    # If signal is shorter than sample_length, pad it
    if len(segments) == 0 and len(signal) > 0:
        padded = np.pad(signal, (0, sample_length - len(signal)), 'constant')
        processed_segment = normalize_signal(butter_bandpass_filter(padded))
        segments.append(processed_segment)
        segment_indices.append(0)
        
    return np.array(segments), segment_indices

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
    model_predictions = {}

    for name, model in models.items():
        with st.spinner(f"Getting predictions from {name.upper()} model..."):
            try:
                # Make sure input shape is correct
                if X_data.shape[1] != SAMPLE_LENGTH:
                    X_data_reshaped = np.zeros((X_data.shape[0], SAMPLE_LENGTH))
                    for i in range(X_data.shape[0]):
                        X_data_reshaped[i] = resample(X_data[i], SAMPLE_LENGTH)
                    X_data = X_data_reshaped
                
                y_pred = model.predict(X_data)
                
                # Automatic bias correction for transformer
                if name == 'transformer':
                    # Apply fixed correction factor (0.5 reduces ventricular predictions by 50%)
                    redistribution = y_pred[:, 2] * 0.5  # Class 2 is ventricular
                    y_pred[:, 2] -= redistribution
                    y_pred[:, 0] += redistribution  # Add to normal class
                    
                    # Renormalize probabilities
                    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
                
                # Store predictions with weighting
                weight = 0.5 if name == 'transformer' else 1.0
                all_preds.append(y_pred * weight)
                
                # Display individual model predictions
                st.write(f"{name.upper()} model predictions:")
                pred_classes = np.argmax(y_pred, axis=1)
                class_counts = np.bincount(pred_classes, minlength=len(CLASS_NAMES))
                class_pcts = class_counts / np.sum(class_counts) * 100
                
                model_results = pd.DataFrame({
                    'Arrhythmia Type': CLASS_NAMES,
                    'Count': class_counts,
                    'Percentage': class_pcts
                })
                st.dataframe(model_results)
                
            except Exception as e:
                st.error(f"Error with {name} model: {str(e)}")
                continue

    if not all_preds:
        return None

    # Weighted average based on model weights
    ensemble_preds = np.sum(all_preds, axis=0) / sum(0.5 if name == 'transformer' else 1.0 
                                                     for name in models.keys())
    
    return ensemble_preds
def display_prediction_results(predictions, segment_indices=None, signal=None):
    """Display prediction results as charts and tables"""
    # Get the predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Show raw probabilities for inspection
    st.subheader("Raw Prediction Probabilities")
    prob_df = pd.DataFrame(predictions, columns=CLASS_NAMES)
    prob_df.index = [f"Segment {i+1}" for i in range(len(predictions))]
    st.dataframe(prob_df)
    
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
            end_idx = min(start_idx + SAMPLE_LENGTH, len(signal))
            ax.axvspan(start_idx, end_idx, alpha=0.3, color=colors[pred_class], 
                       label=f"{CLASS_NAMES[pred_class]}" if CLASS_NAMES[pred_class] not in [l.get_label() for l in ax.get_lines()[1:]] else "")
            
        ax.set_title('ECG Signal with Classified Segments')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        
        # Create custom legend entries
        handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.3) for i in range(len(CLASS_NAMES))]
        ax.legend(handles, CLASS_NAMES, loc='upper right')
        
        st.pyplot(fig)

def get_sample_ecg():
    """Return a sample ECG signal for demonstration"""
    # Generate a synthetic ECG-like signal for normal sinus rhythm
    t = np.linspace(0, 10, 3600)  # 10 seconds at 360 Hz
    
    # Base signal (normal sinus rhythm has regular P-QRS-T waves)
    signal = np.zeros_like(t)
    
    # Normal heart rate around 60-100 BPM (we'll use 75 BPM = 1.25 Hz)
    heart_rate = 75  # beats per minute
    cycle_duration = 60 / heart_rate  # seconds per beat
    
    # Add regular patterns for 10 cardiac cycles
    for i in range(int(10 / cycle_duration)):
        # Timing of each component relative to the R peak
        cycle_start = i * cycle_duration
        
        # P wave (atrial depolarization) - occurs before QRS
        p_time = cycle_start + 0.2
        signal += 0.25 * np.exp(-((t - p_time) ** 2) / 0.005)
        
        # QRS complex (ventricular depolarization)
        qrs_time = cycle_start + 0.5
        # Q wave (small negative deflection)
        signal -= 0.1 * np.exp(-((t - (qrs_time - 0.05)) ** 2) / 0.001)
        # R wave (large positive deflection)
        signal += 1.0 * np.exp(-((t - qrs_time) ** 2) / 0.001)
        # S wave (negative deflection after R)
        signal -= 0.3 * np.exp(-((t - (qrs_time + 0.05)) ** 2) / 0.002)
        
        # T wave (ventricular repolarization)
        t_time = cycle_start + 0.7
        signal += 0.3 * np.exp(-((t - t_time) ** 2) / 0.01)
    
    # Add some noise (much less for a cleaner signal)
    signal += 0.03 * np.random.randn(len(t))
    
    return signal

def get_sample_ventricular_ecg():
    """Return a sample ventricular arrhythmia ECG signal"""
    t = np.linspace(0, 10, 3600)  # 10 seconds at 360 Hz
    signal = np.zeros_like(t)
    
    # Ventricular tachycardia has wider, bizarre QRS complexes
    # Higher rate (around 120-200 BPM)
    heart_rate = 150
    cycle_duration = 60 / heart_rate
    
    for i in range(int(10 / cycle_duration)):
        # No distinct P waves in ventricular tachycardia
        qrs_time = i * cycle_duration + 0.3
        
        # Wider QRS complex
        signal += 1.2 * np.exp(-((t - qrs_time) ** 2) / 0.008)  # Much wider
        signal -= 0.8 * np.exp(-((t - (qrs_time + 0.12)) ** 2) / 0.01)  # Wider S wave
        
        # T wave often in opposite direction to QRS
        t_time = i * cycle_duration + 0.6
        signal -= 0.4 * np.exp(-((t - t_time) ** 2) / 0.015)
    
    # Add some noise and baseline wander
    signal += 0.05 * np.random.randn(len(t))
    signal += 0.2 * np.sin(2 * np.pi * 0.05 * t)  # Baseline wander
    
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

def debug_preprocessing(signal):
    """Show preprocessing steps for debugging"""
    st.subheader("Debug Preprocessing")
    
    # Show original signal statistics
    st.write("Original Signal Statistics:")
    st.write(f"Length: {len(signal)}")
    st.write(f"Mean: {np.mean(signal):.4f}")
    st.write(f"Std: {np.std(signal):.4f}")
    st.write(f"Min: {np.min(signal):.4f}")
    st.write(f"Max: {np.max(signal):.4f}")
    
    # Filter the signal
    filtered = butter_bandpass_filter(signal)
    
    st.write("After Bandpass Filter:")
    st.write(f"Mean: {np.mean(filtered):.4f}")
    st.write(f"Std: {np.std(filtered):.4f}")
    st.write(f"Min: {np.min(filtered):.4f}")
    st.write(f"Max: {np.max(filtered):.4f}")
    
    # Plot original vs filtered
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.plot(signal)
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    
    ax2.plot(filtered)
    ax2.set_title('Filtered Signal')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Normalize the signal
    normalized = normalize_signal(filtered)
    
    st.write("After Normalization:")
    st.write(f"Mean: {np.mean(normalized):.4f}")
    st.write(f"Std: {np.std(normalized):.4f}")
    st.write(f"Min: {np.min(normalized):.4f}")
    st.write(f"Max: {np.max(normalized):.4f}")
    
    # Show a segment
    if len(normalized) >= SAMPLE_LENGTH:
        segment = normalized[:SAMPLE_LENGTH]
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(segment)
        ax.set_title(f'First Segment (Length: {SAMPLE_LENGTH})')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
    
    return normalized

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

    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    # Add model selection in sidebar
    st.sidebar.subheader("Model Selection")
    use_cnn = st.sidebar.checkbox("Use CNN Model", value=True)
    use_lstm = st.sidebar.checkbox("Use LSTM Model", value=True)
    use_transformer = st.sidebar.checkbox("Use Transformer Model", value=True)

    # Load models button
    if st.button("Load Models") or st.session_state.models is not None:
        if st.session_state.models is None:
            all_models = load_models()
            # Filter models based on selection
            st.session_state.models = {}
            if use_cnn and 'cnn' in all_models:
                st.session_state.models['cnn'] = all_models['cnn']
            if use_lstm and 'lstm' in all_models:
                st.session_state.models['lstm'] = all_models['lstm']
            if use_transformer and 'transformer' in all_models:
                st.session_state.models['transformer'] = all_models['transformer']

    # Check if models are loaded
    if st.session_state.models is None or len(st.session_state.models) == 0:
        st.warning("Please load the models first")
    else:
        st.success(f"Models loaded: {', '.join([name.upper() for name in st.session_state.models.keys()])}")
        # Input selection
        st.subheader("Select Input Data")
        
        input_method = st.radio(
            "Choose input method:",
            ["Use sample Normal ECG", "Use sample Ventricular ECG", "Upload ECG data"]
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
        
        elif input_method == "Use sample Normal ECG":
            signal = get_sample_ecg()
            st.info("Using sample Normal ECG data")
        
        else:  # Use sample Ventricular ECG
            signal = get_sample_ventricular_ecg()
            st.info("Using sample Ventricular ECG data")
        
        if signal is not None:
            # Show original signal
            st.subheader("Original ECG Signal")
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(signal)
            ax.set_title('Original ECG Signal')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig)
            
            # Debug preprocessing if enabled
            if debug_mode:
                preprocessed_signal = debug_preprocessing(signal)
            
            # Preprocess and classify
            if st.button("Analyze ECG"):
                with st.spinner("Preprocessing signal..."):
                    # Segment the signal and preprocess each segment
                    segments, segment_indices = segment_signal(signal)
                    
                    st.write(f"Created {len(segments)} segments from signal of length {len(signal)}")
                    
                    # Show first few segments if in debug mode
                    if debug_mode and len(segments) > 0:
                        st.subheader("Sample Segments")
                        fig, axes = plt.subplots(min(3, len(segments)), 1, figsize=(15, 10))
                        if len(segments) == 1:
                            axes = [axes]  # Make it iterable when only one segment
                        
                        for i, ax in enumerate(axes):
                            if i < len(segments):
                                ax.plot(segments[i])
                                ax.set_title(f'Segment {i+1} (Start: {segment_indices[i]})')
                                ax.set_xlabel('Sample')
                                ax.set_ylabel('Amplitude')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
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