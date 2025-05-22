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
# Define the heart icon using a UTF-8 character
HEART_ICON = "❤️"

# Preprocessing functions
def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=5):
    """Apply a bandpass filter to the signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
    except ValueError as e:
        st.error(f"Filtering error: {e}. Data might be too short or constant.")
        return data # Return original data if filtering fails
    return filtered_data

def normalize_segment(segment):
    """Normalize a single segment"""
    return (segment - np.mean(segment)) / (np.std(segment) + 1e-7)

def load_models():
    """Load all trained models"""
    models = {}
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    st.info("Attempting to load models...")
    try:
        for model_name in ['cnn', 'lstm', 'transformer']:
            model_path = f'{model_name}_model.keras'
            if os.path.exists(model_path):
                models[model_name] = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                st.success(f"Successfully loaded {model_name.upper()} model from {model_path}")
            else:
                st.warning(f"Could not find {model_name.upper()} model at '{model_path}'. Please ensure the models are in the same directory as this app or provide the correct path.")
                st.info("You need to train your models using `ari.ipynb` and save them as `.keras` files in the app's directory.")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}. Make sure TensorFlow and Keras are installed correctly and the model files are valid.")
        
    return models

def ensemble_predict(models, X_data):
    """Get ensemble predictions from multiple models"""
    if not models:
        st.error("No models available for prediction! Please load models first.")
        return None
    
    # Collect predictions from each model
    all_preds = []
    for name, model in models.items():
        with st.spinner(f"Getting predictions from {name.upper()} model..."):
            try:
                y_pred = model.predict(X_data, verbose=0) # Set verbose to 0 to suppress progress bar
                all_preds.append(y_pred)
            except Exception as e:
                st.warning(f"Error predicting with {name.upper()} model: {e}. This model's predictions will be skipped.")
    
    if not all_preds:
        st.error("No successful predictions from any model. Check your data and models.")
        return None

    # Average the predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds

def display_prediction_results(predictions, segment_indices=None, original_signal=None):
    """Display prediction results as charts and tables"""
    # Get the predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Count occurrences of each class
    class_counts = np.bincount(pred_classes, minlength=len(CLASS_NAMES))
    
    # Create a dataframe for the class distribution
    total_segments = np.sum(class_counts)
    if total_segments == 0:
        st.warning("No segments were classified. Please ensure the input signal is valid.")
        return

    df_distribution = pd.DataFrame({
        'Arrhythmia Type': CLASS_NAMES,
        'Count': class_counts,
        'Percentage': (class_counts / total_segments * 100).round(2)
    })
    
    # Get the dominant class (mode)
    dominant_class_index = np.argmax(class_counts)
    dominant_class = CLASS_NAMES[dominant_class_index]
    
    # Display summary information
    st.subheader("Classification Results Summary")
    st.markdown(f"**Dominant arrhythmia type detected: <span style='color:green; font-size:20px;'>{dominant_class}</span>**", unsafe_allow_html=True)
    
    # Display distribution as bar chart
    st.subheader("Distribution of Detected Arrhythmia Types")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Arrhythmia Type', y='Percentage', data=df_distribution, ax=ax, palette='viridis')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Distribution of Detected Arrhythmia Types Across Segments')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # Display as a table
    st.subheader("Detailed Classification Results Table")
    st.table(df_distribution)
    
    # If we have segments and original signal, show segment classifications
    if segment_indices is not None and original_signal is not None:
        st.subheader("ECG Signal with Classified Segments Overlaid")
        st.markdown("Colored regions indicate the predicted arrhythmia type for each segment.")
        
        # Plot the signal with colored regions for each segment
        fig, ax = plt.subplots(figsize=(18, 6)) # Larger figure for better visibility
        ax.plot(original_signal, color='gray', alpha=0.7, linewidth=1, label='Original Signal')
        
        colors = plt.cm.get_cmap('tab10', len(CLASS_NAMES)) # Using a clearer colormap
        
        # Plot only unique legend entries
        legend_patches = []
        seen_classes = set()

        for i, (start_idx, pred_class) in enumerate(zip(segment_indices, pred_classes)):
            end_idx = start_idx + SAMPLE_LENGTH
            color_for_class = colors(pred_class)
            ax.axvspan(start_idx, end_idx, alpha=0.3, color=color_for_class)
            
            if CLASS_NAMES[pred_class] not in seen_classes:
                legend_patches.append(plt.Rectangle((0,0),1,1, color=color_for_class, alpha=0.5, label=CLASS_NAMES[pred_class]))
                seen_classes.add(CLASS_NAMES[pred_class])
                
        ax.set_title('ECG Signal with Classified Segments')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend
        ax.legend(handles=legend_patches, title='Arrhythmia Type', loc='upper right', bbox_to_anchor=(1.15, 1))
        
        st.pyplot(fig)

def get_sample_ecg(duration_seconds=10, fs=360):
    """Return a more realistic synthetic ECG signal for demonstration"""
    t = np.linspace(0, duration_seconds, int(duration_seconds * fs), endpoint=False)
    
    # Base signal (simulated normal sinus rhythm)
    signal = np.sin(2 * np.pi * 1 * t) * 0.3 # Baseline rhythm
    
    # Add QRS complexes
    qrs_peak_times = np.arange(0.5, duration_seconds, 0.8) # Approx 75 BPM
    for pk_time in qrs_peak_times:
        # R-peak
        signal += 1.5 * np.exp(-((t - pk_time) ** 2) / 0.005)
        # Q-wave
        signal -= 0.5 * np.exp(-((t - (pk_time - 0.05)) ** 2) / 0.002)
        # S-wave
        signal -= 0.7 * np.exp(-((t - (pk_time + 0.05)) ** 2) / 0.003)
    
    # Add P waves
    p_wave_times = qrs_peak_times - 0.2
    for p_time in p_wave_times:
        signal += 0.3 * np.exp(-((t - p_time) ** 2) / 0.008)
        
    # Add T waves
    t_wave_times = qrs_peak_times + 0.3
    for t_time in t_wave_times:
        signal += 0.4 * np.exp(-((t - t_time) ** 2) / 0.02)
        
    # Add some random noise
    signal += 0.05 * np.random.randn(len(t))
    
    return signal

def parse_csv_data(content):
    """Parse CSV content into a numpy array"""
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        if len(df.columns) == 1:
            signal = df.iloc[:, 0].values
        else:
            st.info("Multiple columns detected in the CSV file. Please select the column containing the ECG signal.")
            selected_column = st.selectbox("Select ECG Column:", df.columns)
            if selected_column:
                signal = df[selected_column].values
            else:
                st.error("No ECG column selected. Please choose one to proceed.")
                return None
            
        return signal
    except Exception as e:
        st.error(f"Error parsing CSV data: {str(e)}. Please ensure it's a valid CSV with numerical data.")
        return None

def parse_wfdb_data(uploaded_file):
    """Parse WFDB format data from uploaded .dat/.hea pair"""
    signal = None
    annotations = None
    
    # Create a temporary directory to save WFDB files
    with tempfile.TemporaryDirectory() as tmpdirname:
        base_filename = os.path.splitext(uploaded_file.name)[0]
        
        # Streamlit f.read() can only be called once, so we save to temp file
        file_content = uploaded_file.read()
        
        # Check if a .dat file was uploaded, if so, look for a matching .hea
        if uploaded_file.name.endswith('.dat'):
            dat_file_path = os.path.join(tmpdirname, uploaded_file.name)
            with open(dat_file_path, 'wb') as f:
                f.write(file_content)
            
            # Try to find the corresponding .hea file in the same upload session or inform user
            st.warning("You uploaded a `.dat` file. Please also upload its corresponding `.hea` file for proper WFDB reading if available.")
            # We cannot programmatically get the .hea file from the previous upload if they were separate.
            # For simplicity, we assume if .dat is uploaded alone, it might not work or we try reading it.
            # wfdb.rdrecord works with just .dat if .hea is in the same dir and has same name.
            # In a real scenario, you'd need the user to upload both or fetch from a known source.
            
            try:
                record = wfdb.rdrecord(os.path.join(tmpdirname, base_filename))
                signal = record.p_signal[:, 0] if record.p_signal is not None else None
            except Exception as e:
                st.error(f"Error reading WFDB .dat file: {e}. Ensure the matching .hea file is present or the .dat file is self-contained.")
                signal = None

        elif uploaded_file.name.endswith('.hea'):
            hea_file_path = os.path.join(tmpdirname, uploaded_file.name)
            with open(hea_file_path, 'wb') as f:
                f.write(file_content)

            # Inform user to also upload .dat
            st.warning("You uploaded a `.hea` file. Please also upload its corresponding `.dat` file for the signal data.")
            # We need to assume the .dat file would be uploaded next or is already present.
            # To make this robust, a common practice is to have a dedicated section for "upload .dat and .hea together".
            
            # Try to read the record assuming .dat is present in tmpdirname from another upload if any
            # (which is not how Streamlit file_uploader works directly for multiple files in one go unless in list mode)
            # For demonstration, we'll assume the base_filename is sufficient for wfdb.rdrecord
            try:
                record = wfdb.rdrecord(os.path.join(tmpdirname, base_filename))
                signal = record.p_signal[:, 0] if record.p_signal is not None else None
                annotations = wfdb.rdann(os.path.join(tmpdirname, base_filename), 'atr')
            except Exception as e:
                st.error(f"Error reading WFDB .hea file: {e}. Ensure the corresponding .dat file is uploaded.")
                signal = None
                
    return signal, annotations


# Main app
def main():
    st.set_page_config(
        page_title="ECG Arrhythmia Classifier",
        page_icon=HEART_ICON,
        layout="wide"
    )
    
    st.title(f"{HEART_ICON} ECG Arrhythmia Classification App")
    st.markdown("""
    This application utilizes an ensemble of deep learning models (Convolutional Neural Network - CNN, 
    Long Short-Term Memory - LSTM, and Transformer) to analyze Electrocardiogram (ECG) signals and 
    classify heartbeats into five common arrhythmia categories based on the AAMI (Association for the Advancement of Medical Instrumentation) standard:
    - **Normal (N)**: Normal sinus rhythm.
    - **Supraventricular (S)**: Atrial premature beat, aberrant atrial premature beat, nodal premature beat, supraventricular premature beat.
    - **Ventricular (V)**: Premature ventricular contraction, ventricular escape beat.
    - **Fusion (F)**: Fusion of ventricular and normal beat.
    - **Unknown (Q)**: Unclassifiable beat, paced beat.
    
    Upload your ECG data (CSV, TXT, or WFDB formats) or use our sample data to get started!
    """)
    
    # Initialize session state for models
    if 'models' not in st.session_state:
        st.session_state.models = None
    
    st.sidebar.header("Model Loading")
    load_button = st.sidebar.button("Load Models")
    
    # Only load models if button is clicked or if they are already loaded
    if load_button or (st.session_state.models is not None and len(st.session_state.models) > 0):
        if st.session_state.models is None or len(st.session_state.models) == 0:
            st.session_state.models = load_models()
        
        if st.session_state.models:
            st.sidebar.success(f"Models loaded: {', '.join([name.upper() for name in st.session_state.models.keys()])}")
        else:
            st.sidebar.error("Failed to load any models. Classification will not proceed.")
    else:
        st.sidebar.info("Click 'Load Models' to begin classification.")
        
    # Proceed only if models are loaded
    if st.session_state.models and len(st.session_state.models) > 0:
        st.subheader("Input ECG Data")
        
        input_method = st.radio(
            "How would you like to provide the ECG data?",
            ["Upload ECG data", "Use sample data"],
            key="input_method_radio"
        )
        
        signal = None
        
        if input_method == "Upload ECG data":
            st.markdown("Upload your ECG signal file. Supported formats: `.csv`, `.txt` (single column), or WFDB format (`.dat` and `.hea` files).")
            
            uploaded_files = st.file_uploader("Choose one or more files", type=["csv", "txt", "dat", "hea"], accept_multiple_files=True)
            
            if uploaded_files:
                # Prioritize .hea for WFDB, otherwise handle single file
                wfdb_hea_file = None
                wfdb_dat_file = None
                other_file = None

                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith('.hea'):
                        wfdb_hea_file = uploaded_file
                    elif uploaded_file.name.endswith('.dat'):
                        wfdb_dat_file = uploaded_file
                    else:
                        other_file = uploaded_file # For CSV/TXT

                if wfdb_hea_file and wfdb_dat_file:
                    st.info(f"Processing WFDB record: {os.path.splitext(wfdb_hea_file.name)[0]}")
                    # Save both to a temporary directory for wfdb.rdrecord to find them
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        hea_path = os.path.join(tmpdirname, wfdb_hea_file.name)
                        dat_path = os.path.join(tmpdirname, wfdb_dat_file.name)
                        
                        with open(hea_path, 'wb') as f: f.write(wfdb_hea_file.read())
                        with open(dat_path, 'wb') as f: f.write(wfdb_dat_file.read())
                        
                        try:
                            record_name = os.path.splitext(wfdb_hea_file.name)[0]
                            record = wfdb.rdrecord(os.path.join(tmpdirname, record_name))
                            signal = record.p_signal[:, 0] if record.p_signal is not None else None
                            if signal is None:
                                st.error("Could not extract signal from WFDB files. Ensure the .dat file contains signal data.")
                        except Exception as e:
                            st.error(f"Error reading WFDB files: {e}. Ensure both `.dat` and `.hea` files for the same record are uploaded.")
                elif other_file:
                    if other_file.name.endswith(('.csv', '.txt')):
                        st.info(f"Processing CSV/TXT file: {other_file.name}")
                        content = other_file.read()
                        signal = parse_csv_data(content)
                elif uploaded_files: # Case where only one .dat or .hea is uploaded
                     st.warning("For WFDB files, please upload both the `.dat` and `.hea` files for the same record. Trying to read the single file if possible.")
                     # Attempt to read a single WFDB file, though it often requires the pair
                     if uploaded_files[0].name.endswith('.dat') or uploaded_files[0].name.endswith('.hea'):
                         with tempfile.TemporaryDirectory() as tmpdirname:
                             single_file_path = os.path.join(tmpdirname, uploaded_files[0].name)
                             with open(single_file_path, 'wb') as f: f.write(uploaded_files[0].read())
                             
                             base_name = os.path.splitext(uploaded_files[0].name)[0]
                             try:
                                 record = wfdb.rdrecord(os.path.join(tmpdirname, base_name))
                                 signal = record.p_signal[:, 0] if record.p_signal is not None else None
                                 if signal is None:
                                     st.error("Signal could not be extracted from the single WFDB file. A matching .dat/.hea file is likely missing.")
                             except Exception as e:
                                 st.error(f"Could not read single WFDB file ({uploaded_files[0].name}): {e}. Please ensure you upload both `.dat` and `.hea` for WFDB records.")
                                 signal = None
                else:
                    st.warning("No valid ECG file uploaded or selected for processing.")
        
        else:  # Use sample data
            st.info("Using a synthetic sample ECG signal for demonstration.")
            signal = get_sample_ecg()
        
        if signal is not None and len(signal) > 0:
            # Show original signal plot
            st.subheader("Original ECG Signal Preview")
            fig_orig, ax_orig = plt.subplots(figsize=(15, 5))
            ax_orig.plot(signal, color='mediumblue', linewidth=1)
            ax_orig.set_title('Original ECG Signal')
            ax_orig.set_xlabel('Sample Index')
            ax_orig.set_ylabel('Amplitude')
            ax_orig.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_orig)
            
            st.markdown("---") # Separator
            
            # Preprocess and classify button
            if st.button("Analyze ECG Signal", help="Click to preprocess the ECG and classify heartbeats."):
                with st.spinner("Applying filter and segmenting signal..."):
                    # **FIXED:** Apply bandpass filter to the entire signal first, consistent with training
                    filtered_signal = butter_bandpass_filter(signal)
                    
                    # Store indices for plotting later
                    segment_start_indices = []
                    segments_to_classify = []
                    
                    # Perform R-peak detection and segmentation (simplified for general signal here)
                    # For precise R-peak detection, a dedicated algorithm (e.g., Pan-Tompkins) would be needed.
                    # Here, we do overlapping fixed-length windows as a general segmentation strategy.
                    step_size = SAMPLE_LENGTH // 2 # Overlap by half for better coverage
                    
                    if len(filtered_signal) < SAMPLE_LENGTH:
                        # If signal is too short, just process the whole signal (after filtering)
                        processed_segment = normalize_segment(filtered_signal)
                        segments_to_classify.append(processed_segment)
                        segment_start_indices.append(0)
                        st.warning(f"Signal is shorter than {SAMPLE_LENGTH} samples. Processing as a single segment.")
                    else:
                        for i in range(0, len(filtered_signal) - SAMPLE_LENGTH + 1, step_size):
                            segment = filtered_signal[i:i+SAMPLE_LENGTH]
                            processed_segment = normalize_segment(segment)
                            segments_to_classify.append(processed_segment)
                            segment_start_indices.append(i)
                            
                    segments_to_classify = np.array(segments_to_classify)
                    
                    if len(segments_to_classify) == 0:
                        st.error("Could not create any valid segments from the signal. It might be too short or invalid.")
                        return

                with st.spinner("Classifying ECG segments using ensemble models..."):
                    # Make predictions
                    predictions = ensemble_predict(st.session_state.models, segments_to_classify)
                    
                    if predictions is not None:
                        display_prediction_results(predictions, segment_start_indices, filtered_signal) # Pass filtered signal for plotting
                    else:
                        st.error("Failed to get predictions. Please check the console for errors or try another file.")
        elif signal is None and input_method == "Upload ECG data" and uploaded_files:
            st.warning("No valid ECG signal could be extracted from the uploaded file(s). Please check the file format and content.")
        else:
            st.info("Upload an ECG file or select sample data to begin analysis.")
    else:
        st.info("Please load the models using the 'Load Models' button in the sidebar to enable ECG analysis.")

if __name__ == "__main__":
    main()
