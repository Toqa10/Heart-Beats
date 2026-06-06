import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Class labels with descriptions
class_labels = {
    0: {"name": "Normal (N)", "color": "#2Ecc71", "desc": "Normal heartbeat pattern", "advice": "No signs of abnormality. Maintain healthy lifestyle."},
    1: {"name": "Supraventricular (S)", "color": "#F39C12", "desc": "Supraventricular premature beat", "advice": "Consult cardiologist. May require monitoring or treatment."},
    2: {"name": "Ventricular (V)", "color": "#E74C3C", "desc": "Ventricular premature beat", "advice": "Urgent cardiology consultation needed. Close monitoring required."},
    3: {"name": "Fusion (F)", "color": "#9B59B6", "desc": "Fusion beat", "advice": "Medical review needed to determine cause and treatment."},
    4: {"name": "Unknown (Q)", "color": "#95A5A6", "desc": "Unclassified beat pattern", "advice": "Re-examination or specialist consultation recommended."}
}

# Page configuration
st.set_page_config(
    page_title="ECG Heartbeat Diagnosis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Gradient Title */
        .gradient-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
            animation: fadeInDown 0.8s ease-out;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* Card style */
        .card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 1.5rem;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Prediction card */
        .prediction-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            border-left: 5px solid;
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .prediction-card:hover {
            transform: translateY(-5px);
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Button styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        /* File uploader */
        .stFileUploader > div {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1rem;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #888;
            font-size: 0.9rem;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-size: 1.1rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model info
    st.markdown("### 📊 Model Information")
    st.info("""
    - **Type**: CNN (Convolutional Neural Network)
    - **Input**: 187 ECG values
    - **Accuracy**: >95%
    """)
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.markdown("""
    This app uses AI to classify heartbeats 
    from ECG signals into 5 different categories.
    """)
    
    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    for idx, label in class_labels.items():
        st.markdown(f"- **{label['name']}**: {label['desc']}")

# Title
st.markdown('<div class="gradient-title">💓 Heartbeat Diagnosis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent ECG Heartbeat Classification using Deep Learning</div>', unsafe_allow_html=True)

# Two columns for main content
col1, col2 = st.columns([2, 1])

with col1:
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["📁 Upload CSV", "✍️ Manual Input", "🎲 Generate Test Data"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="File must contain exactly 187 numeric values")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                
                # Auto-detect if data is in row or column
                if df.shape[0] == 187 and df.shape[1] == 1:
                    values = df.iloc[:, 0].values
                elif df.shape[0] == 1 and df.shape[1] == 187:
                    values = df.iloc[0, :].values
                else:
                    values = df.values.flatten()
                
                if len(values) != 187:
                    st.error(f"⚠️ File contains {len(values)} values, but exactly 187 values are required")
                else:
                    st.success(f"✅ Successfully loaded {len(values)} values")
                    
                    # Preview data
                    with st.expander("📊 Data Preview"):
                        col1a, col1b, col1c = st.columns(3)
                        with col1a:
                            st.metric("First 10 values", ", ".join([f"{x:.3f}" for x in values[:10]]))
                        with col1b:
                            st.metric("Range", f"{values.min():.3f} - {values.max():.3f}")
                        with col1c:
                            st.metric("Mean", f"{values.mean():.3f}")
                    
                    if st.button("🔍 Analyze ECG", use_container_width=True):
                        with st.spinner("Analyzing heartbeats..."):
                            reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                            prediction = model.predict(reshaped)
                            class_index = int(np.argmax(prediction))
                            confidence = float(np.max(prediction)) * 100
                            
                            # Store in session state
                            st.session_state['prediction'] = class_index
                            st.session_state['confidence'] = confidence
                            st.session_state['values'] = values
                            
                            st.success("Analysis complete!")
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab2:
        st.markdown("Enter 187 comma or space-separated values:")
        user_input = st.text_area("Values:", height=150, 
                                 placeholder="Example: 0.5, 0.7, 0.3, ...")
        
        if st.button("🔍 Analyze Input", use_container_width=True):
            try:
                cleaned = user_input.strip().replace("\t", ",").replace("\n", ",").replace(" ", ",")
                values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
                
                if len(values) != 187:
                    st.error(f"⚠️ You entered {len(values)} values, but exactly 187 values are required")
                else:
                    with st.spinner("Analyzing..."):
                        reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                        prediction = model.predict(reshaped)
                        class_index = int(np.argmax(prediction))
                        confidence = float(np.max(prediction)) * 100
                        
                        st.session_state['prediction'] = class_index
                        st.session_state['confidence'] = confidence
                        st.session_state['values'] = values
                        
                        st.success("Analysis complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with tab3:
        st.markdown("### Generate Synthetic ECG Data")
        
        noise_level = st.slider("Noise Level:", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🎲 Generate Random Beat", use_container_width=True):
            # Generate synthetic ECG-like signal
            t = np.linspace(0, 4*np.pi, 187)
            synthetic = np.sin(t) + 0.3*np.sin(3*t) + np.random.normal(0, noise_level, 187)
            synthetic = synthetic / np.max(np.abs(synthetic))
            
            st.session_state['values'] = synthetic
            st.success("Synthetic data generated!")
            
            with st.expander("📊 Generated Data Preview"):
                col1a, col1b = st.columns(2)
                with col1a:
                    st.metric("Range", f"{synthetic.min():.3f} - {synthetic.max():.3f}")
                with col1b:
                    st.metric("Mean", f"{synthetic.mean():.3f}")
            
            if st.button("🔍 Analyze Generated Data", use_container_width=True):
                with st.spinner("Analyzing..."):
                    reshaped = np.array(synthetic, dtype=np.float32).reshape(1, 187, 1)
                    prediction = model.predict(reshaped)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction)) * 100
                    
                    st.session_state['prediction'] = class_index
                    st.session_state['confidence'] = confidence
                    
                    st.success("Analysis complete!")

with col2:
    # Display results if available
    if 'prediction' in st.session_state:
        pred_idx = st.session_state['prediction']
        confidence = st.session_state['confidence']
        
        # Prediction Card
        st.markdown(f"""
        <div class="prediction-card" style="border-left-color: {class_labels[pred_idx]['color']}">
            <h3 style="margin:0; color:{class_labels[pred_idx]['color']}">Diagnosis:</h3>
            <h2 style="margin:10px 0;">{class_labels[pred_idx]['name']}</h2>
            <p style="color:#666;">{class_labels[pred_idx]['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown("### 📊 Confidence Level")
        st.progress(confidence/100)
        st.markdown(f"<center><b>{confidence:.1f}%</b></center>", unsafe_allow_html=True)
        
        # Color-coded confidence
        if confidence >= 80:
            st.success("✅ Very High Confidence")
        elif confidence >= 60:
            st.warning("⚠️ Moderate Confidence")
        else:
            st.error("❌ Low Confidence - Re-examination recommended")
        
        # Medical advice
        st.markdown("### 💊 Medical Advice")
        st.info(class_labels[pred_idx]['advice'])
        
        # Display ECG values summary
        if 'values' in st.session_state:
            st.markdown("### 📈 Signal Statistics")
            vals = st.session_state['values']
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Min Value", f"{vals.min():.3f}")
                st.metric("Std Dev", f"{vals.std():.3f}")
            with col_stats2:
                st.metric("Max Value", f"{vals.max():.3f}")
                st.metric("Median", f"{np.median(vals):.3f}")
        
        # Download results
        if st.button("📥 Download Diagnosis Report", use_container_width=True):
            report_df = pd.DataFrame({
                'Feature': ['Diagnosis', 'Confidence', 'Medical Advice', 'Description'],
                'Value': [class_labels[pred_idx]['name'], f"{confidence:.2f}%", class_labels[pred_idx]['advice'], class_labels[pred_idx]['desc']]
            })
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="💾 Save as CSV",
                data=csv,
                file_name="ecg_diagnosis_report.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("👈 Upload a file or enter data to start analysis")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🤖 Powered by Deep Learning CNN | 📊 Model Accuracy >95% | 💡 For reference use only</p>
    <p>© 2024 Heartbeat Diagnosis AI - Built with TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Additional information
with st.expander("ℹ️ How it works"):
    st.markdown("""
    ### How does the app work?
    1. The model analyzes 187 time points from the ECG signal
    2. Uses a CNN (Convolutional Neural Network) to recognize patterns
    3. Classifies the beat into one of 5 categories:
       - **N**: Normal beat
       - **S**: Supraventricular premature beat
       - **V**: Ventricular premature beat
       - **F**: Fusion beat
       - **Q**: Unclassified beat
    
    ### Usage tips:
    - Ensure your CSV file contains exactly 187 numeric values
    - Values should be normalized between -1 and 1
    - The model is designed for assistive diagnostic purposes only
    """)
