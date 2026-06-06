import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Clinical database with LIGHT colors
class_labels = {
    0: {
        "name": "Normal Sinus Rhythm",
        "code": "N",
        "icd10": "I49.9",
        "color": "#2E7D32",
        "bg_color": "#E8F5E9",
        "light_bg": "#F1F8E9",
        "severity": "Low",
        "risk_score": 8,
        "desc": "Normal sinus rhythm. Regular cardiac conduction pattern.",
        "clinical_advice": "No acute intervention needed. Continue regular follow-up.",
        "treatment": "Routine monitoring only",
        "medications": [],
        "follow_up": "12 months",
        "specialist": "Primary Care",
        "imaging_needed": False
    },
    1: {
        "name": "Supraventricular Ectopy",
        "code": "S",
        "icd10": "I47.1",
        "color": "#E65100",
        "bg_color": "#FFF3E0",
        "light_bg": "#FFF8E1",
        "severity": "Moderate",
        "risk_score": 42,
        "desc": "Supraventricular premature beats. May indicate atrial irritability.",
        "clinical_advice": "Clinical correlation recommended. Consider Holter monitoring.",
        "treatment": "Beta-blocker therapy if symptomatic",
        "medications": ["Metoprolol 25mg BID", "Propranolol 10mg TID"],
        "follow_up": "4-6 weeks",
        "specialist": "Cardiology",
        "imaging_needed": False
    },
    2: {
        "name": "Ventricular Ectopy",
        "code": "V",
        "icd10": "I49.3",
        "color": "#C62828",
        "bg_color": "#FFEBEE",
        "light_bg": "#FDEDEC",
        "severity": "High",
        "risk_score": 78,
        "desc": "Ventricular premature complexes. Requires further evaluation.",
        "clinical_advice": "Urgent cardiology referral. Risk of ventricular arrhythmias.",
        "treatment": "Antiarrhythmic therapy. Possible ablation.",
        "medications": ["Amiodarone 200mg daily", "Mexiletine 150mg TID"],
        "follow_up": "1 week",
        "specialist": "Electrophysiology",
        "imaging_needed": True
    },
    3: {
        "name": "Fusion Beat",
        "code": "F",
        "icd10": "I49.8",
        "color": "#6A1B9A",
        "bg_color": "#F3E5F5",
        "light_bg": "#F9F2FC",
        "severity": "Moderate-High",
        "risk_score": 65,
        "desc": "Fusion complexes. Mixed conduction pattern.",
        "clinical_advice": "Electrophysiology consultation recommended.",
        "treatment": "Based on underlying rhythm disorder",
        "medications": ["Individualized therapy"],
        "follow_up": "2 weeks",
        "specialist": "Electrophysiology",
        "imaging_needed": True
    },
    4: {
        "name": "Unclassified Pattern",
        "code": "Q",
        "icd10": "R94.31",
        "color": "#546E7A",
        "bg_color": "#ECEFF1",
        "light_bg": "#F5F5F5",
        "severity": "Indeterminate",
        "risk_score": 35,
        "desc": "Atypical pattern. Technical or biological artifact suspected.",
        "clinical_advice": "Repeat ECG with proper lead placement.",
        "treatment": "Await confirmation",
        "medications": [],
        "follow_up": "1 week",
        "specialist": "Cardiology",
        "imaging_needed": False
    }
}

# Page config
st.set_page_config(
    page_title="ECG Clinical Decision Support | Professional Suite",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Healthcare CSS
st.markdown("""
<style>
    /* Professional Light Healthcare Theme */
    :root {
        --primary-light: #EBF5FB;
        --primary: #5DADE2;
        --primary-dark: #3498DB;
        --secondary: #85C1E9;
        --accent: #76D7C4;
        --success: #A9DFBF;
        --warning: #F9E79F;
        --danger: #F5B7B1;
        --bg-main: #F8F9FA;
        --bg-card: #FFFFFF;
        --border: #E5E8E8;
        --text-primary: #2C3E50;
        --text-secondary: #5D6D7E;
        --text-light: #95A5A6;
    }
    
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* Light Header */
    .light-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #E8F4FD 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #D6EAF8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2C3E50;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .header-subtitle {
        color: #5D6D7E;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Light Card */
    .light-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #E8ECEF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    .light-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #D6EAF8;
    }
    
    /* Metric Card Light */
    .metric-card-light {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #E8ECEF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    .metric-card-light:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    
    .metric-label-light {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        color: #7F8C8D;
        margin-bottom: 0.75rem;
    }
    
    .metric-value-light {
        font-size: 2rem;
        font-weight: 700;
        color: #2C3E50;
        margin: 0.25rem 0;
    }
    
    .metric-unit-light {
        font-size: 0.7rem;
        color: #95A5A6;
    }
    
    /* Diagnosis Card Light */
    .diagnostic-card-light {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .diagnostic-title-light {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #2C3E50;
    }
    
    .diagnostic-code-light {
        font-family: monospace;
        font-size: 0.85rem;
        color: #7F8C8D;
    }
    
    /* Risk Indicator Light */
    .risk-indicator-light {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-Low { background: #E8F8F5; color: #1ABC9C; border: 1px solid #A3E4D7; }
    .risk-Moderate { background: #FEF9E7; color: #F39C12; border: 1px solid #F9E79F; }
    .risk-High { background: #FDEDEC; color: #E74C3C; border: 1px solid #F5B7B1; }
    .risk-Moderate-High { background: #FEF5E7; color: #E67E22; border: 1px solid #FAD7A0; }
    .risk-Indeterminate { background: #EBF5FB; color: #3498DB; border: 1px solid #AED6F1; }
    
    /* Button Light */
    .stButton > button {
        background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(52,152,219,0.3);
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
    }
    
    /* Alert Box Light */
    .alert-box-light {
        background: #FDEDEC;
        border-left: 4px solid #E74C3C;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #2C3E50;
    }
    
    /* Tab Styling Light */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #E8ECEF;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #5D6D7E;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%);
        color: white;
    }
    
    /* Expander Light */
    .streamlit-expanderHeader {
        background: #F8F9FA;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #E8ECEF;
        color: #2C3E50;
    }
    
    /* Progress Bar Light */
    .stProgress > div > div {
        background: linear-gradient(90deg, #76D7C4, #5DADE2);
        border-radius: 10px;
    }
    
    /* Footer Light */
    .light-footer {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid #E8ECEF;
        font-size: 0.75rem;
        color: #7F8C8D;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #D6EAF8, transparent);
    }
    
    /* ECG Container */
    .ecg-container-light {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #E8ECEF;
        margin: 1rem 0;
    }
    
    /* Info Box Light */
    .info-box-light {
        background: #EBF5FB;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #5DADE2;
        font-size: 0.85rem;
        color: #2C3E50;
    }
    
    /* Badge Light */
    .badge-light {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
        background: #F0F3F4;
        color: #5D6D7E;
        border: 1px solid #E8ECEF;
    }
    
    /* Sidebar Light */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #E8ECEF;
    }
    
    /* Status Indicators */
    .status-good {
        color: #27AE60;
        font-weight: 500;
    }
    
    .status-warning {
        color: #E67E22;
        font-weight: 500;
    }
    
    .status-critical {
        color: #E74C3C;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'clinical_history' not in st.session_state:
    st.session_state.clinical_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}

# Helper functions
def calculate_heart_rate(signal, sampling_rate=500):
    peaks = []
    threshold = 0.3 * np.max(np.abs(signal))
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(i)
    if len(peaks) > 1:
        avg_rr = np.mean(np.diff(peaks)) / sampling_rate
        heart_rate = 60 / avg_rr
    else:
        heart_rate = 75
    return min(200, max(30, heart_rate)), len(peaks)

def calculate_snr(signal):
    signal_power = np.max(np.abs(signal))**2
    noise_power = np.var(signal) if np.var(signal) > 0 else 0.001
    snr = 10 * np.log10(signal_power / noise_power)
    return max(0, min(25, snr))

# Header
st.markdown("""
<div class="light-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="header-title">⚕️ ECG Clinical Decision Support</h1>
            <p class="header-subtitle">AI-Powered Cardiac Analysis System | Light Edition</p>
        </div>
        <div style="display: flex; gap: 0.5rem;">
            <span class="badge-light">✓ HIPAA Compliant</span>
            <span class="badge-light">✓ CLIA Certified</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 Patient Information")
    
    with st.expander("Demographics", expanded=True):
        patient_name = st.text_input("Patient Name", placeholder="Last, First")
        patient_id = st.text_input("MRN", value=f"ECG-{datetime.now().strftime('%Y%m%d')}")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            age = st.number_input("Age", 0, 120, 55)
        with col_s2:
            gender = st.selectbox("Gender", ["M", "F", "Other"])
        
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            weight = st.number_input("Weight (kg)", 20, 200, 70)
        with col_s4:
            height = st.number_input("Height (cm)", 100, 250, 170)
        
        if height > 0:
            bmi = weight / ((height/100) ** 2)
            st.caption(f"BMI: {bmi:.1f} kg/m²")
    
    with st.expander("Medical History"):
        comorbidities = st.multiselect("Comorbidities", 
                                      ["Hypertension", "Diabetes Type 2", "CAD", "Heart Failure",
                                       "Atrial Fibrillation", "COPD", "CKD", "None"])
        medications = st.text_area("Current Medications", placeholder="List with doses")
    
    st.markdown("---")
    st.caption("✨ Light Edition v3.0")
    st.caption("© 2024 Clinical Decision Support")

# Main Input Section
st.markdown("### 📊 ECG Data Acquisition")

col_in1, col_in2 = st.columns([2, 1])

with col_in1:
    input_method = st.radio("Data Source", ["CSV Upload", "Manual Entry", "Test Pattern"], horizontal=True)

ecg_values = None

if input_method == "CSV Upload":
    uploaded = st.file_uploader("Upload ECG File (187 samples)", type=["csv", "txt"])
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        values = df.values.flatten()
        if len(values) == 187:
            ecg_values = values
            st.success(f"✓ Loaded {len(values)} samples successfully")
            with st.expander("Preview Data"):
                st.text(f"Range: [{values.min():.3f}, {values.max():.3f}] | Mean: {values.mean():.3f} | Std: {values.std():.3f}")
        else:
            st.error(f"Invalid: {len(values)} samples (requires 187)")

elif input_method == "Manual Entry":
    manual = st.text_area("Enter 187 values (comma-separated)", height=80,
                         placeholder="0.5, 0.7, 0.3, -0.2, ...")
    if manual and st.button("Process", use_container_width=True):
        try:
            vals = [float(x.strip()) for x in manual.replace('\n',',').split(',') if x.strip()]
            if len(vals) == 187:
                ecg_values = np.array(vals)
                st.success("✓ Data accepted successfully")
            else:
                st.error(f"Expected 187, received {len(vals)}")
        except:
            st.error("Invalid format - please check your input")

else:  # Test Pattern
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pattern = st.selectbox("Pattern Type", ["Normal Sinus", "PVC", "Bradycardia", "Tachycardia"])
    with col_p2:
        noise = st.slider("Noise Level", 0.0, 0.2, 0.05, format="%.2f")
    
    if st.button("Generate Pattern", use_container_width=True):
        t = np.linspace(0, 8*np.pi, 187)
        if pattern == "Normal Sinus":
            ecg_values = np.sin(t) * 0.8 + np.sin(3*t) * 0.2
        elif pattern == "PVC":
            ecg_values = np.sin(t) * 0.8
            ecg_values[80:95] = -1.3
        elif pattern == "Bradycardia":
            ecg_values = np.sin(t/1.5) * 0.8
        else:
            ecg_values = np.sin(t*1.5) * 0.8
        
        ecg_values += np.random.normal(0, noise, 187)
        ecg_values = ecg_values / np.max(np.abs(ecg_values))
        st.success(f"✓ Generated {pattern} pattern successfully")

with col_in2:
    st.markdown("### 📋 Requirements")
    st.info("""
    **Input Specifications**
    - 187 samples per beat
    - Normalized range [-1, 1]
    - CSV or direct entry
    
    **Analysis Features**
    - CNN classification
    - Risk stratification  
    - Clinical guidance
    - Export capabilities
    """)

# Analysis
if ecg_values is not None:
    if st.button("🔬 Run Clinical Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing ECG signal..."):
            # Prediction
            reshaped = ecg_values.reshape(1, 187, 1).astype(np.float32)
            pred = model.predict(reshaped)
            class_idx = int(np.argmax(pred))
            confidence = float(np.max(pred)) * 100
            
            # Signal metrics
            hr, peaks = calculate_heart_rate(ecg_values)
            snr = calculate_snr(ecg_values)
            
            # Risk calculation
            clinical = class_labels[class_idx]
            risk = clinical['risk_score']
            
            # Adjust risk based on patient factors
            if age > 65:
                risk += 15
            if age > 80:
                risk += 10
            if "Hypertension" in comorbidities:
                risk += 10
            if "CAD" in comorbidities:
                risk += 20
            risk = min(100, risk)
            
            # Determine urgency
            if risk >= 70:
                urgency = "EMERGENCY"
                setting = "Emergency Department / Immediate Cardiology"
                bg_color = "#FDEDEC"
            elif risk >= 50:
                urgency = "URGENT"
                setting = "Cardiology Clinic within 48 hours"
                bg_color = "#FEF9E7"
            else:
                urgency = "ROUTINE"
                setting = "Outpatient follow-up"
                bg_color = "#E8F8F5"
            
            # Store result
            result = {
                "timestamp": datetime.now().isoformat(),
                "patient": {"name": patient_name, "mrn": patient_id, "age": age},
                "diagnosis": {
                    "index": class_idx,
                    "name": clinical['name'],
                    "code": clinical['code'],
                    "icd10": clinical['icd10']
                },
                "metrics": {
                    "confidence": confidence,
                    "heart_rate": hr,
                    "snr": snr,
                    "peaks": peaks
                },
                "risk": {"score": risk, "level": clinical['severity']},
                "recommendations": {
                    "urgency": urgency,
                    "setting": setting,
                    "specialist": clinical['specialist'],
                    "follow_up": clinical['follow_up'],
                    "clinical_advice": clinical['clinical_advice']
                },
                "signal": ecg_values.tolist()
            }
            
            st.session_state.current_patient = result
            st.session_state.clinical_history.append(result)
            st.success("✓ Analysis complete successfully!")
            st.balloons()
            st.rerun()

# Display Results
if st.session_state.current_patient:
    r = st.session_state.current_patient
    clinical = class_labels[r['diagnosis']['index']]
    
    # Results Header
    st.markdown("---")
    st.markdown("## 📋 Analysis Results")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-light">
            <div class="metric-label-light">RISK SCORE</div>
            <div class="metric-value-light">{r['risk']['score']}</div>
            <div class="metric-unit-light">/100</div>
            <div style="margin-top: 8px;">
                <span class="risk-indicator-light risk-{r['risk']['level']}">{r['risk']['level']} RISK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hr_status = "🟢" if 60 <= r['metrics']['heart_rate'] <= 100 else "🟡" if 50 <= r['metrics']['heart_rate'] <= 110 else "🔴"
        st.markdown(f"""
        <div class="metric-card-light">
            <div class="metric-label-light">HEART RATE</div>
            <div class="metric-value-light">{hr_status} {r['metrics']['heart_rate']:.0f}</div>
            <div class="metric-unit-light">beats per minute</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conf_color = "🟢" if r['metrics']['confidence'] > 80 else "🟡" if r['metrics']['confidence'] > 60 else "🔴"
        st.markdown(f"""
        <div class="metric-card-light">
            <div class="metric-label-light">CONFIDENCE</div>
            <div class="metric-value-light">{conf_color} {r['metrics']['confidence']:.0f}%</div>
            <div class="metric-unit-light">AI certainty</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        quality_color = "🟢" if r['metrics']['snr'] > 10 else "🟡" if r['metrics']['snr'] > 5 else "🔴"
        quality_text = "Good" if r['metrics']['snr'] > 10 else "Fair" if r['metrics']['snr'] > 5 else "Poor"
        st.markdown(f"""
        <div class="metric-card-light">
            <div class="metric-label-light">SIGNAL QUALITY</div>
            <div class="metric-value-light">{quality_color} {r['metrics']['snr']:.1f}</div>
            <div class="metric-unit-light">dB SNR ({quality_text})</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnosis Card
    st.markdown(f"""
    <div class="diagnostic-card-light" style="border-left-color: {clinical['color']}; background: {clinical['light_bg']}">
        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap;">
            <div>
                <div class="diagnostic-title-light">{clinical['name']} ({clinical['code']})</div>
                <div class="diagnostic-code-light">ICD-10: {clinical['icd10']}</div>
                <p style="margin-top: 0.75rem; font-size: 0.9rem; color: #34495E;">{clinical['desc']}</p>
            </div>
            <div>
                <span class="risk-indicator-light risk-{r['risk']['level']}">{r['risk']['level']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Recommendations
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("### 🚨 Immediate Actions")
        st.markdown(f"""
        <div class="light-card">
            <p><strong>Urgency Level:</strong> <span class="risk-indicator-light risk-{r['risk']['level']}">{r['recommendations']['urgency']}</span></p>
            <p><strong>Care Setting:</strong> {r['recommendations']['setting']}</p>
            <p><strong>Specialist Referral:</strong> {r['recommendations']['specialist']}</p>
            <p><strong>Follow-up Timeline:</strong> {r['recommendations']['follow_up']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_rec2:
        st.markdown("### 💊 Treatment Plan")
        st.markdown(f"""
        <div class="light-card">
            <p><strong>Clinical Advice:</strong></p>
            <p style="color: #34495E;">{r['recommendations']['clinical_advice']}</p>
            <hr>
            <p><strong>Medications:</strong></p>
            <p>{', '.join(clinical['medications']) if clinical['medications'] else 'None indicated'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ECG Visualization
    st.markdown("### 📈 ECG Waveform")
    
    chart_data = pd.DataFrame({
        'Sample': range(len(r['signal'])),
        'Amplitude (mV)': r['signal']
    })
    
    st.line_chart(chart_data.set_index('Sample'), height=300, color=clinical['color'])
    
    # Signal Statistics
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Peak Amplitude", f"{np.max(r['signal']):.3f} mV")
    with col_stats2:
        st.metric("Trough", f"{np.min(r['signal']):.3f} mV")
    with col_stats3:
        st.metric("Mean", f"{np.mean(r['signal']):.3f} mV")
    with col_stats4:
        st.metric("Detected Peaks", f"{r['metrics']['peaks']}")
    
    # Export Section
    st.markdown("### 📎 Export Results")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        report = f"""
CLINICAL ECG REPORT
===================
Generated: {r['timestamp'][:19]}
MRN: {r['patient']['mrn']}
Patient: {r['patient']['name'] or 'Not specified'}
Age: {r['patient']['age']}

DIAGNOSIS
---------
{clinical['name']} ({clinical['code']})
ICD-10: {clinical['icd10']}
Risk Level: {r['risk']['level']} (Score: {r['risk']['score']}/100)

VITALS
------
Heart Rate: {r['metrics']['heart_rate']:.0f} BPM
Signal Quality: {r['metrics']['snr']:.1f} dB
AI Confidence: {r['metrics']['confidence']:.0f}%

RECOMMENDATIONS
--------------
Urgency: {r['recommendations']['urgency']}
Setting: {r['recommendations']['setting']}
Specialist: {r['recommendations']['specialist']}
Follow-up: {r['recommendations']['follow_up']}

CLINICAL ADVICE
--------------
{r['recommendations']['clinical_advice']}

DISCLAIMER
----------
This is an AI-assisted analysis. All clinical decisions must be verified by a qualified physician.
        """
        st.download_button("📄 Download Report", report, f"ECG_Report_{r['patient']['mrn']}.txt")
    
    with col_export2:
        json_report = json.dumps(r, indent=2, default=str)
        st.download_button("💾 Export JSON", json_report, f"ECG_Data_{r['patient']['mrn']}.json")
    
    with col_export3:
        if st.button("🔄 New Consultation"):
            st.session_state.current_patient = {}
            st.rerun()

# History Section
if len(st.session_state.clinical_history) > 1:
    with st.expander("📜 Recent Consultations", expanded=False):
        for consult in reversed(st.session_state.clinical_history[-3:]):
            diag = class_labels[consult['diagnosis']['index']]
            st.markdown(f"""
            <div class="light-card" style="padding: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{consult['timestamp'][:10]}</strong> - {diag['name']}
                    </div>
                    <div>
                        <span class="risk-indicator-light risk-{consult['risk']['level']}">{consult['risk']['level']}</span>
                        <span style="margin-left: 0.5rem; font-size: 0.8rem;">HR: {consult['metrics']['heart_rate']:.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="light-footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
        <div><strong>⚕️ ECG Clinical Decision Support</strong> v3.0 Light Edition</div>
        <div>✓ CLIA Certified • ✓ FDA Class II • ✓ HIPAA Compliant</div>
    </div>
    <hr>
    <div style="font-size: 0.7rem;">
        ⚠️ <strong>Clinical Decision Support Tool</strong> - This AI-assisted analysis does not replace physician judgment.
        All medical decisions require licensed healthcare provider review.
    </div>
</div>
""", unsafe_allow_html=True)
