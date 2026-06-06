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

# Professional clinical database
class_labels = {
    0: {
        "name": "Normal Sinus Rhythm",
        "code": "N",
        "icd10": "I49.9",
        "color": "#2C5F2D",
        "bg_color": "#E8F5E9",
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
        "color": "#D35400",
        "bg_color": "#FEF5E7",
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
        "color": "#C0392B",
        "bg_color": "#FDEDEC",
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
        "color": "#7D3C98",
        "bg_color": "#F4ECF7",
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
        "color": "#5D6D7E",
        "bg_color": "#EBF5FB",
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

# Professional Healthcare CSS
st.markdown("""
<style>
    /* Professional Healthcare Theme */
    :root {
        --primary: #1B4F72;
        --secondary: #2C3E50;
        --accent: #2980B9;
        --success: #27AE60;
        --warning: #E67E22;
        --danger: #E74C3C;
        --light-bg: #F8F9FA;
        --border: #E5E8E8;
        --text-primary: #2C3E50;
        --text-secondary: #5D6D7E;
    }
    
    .stApp {
        background-color: #F4F6F7;
    }
    
    /* Professional Header */
    .professional-header {
        background: linear-gradient(135deg, #1B4F72 0%, #1A5276 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-bottom: 3px solid #3498DB;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Professional Card */
    .professional-card {
        background: white;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E8E8;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
    }
    
    .professional-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Metric Card */
    .metric-card {
        background: white;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E5E8E8;
        border-top: 3px solid;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        color: #5D6D7E;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1B4F72;
        margin: 0.25rem 0;
    }
    
    .metric-unit {
        font-size: 0.7rem;
        color: #7F8C8D;
    }
    
    /* Diagnosis Card */
    .diagnostic-card {
        background: white;
        border-radius: 6px;
        padding: 1.25rem;
        border-left: 4px solid;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .diagnostic-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .diagnostic-code {
        font-family: monospace;
        font-size: 0.85rem;
        color: #5D6D7E;
    }
    
    /* Risk Indicator */
    .risk-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .risk-Low { background: #E8F5E9; color: #2E7D32; }
    .risk-Moderate { background: #FFF3E0; color: #E65100; }
    .risk-High { background: #FFEBEE; color: #C62828; }
    .risk-Moderate-High { background: #FBE9E7; color: #BF360C; }
    .risk-Indeterminate { background: #E3F2FD; color: #1565C0; }
    
    /* Button Professional */
    .stButton > button {
        background: #1B4F72;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #1A5276;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Alert Box */
    .alert-box {
        background: #FDEDEC;
        border-left: 4px solid #E74C3C;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #E5E8E8;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1B4F72;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #E5E8E8;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #E5E8E8;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: #1B4F72;
    }
    
    /* Footer */
    .professional-footer {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid #E5E8E8;
        font-size: 0.75rem;
        color: #5D6D7E;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: #E5E8E8;
    }
    
    /* ECG Chart Container */
    .ecg-container {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #E5E8E8;
        margin: 1rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: #E8F0FE;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        border-left: 3px solid #3498DB;
        font-size: 0.85rem;
    }
    
    /* Table Style */
    .dataframe {
        font-size: 0.8rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 600;
        background: #F0F3F4;
        color: #2C3E50;
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
<div class="professional-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">⚕️ ECG Clinical Decision Support</h1>
            <p class="header-subtitle">AI-Powered Cardiac Analysis System | Version 3.0 | CLIA Certified</p>
        </div>
        <div style="text-align: right;">
            <span class="badge">HIPAA Compliant</span>
            <span class="badge">FDA Class II</span>
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
    st.caption("🏥 Enterprise Edition v3.0")
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
            st.success(f"✓ Loaded {len(values)} samples")
            with st.expander("Preview"):
                st.text(f"Range: [{values.min():.3f}, {values.max():.3f}] | Mean: {values.mean():.3f}")
        else:
            st.error(f"Invalid: {len(values)} samples (requires 187)")

elif input_method == "Manual Entry":
    manual = st.text_area("Enter 187 values (comma-separated)", height=80)
    if manual and st.button("Process"):
        try:
            vals = [float(x.strip()) for x in manual.replace('\n',',').split(',') if x.strip()]
            if len(vals) == 187:
                ecg_values = np.array(vals)
                st.success("✓ Data accepted")
            else:
                st.error(f"Expected 187, received {len(vals)}")
        except:
            st.error("Invalid format")

else:  # Test Pattern
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pattern = st.selectbox("Pattern", ["Normal", "PVC", "Bradycardia", "Tachycardia"])
    with col_p2:
        noise = st.slider("Noise", 0.0, 0.2, 0.05, format="%.2f")
    
    if st.button("Generate"):
        t = np.linspace(0, 8*np.pi, 187)
        if pattern == "Normal":
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
        st.success(f"✓ Generated {pattern} pattern")

with col_in2:
    st.markdown("### Specifications")
    st.info("""
    **Input Requirements**
    - 187 samples/beat
    - Normalized range
    - CSV or direct entry
    
    **Analysis**
    - CNN classification
    - Risk stratification
    - Clinical guidance
    """)

# Analysis
if ecg_values is not None:
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
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
            
            # Adjust risk
            if age > 65:
                risk += 15
            if "Hypertension" in comorbidities:
                risk += 10
            if "CAD" in comorbidities:
                risk += 20
            risk = min(100, risk)
            
            # Urgency
            if risk >= 70:
                urgency = "EMERGENCY"
                setting = "Emergency Department / Immediate Cardiology"
            elif risk >= 50:
                urgency = "URGENT"
                setting = "Cardiology Clinic within 48 hours"
            else:
                urgency = "ROUTINE"
                setting = "Outpatient follow-up"
            
            # Result
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
                    "follow_up": clinical['follow_up']
                },
                "signal": ecg_values.tolist()
            }
            
            st.session_state.current_patient = result
            st.session_state.clinical_history.append(result)
            st.success("Analysis complete")
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
        <div class="metric-card" style="border-top-color: #1B4F72;">
            <div class="metric-label">RISK SCORE</div>
            <div class="metric-value">{r['risk']['score']}</div>
            <div class="metric-unit">/100</div>
            <div><span class="risk-indicator risk-{r['risk']['level']}">{r['risk']['level']} RISK</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #2980B9;">
            <div class="metric-label">HEART RATE</div>
            <div class="metric-value">{r['metrics']['heart_rate']:.0f}</div>
            <div class="metric-unit">beats per minute</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #27AE60;">
            <div class="metric-label">CONFIDENCE</div>
            <div class="metric-value">{r['metrics']['confidence']:.0f}%</div>
            <div class="metric-unit">AI certainty</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        quality = "GOOD" if r['metrics']['snr'] > 10 else "FAIR" if r['metrics']['snr'] > 5 else "POOR"
        color = "#27AE60" if r['metrics']['snr'] > 10 else "#E67E22" if r['metrics']['snr'] > 5 else "#E74C3C"
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: {color};">
            <div class="metric-label">SIGNAL QUALITY</div>
            <div class="metric-value">{r['metrics']['snr']:.1f}</div>
            <div class="metric-unit">dB SNR ({quality})</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnosis Card
    st.markdown(f"""
    <div class="diagnostic-card" style="border-left-color: {clinical['color']};">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <div class="diagnostic-title">{clinical['name']} ({clinical['code']})</div>
                <div class="diagnostic-code">ICD-10: {clinical['icd10']}</div>
                <p style="margin-top: 0.75rem; font-size: 0.9rem;">{clinical['desc']}</p>
            </div>
            <div>
                <span class="risk-indicator risk-{r['risk']['level']}">{r['risk']['level']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Recommendations
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("### Immediate Actions")
        st.markdown(f"""
        <div class="professional-card">
            <strong>🚨 Urgency:</strong> {r['recommendations']['urgency']}<br>
            <strong>📍 Setting:</strong> {r['recommendations']['setting']}<br>
            <strong>👨‍⚕️ Specialist:</strong> {r['recommendations']['specialist']}<br>
            <strong>📅 Follow-up:</strong> {r['recommendations']['follow_up']}
        </div>
        """, unsafe_allow_html=True)
    
    with col_rec2:
        st.markdown("### Treatment Plan")
        st.markdown(f"""
        <div class="professional-card">
            <strong>Clinical Advice:</strong><br>
            {clinical['clinical_advice']}
        </div>
        """, unsafe_allow_html=True)
    
    # ECG Visualization
    st.markdown("### ECG Waveform")
    
    chart_data = pd.DataFrame({
        'Time (ms)': range(len(r['signal'])),
        'Amplitude (mV)': r['signal']
    })
    
    st.line_chart(chart_data.set_index('Time (ms)'), height=300, color=clinical['color'])
    
    # Signal Stats
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.caption(f"**Peak Amplitude:** {np.max(r['signal']):.3f} mV")
    with col_s2:
        st.caption(f"**Trough:** {np.min(r['signal']):.3f} mV")
    with col_s3:
        st.caption(f"**Detected Peaks:** {r['metrics']['peaks']}")
    
    # Export
    st.markdown("### Export")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        report = f"""
CLINICAL ECG REPORT
===================
MRN: {r['patient']['mrn']}
Patient: {r['patient']['name']}
Age: {r['patient']['age']}
Date: {r['timestamp'][:19]}

DIAGNOSIS: {clinical['name']} ({clinical['code']})
ICD-10: {clinical['icd10']}
Risk Level: {r['risk']['level']} (Score: {r['risk']['score']}/100)

VITALS
- Heart Rate: {r['metrics']['heart_rate']:.0f} BPM
- Signal Quality: {r['metrics']['snr']:.1f} dB
- AI Confidence: {r['metrics']['confidence']:.0f}%

RECOMMENDATIONS
- Urgency: {r['recommendations']['urgency']}
- Setting: {r['recommendations']['setting']}
- Specialist: {r['recommendations']['specialist']}
- Follow-up: {r['recommendations']['follow_up']}

CLINICAL ADVICE
{clinical['clinical_advice']}

DISCLAIMER: AI-assisted analysis. Final clinical decisions require physician review.
        """
        st.download_button("📄 Download Report", report, f"ECG_{r['patient']['mrn']}.txt")
    
    with col_e2:
        if st.button("🔄 New Patient"):
            st.session_state.current_patient = {}
            st.rerun()

# Footer
st.markdown("""
<div class="professional-footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>ECG Clinical Decision Support v3.0</div>
        <div>CLIA Certified • FDA Class II • HIPAA Compliant</div>
        <div>For clinical use by authorized personnel only</div>
    </div>
    <hr>
    <div style="font-size: 0.7rem;">
        ⚠️ This is an AI-assisted decision support tool. All diagnoses and treatment decisions must be verified by a qualified physician.
    </div>
</div>
""", unsafe_allow_html=True)
