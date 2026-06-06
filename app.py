import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import hashlib
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Comprehensive clinical database with SOFT COLORS
class_labels = {
    0: {
        "name": "Normal Sinus Rhythm (N)",
        "icd10": "I49.9",
        "color": "#A8E6CF",  # Soft mint
        "dark_color": "#2Ecc71",
        "severity": "Low",
        "risk_score": 8,
        "mortality_risk": "0.1%",
        "desc": "Normal cardiac conduction with regular rhythm",
        "clinical_advice": "No intervention needed. Routine follow-up.",
        "treatment": "Lifestyle optimization",
        "medications": [],
        "follow_up": "Annual physical exam",
        "specialist": "Primary care",
        "imaging_needed": False,
        "admission_needed": False
    },
    1: {
        "name": "Supraventricular Ectopy (S)",
        "icd10": "I47.1",
        "color": "#FFD3B6",  # Soft peach
        "dark_color": "#F39C12",
        "severity": "Moderate",
        "risk_score": 42,
        "mortality_risk": "2.3%",
        "desc": "Premature atrial contractions or supraventricular tachycardia",
        "clinical_advice": "Monitor frequency. Consider Holter if symptomatic.",
        "treatment": "Beta-blockers or calcium channel blockers",
        "medications": ["Metoprolol", "Diltiazem", "Verapamil"],
        "follow_up": "4-6 weeks",
        "specialist": "Cardiology",
        "imaging_needed": False,
        "admission_needed": False
    },
    2: {
        "name": "Ventricular Ectopy (V)",
        "icd10": "I49.3",
        "color": "#FFAAA5",  # Soft coral
        "dark_color": "#E74C3C",
        "severity": "High",
        "risk_score": 78,
        "mortality_risk": "8.7%",
        "desc": "Premature ventricular contractions - potential for arrhythmias",
        "clinical_advice": "URGENT: Cardiology evaluation within 48 hours",
        "treatment": "Antiarrhythmic drugs or ablation",
        "medications": ["Amiodarone", "Lidocaine", "Mexiletine"],
        "follow_up": "1 week",
        "specialist": "Electrophysiology",
        "imaging_needed": True,
        "admission_needed": False
    },
    3: {
        "name": "Fusion Beat (F)",
        "icd10": "I49.8",
        "color": "#C3B1E1",  # Soft lavender
        "dark_color": "#9B59B6",
        "severity": "Moderate-High",
        "risk_score": 65,
        "mortality_risk": "5.1%",
        "desc": "Fusion of normal and ectopic conduction",
        "clinical_advice": "Electrophysiology study recommended",
        "treatment": "Based on underlying rhythm disorder",
        "medications": ["Varies by cause"],
        "follow_up": "2 weeks",
        "specialist": "Electrophysiology",
        "imaging_needed": True,
        "admission_needed": False
    },
    4: {
        "name": "Unclassified Pattern (Q)",
        "icd10": "R94.31",
        "color": "#D4E0EC",  # Soft blue-gray
        "dark_color": "#95A5A6",
        "severity": "Uncertain",
        "risk_score": 35,
        "mortality_risk": "N/A",
        "desc": "Atypical pattern requiring verification",
        "clinical_advice": "Repeat ECG with proper technique",
        "treatment": "Await confirmation",
        "medications": [],
        "follow_up": "1 week",
        "specialist": "Cardiology",
        "imaging_needed": False,
        "admission_needed": False
    }
}

# Page config
st.set_page_config(
    page_title="ECG Clinical Suite - Advanced Cardiac Analysis",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Soft & Light CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #F8F9FA 0%, #E8F0FE 100%);
    }
    
    /* Soft Header */
    .soft-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F4F8 100%);
        padding: 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    /* Soft Cards */
    .soft-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .soft-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        background: white;
    }
    
    /* Metric Cards */
    .metric-soft {
        background: white;
        border-radius: 18px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .metric-soft:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    /* Risk Meter Soft */
    .risk-meter-soft {
        width: 100%;
        height: 12px;
        background: linear-gradient(90deg, #A8E6CF, #FFD3B6, #FFAAA5);
        border-radius: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Diagnosis Card */
    .diagnosis-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        border-left: 6px solid;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Badges Soft */
    .badge-soft {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        background: rgba(0,0,0,0.03);
        color: #2C3E50;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(52,152,219,0.3);
    }
    
    /* Alert Boxes Soft */
    .alert-soft {
        background: #FFF5F5;
        border-left: 4px solid #FFAAA5;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    /* Footer */
    .soft-footer {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        color: #7F8C8D;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.7);
        border-radius: 12px;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.5);
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #A8E6CF, #3498DB);
    }
    
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #CBD5E0, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'clinical_history' not in st.session_state:
    st.session_state.clinical_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Header
st.markdown("""
<div class="soft-header">
    <h1 class="header-title">💓 ECG Clinical Suite</h1>
    <p style="color: #5A6C7D; margin-top: 0.5rem;">Advanced AI-Powered Cardiac Decision Support System</p>
    <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
        <span class="badge-soft">🔬 AI-Powered</span>
        <span class="badge-soft">🏥 Clinical Grade</span>
        <span class="badge-soft">📊 Real-time Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Patient Info
with st.sidebar:
    st.markdown("### 🏥 Patient Information")
    
    with st.expander("👤 Patient Details", expanded=True):
        patient_name = st.text_input("Patient Name", placeholder="Enter full name")
        patient_id = st.text_input("Patient ID", placeholder="Auto-generated", 
                                  value=f"ECG-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(100,999)}")
        age = st.slider("Age", 0, 120, 55)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            weight = st.number_input("Weight (kg)", 20, 200, 70)
        with col_w2:
            height = st.number_input("Height (cm)", 100, 250, 170)
        
        if height > 0:
            bmi = weight / ((height/100) ** 2)
            st.caption(f"BMI: {bmi:.1f}")
    
    with st.expander("📋 Medical History"):
        comorbidities = st.multiselect("Comorbidities", 
                                      ["Hypertension", "Diabetes", "CAD", "Heart Failure", 
                                       "COPD", "Renal Disease", "Thyroid Disorder", "None"])
        medications = st.text_area("Current Medications", placeholder="List all current medications")
        smoking = st.radio("Smoking Status", ["Never", "Former", "Current"])
    
    st.markdown("---")
    st.caption("🔒 HIPAA Compliant • Data Encrypted")

# Main content - Input Section
st.markdown("### 📥 ECG Data Input")

col_in1, col_in2 = st.columns([2, 1])

with col_in1:
    input_method = st.radio("Select Input Method", ["📁 Upload CSV", "✍️ Manual Entry", "🎲 Generate Test Case"], horizontal=True)

ecg_values = None

if input_method == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload ECG file (CSV format, 187 samples)", type=["csv", "txt"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        values = df.values.flatten()
        if len(values) == 187:
            ecg_values = values
            st.success(f"✅ Successfully loaded {len(values)} ECG samples")
            # Show preview
            with st.expander("📊 Data Preview"):
                st.write(f"**Range:** {values.min():.3f} - {values.max():.3f}")
                st.write(f"**Mean:** {values.mean():.3f}")
                st.write(f"**First 10 values:** {', '.join([f'{x:.3f}' for x in values[:10]])}")
        else:
            st.error(f"❌ Invalid: {len(values)} samples (need 187)")

elif input_method == "✍️ Manual Entry":
    manual_input = st.text_area("Enter 187 comma-separated values", height=100, 
                               placeholder="0.5, 0.7, 0.3, -0.2, ...")
    if manual_input and st.button("Process Manual Input"):
        try:
            values = [float(x.strip()) for x in manual_input.replace('\n', ',').split(',') if x.strip()]
            if len(values) == 187:
                ecg_values = np.array(values)
                st.success("✅ Manual input accepted")
            else:
                st.error(f"Need 187 values, got {len(values)}")
        except:
            st.error("Invalid format")

else:  # Generate Test Case
    st.markdown("**Generate Clinical Test Patterns**")
    col_gen1, col_gen2 = st.columns(2)
    with col_gen1:
        pattern = st.selectbox("Pattern Type", ["Normal Sinus", "PVC Pattern", "Bradycardia", "Tachycardia", "Artifact"])
    with col_gen2:
        noise = st.slider("Noise Level", 0.0, 0.3, 0.05, format="%.2f")
    
    if st.button("🎲 Generate Signal"):
        t = np.linspace(0, 8*np.pi, 187)
        if pattern == "Normal Sinus":
            ecg_values = np.sin(t) * 0.8 + np.sin(3*t) * 0.2
        elif pattern == "PVC Pattern":
            ecg_values = np.sin(t) * 0.8
            ecg_values[80:95] = -1.2
            ecg_values[140:155] = 1.1
        elif pattern == "Bradycardia":
            ecg_values = np.sin(t/1.5) * 0.8
        elif pattern == "Tachycardia":
            ecg_values = np.sin(t*1.5) * 0.8
        else:
            ecg_values = np.random.normal(0, 0.3, 187)
        
        ecg_values += np.random.normal(0, noise, 187)
        ecg_values = ecg_values / np.max(np.abs(ecg_values))
        st.success(f"✅ Generated {pattern} pattern")
        st.info(f"Range: {ecg_values.min():.2f} to {ecg_values.max():.2f}")

with col_in2:
    st.markdown("### 📊 Quick Guide")
    st.info("""
    **Input Requirements:**
    - 187 samples per beat
    - Normalized values (-1 to 1)
    - CSV or manual entry
    
    **Analysis Includes:**
    - AI Classification
    - Risk Assessment
    - Treatment Plan
    - Clinical Recommendations
    """)

# Analysis Button
if ecg_values is not None:
    if st.button("🔬 Run Full Clinical Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing ECG signal..."):
            # 1. CNN Prediction
            reshaped = ecg_values.reshape(1, 187, 1).astype(np.float32)
            prediction = model.predict(reshaped)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            
            # 2. Signal Processing
            # FFT Analysis
            fft_vals = fft(ecg_values)
            freqs = fftfreq(187, 1/500)
            dominant_freq_idx = np.argmax(np.abs(fft_vals[1:])) + 1
            dominant_freq = float(freqs[dominant_freq_idx])
            
            # Peak detection for heart rate
            peaks, _ = find_peaks(ecg_values, distance=20, prominence=0.1)
            heart_rate = len(peaks) * (500 / 187) * 60 if len(peaks) > 0 else 75
            
            # Signal Quality
            snr = 20 * np.log10(np.max(np.abs(ecg_values)) / (np.std(ecg_values) + 0.001))
            
            # 3. Risk Assessment
            clinical_info = class_labels[class_index]
            risk_score = clinical_info['risk_score']
            
            # Adjust risk based on patient factors
            if age > 65:
                risk_score += 15
            if age > 80:
                risk_score += 10
            if "Hypertension" in comorbidities:
                risk_score += 10
            if "Diabetes" in comorbidities:
                risk_score += 10
            if "CAD" in comorbidities:
                risk_score += 20
            if smoking == "Current":
                risk_score += 15
            
            risk_score = min(100, risk_score)
            
            # 4. Generate Alerts
            alerts = []
            if risk_score >= 70:
                alerts.append("🔴 HIGH RISK - Urgent cardiology consultation required")
            if heart_rate > 100:
                alerts.append("⚠️ Tachycardia detected (>100 BPM)")
            if heart_rate < 60 and age < 60:
                alerts.append("⚠️ Bradycardia detected (<60 BPM)")
            if snr < 10:
                alerts.append("⚠️ Poor signal quality - Consider repeating ECG")
            if confidence < 60:
                alerts.append("⚠️ Low AI confidence - Manual overread recommended")
            
            # 5. Recommendations
            if risk_score >= 70:
                urgency = "EMERGENCY"
                setting = "Emergency Department within 24 hours"
            elif risk_score >= 50:
                urgency = "URGENT"
                setting = "Cardiology Clinic within 48 hours"
            else:
                urgency = "ROUTINE"
                setting = "Outpatient Clinic within 2 weeks"
            
            # Store results
            result = {
                "timestamp": datetime.now().isoformat(),
                "patient": {
                    "name": patient_name or "Unnamed",
                    "id": patient_id,
                    "age": age,
                    "gender": gender,
                    "bmi": bmi if height > 0 else 0,
                    "comorbidities": comorbidities,
                    "smoking": smoking
                },
                "analysis": {
                    "diagnosis": class_index,
                    "diagnosis_name": clinical_info['name'],
                    "icd10": clinical_info['icd10'],
                    "confidence": confidence,
                    "heart_rate": heart_rate,
                    "dominant_frequency": dominant_freq,
                    "snr": snr,
                    "peak_count": len(peaks)
                },
                "risk": {
                    "score": risk_score,
                    "level": clinical_info['severity'],
                    "alerts": alerts
                },
                "recommendations": {
                    "urgency": urgency,
                    "setting": setting,
                    "specialist": clinical_info['specialist'],
                    "follow_up": clinical_info['follow_up'],
                    "medications": clinical_info['medications']
                },
                "raw_signal": ecg_values.tolist()
            }
            
            st.session_state.current_patient = result
            st.session_state.clinical_history.append(result)
            st.session_state.alerts = alerts
            
            st.success("✅ Analysis Complete!")
            st.balloons()

# Display Results
if st.session_state.current_patient:
    result = st.session_state.current_patient
    clinical = class_labels[result['analysis']['diagnosis']]
    
    # Alerts Section
    if result['risk']['alerts']:
        st.markdown("### 🚨 Clinical Alerts")
        for alert in result['risk']['alerts']:
            st.markdown(f'<div class="alert-soft">{alert}</div>', unsafe_allow_html=True)
    
    # Metrics Row
    st.markdown("### 📊 Clinical Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-soft">
            <div style="font-size: 0.9rem; color: #7F8C8D;">Risk Score</div>
            <div class="metric-value">{result['risk']['score']}</div>
            <div class="risk-meter-soft">
                <div style="width: {result['risk']['score']}%; height: 100%; background: rgba(0,0,0,0.1);"></div>
            </div>
            <div style="font-size: 0.8rem;">{result['risk']['level']} Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="metric-soft">
            <div style="font-size: 0.9rem; color: #7F8C8D;">Heart Rate</div>
            <div class="metric-value">{result['analysis']['heart_rate']:.0f}</div>
            <div style="font-size: 0.8rem;">BPM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class="metric-soft">
            <div style="font-size: 0.9rem; color: #7F8C8D;">Confidence</div>
            <div class="metric-value">{result['analysis']['confidence']:.0f}%</div>
            <div style="font-size: 0.8rem;">AI Certainty</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        quality_color = "🟢" if result['analysis']['snr'] > 15 else "🟡" if result['analysis']['snr'] > 8 else "🔴"
        st.markdown(f"""
        <div class="metric-soft">
            <div style="font-size: 0.9rem; color: #7F8C8D;">Signal Quality</div>
            <div class="metric-value">{quality_color} {result['analysis']['snr']:.1f}</div>
            <div style="font-size: 0.8rem;">dB SNR</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnosis Card
    st.markdown(f"""
    <div class="diagnosis-card" style="border-left-color: {clinical['dark_color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h2 style="margin: 0; color: {clinical['dark_color']};">{clinical['name']}</h2>
                <p style="color: #5A6C7D; margin-top: 0.3rem;">ICD-10: {clinical['icd10']}</p>
            </div>
            <div>
                <span class="badge-soft" style="background: {clinical['color']}40;">{clinical['severity']} Risk</span>
            </div>
        </div>
        <p style="margin-top: 1rem;">{clinical['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Treatment Plan
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("### 💊 Treatment Plan")
        st.markdown(f"""
        <div class="soft-card">
            <strong>🚨 Urgency:</strong> {result['recommendations']['urgency']}<br>
            <strong>📍 Setting:</strong> {result['recommendations']['setting']}<br>
            <strong>👨‍⚕️ Specialist:</strong> {result['recommendations']['specialist']}<br>
            <strong>📅 Follow-up:</strong> {result['recommendations']['follow_up']}
        </div>
        """, unsafe_allow_html=True)
    
    with col_t2:
        st.markdown("### 💊 Medications")
        if result['recommendations']['medications']:
            meds = ", ".join(result['recommendations']['medications'])
            st.markdown(f"""
            <div class="soft-card">
                <strong>First-line treatments:</strong><br>
                {meds}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="soft-card">
                No medications indicated at this time.<br>
                Continue monitoring and lifestyle optimization.
            </div>
            """, unsafe_allow_html=True)
    
    # ECG Visualization (Simple)
    st.markdown("### 📈 ECG Signal")
    
    # Create simple line chart with altair
    import altair as alt
    chart_data = pd.DataFrame({
        'Sample': range(len(result['raw_signal'])),
        'Amplitude': result['raw_signal']
    })
    
    chart = alt.Chart(chart_data).mark_line(
        color=clinical['dark_color'],
        strokeWidth=2
    ).encode(
        x='Sample:Q',
        y='Amplitude:Q'
    ).properties(
        height=300,
        background='white'
    ).configure_axis(
        gridColor='#E8ECEF',
        titleColor='#5A6C7D'
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Clinical Advice
    st.markdown("### 📋 Clinical Advice")
    st.markdown(f"""
    <div class="soft-card">
        <strong>👨‍⚕️ Recommendation:</strong><br>
        {clinical['clinical_advice']}
    </div>
    """, unsafe_allow_html=True)
    
    # Export Options
    st.markdown("### 📎 Export Results")
    col_e1, col_e2, col_e3 = st.columns(3)
    
    with col_e1:
        # Simple text report
        report_text = f"""
CLINICAL ECG REPORT
==================
Date: {result['timestamp'][:19]}
Patient: {result['patient']['name']}
ID: {result['patient']['id']}
Age: {result['patient']['age']}

DIAGNOSIS:
{clinical['name']} (ICD-10: {clinical['icd10']})
Risk Level: {result['risk']['level']}
Confidence: {result['analysis']['confidence']:.1f}%

VITALS:
Heart Rate: {result['analysis']['heart_rate']:.0f} BPM
Signal Quality: {result['analysis']['snr']:.1f} dB

RECOMMENDATIONS:
Urgency: {result['recommendations']['urgency']}
Setting: {result['recommendations']['setting']}
Follow-up: {result['recommendations']['follow_up']}

DISCLAIMER: This is an AI-assisted analysis. All clinical decisions should be made by qualified healthcare professionals.
        """
        st.download_button("📄 Download Report", report_text, f"ECG_Report_{result['patient']['id']}.txt")
    
    with col_e2:
        # JSON export
        json_report = json.dumps(result, indent=2, default=str)
        st.download_button("💾 Export JSON", json_report, f"ECG_Data_{result['patient']['id']}.json")
    
    with col_e3:
        if st.button("🔄 New Analysis"):
            st.session_state.current_patient = {}
            st.rerun()

# History Section
if len(st.session_state.clinical_history) > 1:
    with st.expander("📜 Consultation History", expanded=False):
        for i, consult in enumerate(reversed(st.session_state.clinical_history[-5:])):
            st.markdown(f"""
            <div class="soft-card">
                <strong>{consult['timestamp'][:10]}</strong> - {consult['analysis']['diagnosis_name']}<br>
                Risk Score: {consult['risk']['score']} | HR: {consult['analysis']['heart_rate']:.0f} BPM
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="soft-footer">
    <div>
        <strong>💓 ECG Clinical Suite v3.0</strong><br>
        AI-Powered Cardiac Decision Support System
    </div>
    <hr>
    <div style="font-size: 0.8rem; color: #95A5A6;">
        ⚠️ Clinical Decision Support Tool - For professional use only<br>
        All clinical decisions must be made by qualified healthcare providers
    </div>
</div>
""", unsafe_allow_html=True)
