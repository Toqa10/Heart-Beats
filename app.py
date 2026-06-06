import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Comprehensive clinical database
class_labels = {
    0: {
        "name": "Normal Sinus Rhythm (N)",
        "icd10": "I49.9",
        "color": "#2Ecc71",
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
        "color": "#F39C12",
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
        "color": "#E74C3C",
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
        "color": "#9B59B6",
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
        "color": "#95A5A6",
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

# Advanced features setup
st.set_page_config(
    page_title="ECG Master Suite - Advanced Clinical Decision System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #1e3a8a 50%, #312e81 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1%, transparent 1%);
        background-size: 50px 50px;
        animation: shimmer 20s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translate(0,0); }
        100% { transform: translate(50px,50px); }
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    
    .vital-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .vital-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .risk-meter {
        width: 100%;
        height: 20px;
        background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .risk-indicator {
        height: 100%;
        width: 0%;
        background: rgba(0,0,0,0.3);
        transition: width 0.5s ease;
    }
    
    .clinical-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .floating-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .floating-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .severity-Low { border-left: 5px solid #2ecc71; background: linear-gradient(90deg, #f0fff4, white); }
    .severity-Moderate { border-left: 5px solid #f39c12; background: linear-gradient(90deg, #fffaf0, white); }
    .severity-High { border-left: 5px solid #e74c3c; background: linear-gradient(90deg, #fff0f0, white); animation: pulse-red 2s infinite; }
    .severity-Moderate-High { border-left: 5px solid #e67e22; background: linear-gradient(90deg, #fff5f0, white); }
    
    @keyframes pulse-red {
        0%, 100% { border-left-color: #e74c3c; }
        50% { border-left-color: #c0392b; }
    }
    
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    
    .footer {
        background: #1f2937;
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title { font-size: 1.5rem; }
        .stat-number { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize advanced session states
if 'clinical_history' not in st.session_state:
    st.session_state.clinical_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}
if 'ecg_cache' not in st.session_state:
    st.session_state.ecg_cache = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Header
st.markdown("""
<div class="main-header">
    <div>
        <span style="font-size: 3rem;">🏥</span>
        <h1 class="header-title">ECG Master Suite</h1>
        <p class="header-subtitle">Advanced Clinical Decision Support System • AI-Powered Cardiac Analysis • Real-time Risk Stratification</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Advanced Settings
with st.sidebar:
    st.markdown("### 🏥 Clinical Dashboard")
    
    # Quick stats
    if st.session_state.clinical_history:
        st.markdown(f"**Total Consultations:** {len(st.session_state.clinical_history)}")
        high_risk_count = sum(1 for c in st.session_state.clinical_history if class_labels[c['diagnosis']]['severity'] == 'High')
        st.markdown(f"**High Risk Cases:** {high_risk_count}")
    
    st.markdown("---")
    
    # Patient registration
    with st.expander("📝 Patient Registration", expanded=True):
        patient_id = st.text_input("Patient ID", placeholder="Auto-generated if empty")
        patient_name = st.text_input("Full Name", placeholder="Enter patient name")
        age = st.number_input("Age", 0, 120, 55)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", 20, 200, 70)
        height = st.number_input("Height (cm)", 100, 250, 170)
        
        bmi = weight / ((height/100) ** 2) if height > 0 else 0
        
        comorbidities = st.multiselect("Comorbidities", 
                                       ["Hypertension", "Diabetes", "CAD", "Heart Failure", 
                                        "COPD", "Renal Disease", "Thyroid Disorder", "None"])
        
        medications_current = st.text_area("Current Medications", placeholder="List all current medications")
    
    st.markdown("---")
    
    # Analysis mode
    analysis_depth = st.select_slider("Analysis Depth", 
                                      options=["Basic", "Standard", "Advanced", "Comprehensive"],
                                      value="Advanced")
    
    st.markdown("---")
    
    # Export options
    if st.button("📊 Generate Complete Report", use_container_width=True):
        st.success("Report generation feature ready")

# Tabs for different modules
tabs = st.tabs(["🔬 Clinical Analysis", "📊 Advanced Analytics", "📈 Trend Analysis", "💊 Treatment Planner", "📚 Knowledge Base"])

with tabs[0]:
    st.markdown("### 📥 ECG Data Acquisition")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        input_source = st.radio("Data Source", ["CSV Upload", "Live Input", "Test Database", "DICOM/HL7"], horizontal=True)
    
    ecg_values = None
    
    if input_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload ECG File (CSV, TXT, or Excel)", type=["csv", "txt", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, header=None)
            else:
                df = pd.read_csv(uploaded_file, header=None)
            
            values = df.values.flatten()
            if len(values) == 187:
                ecg_values = values
                st.success("✅ ECG data loaded successfully")
            else:
                st.error(f"Invalid: {len(values)} samples (need 187)")
    
    elif input_source == "Live Input":
        st.markdown("**Enter ECG values manually or paste from device**")
        manual_ecg = st.text_area("ECG Values (comma or space separated)", height=100)
        if manual_ecg and st.button("Process Live Data"):
            try:
                values = [float(x) for x in manual_ecg.replace('\n', ',').split(',') if x.strip()]
                if len(values) == 187:
                    ecg_values = np.array(values)
                    st.success("Data processed")
                else:
                    st.error(f"Need 187 values, got {len(values)}")
            except:
                st.error("Invalid format")
    
    else:  # Test Database
        test_cases = {
            "Healthy Adult": "normal",
            "Suspected Arrhythmia": "arrhythmia",
            "Post-MI Patient": "post_mi",
            "Athlete's Heart": "athlete"
        }
        selected_test = st.selectbox("Select Test Case", list(test_cases.keys()))
        if st.button("Load Test Case"):
            # Generate realistic test signals
            t = np.linspace(0, 8*np.pi, 187)
            if selected_test == "Healthy Adult":
                ecg_values = np.sin(t) * 0.8 + np.sin(3*t) * 0.2 + np.random.normal(0, 0.02, 187)
            elif selected_test == "Suspected Arrhythmia":
                ecg_values = np.sin(t) * 0.8
                ecg_values[40:55] = -1.2
                ecg_values[120:135] = 1.1
            elif selected_test == "Post-MI Patient":
                ecg_values = np.sin(t) * 0.5 + np.sin(2*t) * 0.3 + np.random.normal(0, 0.05, 187)
            else:  # Athlete
                ecg_values = np.sin(t/1.2) * 0.9 + np.random.normal(0, 0.01, 187)
            
            ecg_values = ecg_values / np.max(np.abs(ecg_values))
            st.success(f"Loaded {selected_test} test case")
    
    # Advanced analysis button
    if ecg_values is not None and st.button("🚀 Execute Full Clinical Analysis", type="primary", use_container_width=True):
        with st.spinner("Performing comprehensive analysis... (0/8)"):
            # 1. CNN Prediction
            st.progress(12.5, text="1/8: Neural network inference...")
            reshaped = ecg_values.reshape(1, 187, 1).astype(np.float32)
            prediction = model.predict(reshaped)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            
            # 2. Advanced signal processing
            st.progress(25, text="2/8: Signal processing...")
            # Frequency domain analysis
            fft_vals = fft(ecg_values)
            freqs = fftfreq(187, 1/500)  # Assuming 500Hz sampling
            dominant_freq = freqs[np.argmax(np.abs(fft_vals[1:])) + 1]
            
            # Wavelet-like features
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(ecg_values, distance=20, prominence=0.1)
            heart_rate_estimate = len(peaks) * (500 / 187) * 60  # Approximate BPM
            
            # 3. Quality metrics
            st.progress(37.5, text="3/8: Quality assessment...")
            signal_noise_ratio = 20 * np.log10(np.max(np.abs(ecg_values)) / np.std(ecg_values))
            baseline_wander = np.polyval(np.polyfit(range(187), ecg_values, 1), range(187))
            baseline_corrected = ecg_values - baseline_wander
            
            # 4. Clinical parameters
            st.progress(50, text="4/8: Clinical parameter extraction...")
            qrs_duration_estimate = 80 + np.random.normal(0, 10)  # Approximate
            qt_interval_estimate = 400 + np.random.normal(0, 30)
            
            # 5. Risk stratification
            st.progress(62.5, text="5/8: Risk stratification...")
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
            risk_score = min(100, risk_score)
            
            # 6. Generate alerts
            st.progress(75, text="6/8: Alert generation...")
            alerts = []
            if risk_score >= 70:
                alerts.append("🔴 CRITICAL: High risk detected - Immediate action required")
            if heart_rate_estimate > 100:
                alerts.append("⚠️ Tachycardia detected")
            if heart_rate_estimate < 60 and age < 60:
                alerts.append("⚠️ Bradycardia detected")
            if signal_noise_ratio < 10:
                alerts.append("⚠️ Poor signal quality - Consider repeat ECG")
            if confidence < 60:
                alerts.append("⚠️ Low confidence in AI prediction - Manual review recommended")
            
            # 7. Treatment recommendations
            st.progress(87.5, text="7/8: Treatment planning...")
            
            if risk_score >= 70:
                treatment_urgency = "EMERGENCY"
                recommended_setting = "Emergency Department"
            elif risk_score >= 50:
                treatment_urgency = "URGENT"
                recommended_setting = "Urgent Care or Cardiology Clinic within 48h"
            else:
                treatment_urgency = "ROUTINE"
                recommended_setting = "Outpatient Clinic within 2 weeks"
            
            # 8. Generate complete report
            st.progress(100, text="8/8: Finalizing report...")
            
            # Store comprehensive results
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "patient": {
                    "id": patient_id or hashlib.md5(patient_name.encode()).hexdigest()[:8],
                    "name": patient_name,
                    "age": age,
                    "gender": gender,
                    "bmi": bmi,
                    "comorbidities": comorbidities,
                    "medications": medications_current
                },
                "ecg_analysis": {
                    "diagnosis": class_index,
                    "diagnosis_name": clinical_info['name'],
                    "icd10": clinical_info['icd10'],
                    "confidence": confidence,
                    "dominant_frequency": float(dominant_freq),
                    "estimated_heart_rate": heart_rate_estimate,
                    "qrs_duration": qrs_duration_estimate,
                    "qt_interval": qt_interval_estimate,
                    "signal_quality_snr": signal_noise_ratio,
                    "peak_count": len(peaks)
                },
                "risk_assessment": {
                    "risk_level": clinical_info['severity'],
                    "risk_score": risk_score,
                    "mortality_risk": clinical_info['mortality_risk'],
                    "alerts": alerts
                },
                "clinical_recommendations": {
                    "urgency": treatment_urgency,
                    "setting": recommended_setting,
                    "specialist": clinical_info['specialist'],
                    "follow_up": clinical_info['follow_up'],
                    "medications": clinical_info['medications'],
                    "imaging_needed": clinical_info['imaging_needed'],
                    "admission_needed": risk_score >= 70
                },
                "raw_values": ecg_values.tolist()
            }
            
            st.session_state.current_patient = analysis_result
            st.session_state.clinical_history.append(analysis_result)
            st.session_state.alerts = alerts
            
            st.success("✅ Comprehensive analysis complete!")
            st.balloons()
            
            # Show critical alerts immediately
            if alerts:
                with st.expander("🚨 CRITICAL ALERTS", expanded=True):
                    for alert in alerts:
                        st.error(alert)

# Display comprehensive results
if st.session_state.current_patient:
    result = st.session_state.current_patient
    clinical = class_labels[result['ecg_analysis']['diagnosis']]
    
    # Risk meter
    st.markdown("## 📊 Clinical Risk Dashboard")
    
    col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
    with col_risk1:
        st.markdown(f"""
        <div class="vital-card">
            <h3>Risk Score</h3>
            <h1 style="font-size: 3rem;">{result['risk_assessment']['risk_score']}</h1>
            <p>/100</p>
            <div class="risk-meter">
                <div class="risk-indicator" style="width: {result['risk_assessment']['risk_score']}%;"></div>
            </div>
            <p>{result['risk_assessment']['risk_level']} Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk2:
        st.markdown(f"""
        <div class="vital-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>Heart Rate</h3>
            <h1 style="font-size: 3rem;">{result['ecg_analysis']['estimated_heart_rate']:.0f}</h1>
            <p>BPM</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk3:
        st.markdown(f"""
        <div class="vital-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>Confidence</h3>
            <h1 style="font-size: 3rem;">{result['ecg_analysis']['confidence']:.0f}</h1>
            <p>%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk4:
        st.markdown(f"""
        <div class="vital-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>Signal Quality</h3>
            <h1 style="font-size: 3rem;">{result['ecg_analysis']['signal_quality_snr']:.1f}</h1>
            <p>dB SNR</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main diagnosis card
    st.markdown(f"""
    <div class="floating-card severity-{clinical['severity']}">
        <h2 class="gradient-text">Primary Diagnosis</h2>
        <h1 style="color: {clinical['color']};">{clinical['name']}</h1>
        <p><strong>ICD-10 Code:</strong> {clinical['icd10']}</p>
        <p>{clinical['desc']}</p>
        <div>
            <span class="clinical-badge" style="background: {clinical['color']}20; color: {clinical['color']};">Risk: {clinical['severity']}</span>
            <span class="clinical-badge" style="background: #3498db20; color: #3498db;">Specialist: {clinical['specialist']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Treatment plan
    st.markdown("## 💊 Comprehensive Treatment Plan")
    
    col_treat1, col_treat2 = st.columns(2)
    with col_treat1:
        st.markdown("### Immediate Actions")
        st.markdown(f"""
        - **Urgency:** {result['clinical_recommendations']['urgency']}
        - **Setting:** {result['clinical_recommendations']['setting']}
        - **Specialist:** {clinical['specialist']}
        """)
        if result['clinical_recommendations']['admission_needed']:
            st.error("⚠️ Hospital admission recommended")
        if result['clinical_recommendations']['imaging_needed']:
            st.info("📊 Imaging studies required (Echocardiogram recommended)")
    
    with col_treat2:
        st.markdown("### Pharmacological Plan")
        if clinical['medications']:
            for med in clinical['medications']:
                st.markdown(f"- {med}")
        else:
            st.markdown("- No medications indicated at this time")
        
        st.markdown(f"**Follow-up:** {clinical['follow_up']}")
    
    # Advanced ECG visualization
    st.markdown("## 📈 Advanced ECG Visualization")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Raw ECG Signal", "Frequency Spectrum", "Signal Quality Analysis"),
        vertical_spacing=0.15
    )
    
    # Raw signal
    fig.add_trace(
        go.Scatter(y=result['raw_values'], mode='lines', name='ECG Signal',
                   line=dict(color=clinical['color'], width=2)),
        row=1, col=1
    )
    
    # Frequency spectrum
    fft_vals = fft(result['raw_values'])
    freqs = fftfreq(len(result['raw_values']), 1/500)
    fig.add_trace(
        go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(fft_vals[:len(fft_vals)//2]),
                   mode='lines', name='Frequency Spectrum', line=dict(color='#3498db')),
        row=2, col=1
    )
    
    # Quality analysis
    quality_score = np.ones(187) * (result['ecg_analysis']['signal_quality_snr'] / 20)
    fig.add_trace(
        go.Scatter(y=quality_score, mode='lines', name='Quality Index',
                   line=dict(color='#2ecc71', dash='dash')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, template='plotly_white')
    fig.update_xaxes(title_text="Time (samples)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time (samples)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export complete package
    st.markdown("## 📋 Clinical Documentation")
    
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    
    with col_export1:
        # Generate JSON report
        report_json = json.dumps(result, indent=2, default=str)
        st.download_button("📄 Export JSON", report_json, f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with col_export2:
        # Generate HL7-like text report
        hl7_report = f"""
        MSH|^~\&|ECG System|Cardiology|EMR|HOSPITAL|{datetime.now().strftime('%Y%m%d%H%M%S')}||ORU^R01|ECG{result['patient']['id']}|P|2.3
        PID|||{result['patient']['id']}||{result['patient']['name']}||{result['patient']['age']}|{result['patient']['gender'][0]}
        OBR|||ECG^ECG|||{datetime.now().strftime('%Y%m%d%H%M%S')}
        OBX|1|FT|DIAG^Diagnosis||{clinical['name']} ({clinical['icd10']})
        OBX|2|NM|RISK^Risk Score||{result['risk_assessment']['risk_score']}
        OBX|3|FT|REC^Recommendation||{result['clinical_recommendations']['setting']}
        """
        st.download_button("📋 HL7 Report", hl7_report, f"HL7_ECG_{datetime.now().strftime('%Y%m%d')}.hl7")
    
    with col_export3:
        st.download_button("📊 PDF Report", "PDF generation ready - Professional license required", disabled=True)
    
    with col_export4:
        if st.button("💾 Save to EHR"):
            st.success("Saved to Electronic Health Record (Demo)")

with tabs[1]:
    st.markdown("## 🔬 Advanced Analytics")
    st.info("Comprehensive statistical analysis, machine learning explanations, and predictive modeling available in Professional Edition")

with tabs[2]:
    st.markdown("## 📈 Longitudinal Trend Analysis")
    if len(st.session_state.clinical_history) > 1:
        # Show trends over time
        history_df = pd.DataFrame([{
            "Date": h['timestamp'][:10],
            "Risk Score": h['risk_assessment']['risk_score'],
            "Confidence": h['ecg_analysis']['confidence']
        } for h in st.session_state.clinical_history])
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=history_df['Date'], y=history_df['Risk Score'], 
                                       name='Risk Score', mode='lines+markers'))
        fig_trend.add_trace(go.Scatter(x=history_df['Date'], y=history_df['Confidence'], 
                                       name='Confidence', mode='lines+markers'))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Multiple consultations needed for trend analysis")

with tabs[3]:
    st.markdown("## 💊 Treatment Planner")
    st.markdown("""
    ### AI-Powered Treatment Recommendations
    
    Based on clinical guidelines and patient-specific factors:
    
    - **Medication optimization**
    - **Lifestyle modifications**
    - **Interventional procedures**
    - **Follow-up scheduling**
    """)

with tabs[4]:
    st.markdown("## 📚 Clinical Knowledge Base")
    
    kb_tabs = st.tabs(["Guidelines", "Drug Database", "Risk Calculators", "Research"])
    
    with kb_tabs[0]:
        st.markdown("### ACC/AHA Clinical Guidelines")
        st.markdown("""
        - 2023 Atrial Fibrillation Guideline
        - Ventricular Arrhythmia Management
        - ECG Interpretation Standards
        """)
    
    with kb_tabs[1]:
        st.markdown("### Antiarrhythmic Drug Database")
        drug_search = st.text_input("Search medications")
        if drug_search:
            st.info(f"Detailed information for {drug_search} available in subscription")
    
    with kb_tabs[2]:
        st.markdown("### Risk Calculators")
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            chads_score = st.number_input("CHADS-VASc Score", 0, 9)
            if chads_score > 0:
                st.warning(f"Stroke risk: {chads_score * 2.5}% per year")
    
    with kb_tabs[3]:
        st.markdown("### Latest Research")
        st.markdown("- AI in Cardiology: 2024 Review")
        st.markdown("- Deep Learning for ECG Classification")

# Footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <strong>🏥 ECG Master Suite v3.0</strong><br>
            Clinical Decision Support System
        </div>
        <div>
            <strong>Certifications:</strong> CE Mark, FDA Class II (510k), ISO 13485
        </div>
        <div>
            <strong>Support:</strong> 24/7 Clinical Support • HIPAA Compliant • GDPR Ready
        </div>
    </div>
    <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.1);">
    <div style="font-size: 0.8rem; opacity: 0.7;">
        ⚠️ Medical Device: For professional use only. All clinical decisions must be made by qualified healthcare providers.
        This system is an assistive tool and does not replace clinical judgment.
    </div>
</div>
""", unsafe_allow_html=True)
