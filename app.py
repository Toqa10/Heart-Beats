import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import base64

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Advanced class labels with clinical info
class_labels = {
    0: {
        "name": "Normal Sinus Rhythm (N)", 
        "color": "#2Ecc71", 
        "severity": "Low",
        "risk_score": 10,
        "desc": "Normal heartbeat pattern within expected parameters",
        "clinical_advice": "No immediate concerns. Regular check-ups recommended.",
        "treatment": "None required. Maintain healthy lifestyle.",
        "follow_up": "Routine annual physical examination"
    },
    1: {
        "name": "Supraventricular (S)", 
        "color": "#F39C12", 
        "severity": "Moderate",
        "risk_score": 45,
        "desc": "Supraventricular premature beat originating above ventricles",
        "clinical_advice": "Monitor symptoms. Consider Holter monitoring if frequent.",
        "treatment": "Beta-blockers or calcium channel blockers if symptomatic",
        "follow_up": "Follow-up in 4-6 weeks or sooner if symptoms worsen"
    },
    2: {
        "name": "Ventricular (V)", 
        "color": "#E74C3C", 
        "severity": "High",
        "risk_score": 75,
        "desc": "Ventricular premature beat - potentially serious finding",
        "clinical_advice": "URGENT: Cardiology referral required. Risk of arrhythmias.",
        "treatment": "Antiarrhythmic medications. Possible ablation therapy.",
        "follow_up": "Immediate cardiology consultation (within 1 week)"
    },
    3: {
        "name": "Fusion (F)", 
        "color": "#9B59B6", 
        "severity": "Moderate-High",
        "risk_score": 60,
        "desc": "Fusion beat combining normal and ectopic patterns",
        "clinical_advice": "Requires further evaluation with electrophysiology study",
        "treatment": "Depends on underlying cause and frequency",
        "follow_up": "Cardiology evaluation within 2 weeks"
    },
    4: {
        "name": "Unclassified (Q)", 
        "color": "#95A5A6", 
        "severity": "Uncertain",
        "risk_score": 30,
        "desc": "Pattern doesn't match standard classifications",
        "clinical_advice": "Repeat ECG with proper lead placement recommended",
        "treatment": "Await confirmation with repeat testing",
        "follow_up": "Repeat ECG within 1 week or immediate if symptomatic"
    }
}

# Normal reference ranges for ECG metrics
NORMAL_RANGES = {
    "amplitude_min": -0.5,
    "amplitude_max": 1.5,
    "amplitude_std": (0.2, 0.8),
    "zero_crossings": (30, 60),
    "signal_symmetry": (-0.3, 0.3)
}

# Page config
st.set_page_config(
    page_title="ECG Clinical Decision Support System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical look
st.markdown("""
    <style>
    .clinical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .metric-card-clinical {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .recommendation-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .clinical-badge {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for medical history
if 'consultation_history' not in st.session_state:
    st.session_state.consultation_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}

# Sidebar - Clinical Settings
with st.sidebar:
    st.markdown("### 🏥 Clinical Settings")
    
    # Patient information
    with st.expander("👤 Patient Information", expanded=True):
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.multiselect("Presenting Symptoms", 
                                  ["Palpitations", "Chest pain", "Dizziness", "Shortness of breath", 
                                   "Fatigue", "Syncope", "Asymptomatic"])
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Settings")
    clinical_mode = st.radio("View Mode", ["Patient View", "Clinical View"], index=1)
    show_detailed_stats = st.checkbox("Show Advanced Statistics", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Quick Reference")
    st.info("""
    **Risk Levels:**
    - 🟢 Low: Routine follow-up
    - 🟡 Moderate: Monitor closely
    - 🔴 High: Urgent intervention
    
    **Confidence Interpretation:**
    - >80%: High reliability
    - 60-80%: Moderate reliability  
    - <60%: Repeat recommended
    """)

# Main header
st.markdown("""
<div class="clinical-header">
    <h1 style="margin:0;">💓 ECG Clinical Decision Support System</h1>
    <p style="margin:10px 0 0 0; opacity:0.9;">AI-Powered Cardiac Assessment & Risk Stratification</p>
</div>
""", unsafe_allow_html=True)

# Input section
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("### 📤 ECG Data Input")
    
    input_method = st.radio("Select Input Method", ["Upload CSV File", "Manual Entry", "Generate Test Signal"], horizontal=True)
    
    ecg_values = None
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload ECG CSV (187 samples)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[0] == 187 and df.shape[1] == 1:
                ecg_values = df.iloc[:, 0].values
            elif df.shape[0] == 1 and df.shape[1] == 187:
                ecg_values = df.iloc[0, :].values
            else:
                ecg_values = df.values.flatten()
            
            if len(ecg_values) != 187:
                st.error(f"Invalid: {len(ecg_values)} values (need 187)")
                ecg_values = None
            else:
                st.success(f"✅ Loaded {len(ecg_values)} ECG samples")
    
    elif input_method == "Manual Entry":
        manual_input = st.text_area("Paste 187 comma-separated values", height=100)
        if manual_input and st.button("Process Manual Input"):
            try:
                values = [float(x.strip()) for x in manual_input.replace('\n', ',').split(',') if x.strip()]
                if len(values) == 187:
                    ecg_values = np.array(values)
                    st.success("✅ Values accepted")
                else:
                    st.error(f"Need 187 values, got {len(values)}")
            except:
                st.error("Invalid format")
    
    else:  # Generate test signal
        st.markdown("**Generate Clinical Test Patterns**")
        pattern_type = st.selectbox("Test Pattern", ["Normal Sinus", "PVC Pattern", "Artifact", "Bradycardia", "Tachycardia"])
        noise = st.slider("Noise Level", 0.0, 0.5, 0.05)
        
        if st.button("Generate Signal"):
            t = np.linspace(0, 8*np.pi, 187)
            if pattern_type == "Normal Sinus":
                ecg_values = np.sin(t) * 0.8 + np.sin(3*t) * 0.2
            elif pattern_type == "PVC Pattern":
                ecg_values = np.sin(t) * 0.8
                ecg_values[80:95] = -1.2  # Simulate PVC
            elif pattern_type == "Bradycardia":
                ecg_values = np.sin(t/1.5) * 0.8
            elif pattern_type == "Tachycardia":
                ecg_values = np.sin(t*1.5) * 0.8
            else:
                ecg_values = np.random.normal(0, 0.3, 187)
            
            ecg_values += np.random.normal(0, noise, 187)
            ecg_values = ecg_values / np.max(np.abs(ecg_values))
            st.success(f"Generated {pattern_type} pattern")
    
    # Analyze button
    if ecg_values is not None and st.button("🔬 Run Clinical Analysis", type="primary", use_container_width=True):
        with st.spinner("Performing advanced clinical analysis..."):
            # Model prediction
            reshaped = ecg_values.reshape(1, 187, 1).astype(np.float32)
            prediction = model.predict(reshaped)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            
            # Advanced signal analysis
            signal_quality = 1.0 - min(1.0, np.std(ecg_values) / 1.5)
            amplitude_range = np.max(ecg_values) - np.min(ecg_values)
            zero_crossings = np.sum(np.diff(np.sign(ecg_values)) != 0)
            signal_skewness = pd.Series(ecg_values).skew()
            
            # Risk assessment
            clinical_data = class_labels[class_index]
            risk_level = clinical_data["severity"]
            risk_score = clinical_data["risk_score"]
            
            # Adjust risk based on confidence
            if confidence < 60:
                risk_score = min(100, risk_score + 20)
            
            # Generate clinical recommendations
            if risk_level == "High":
                urgency = "IMMEDIATE CARDIOLOGY REFERRAL"
                recommended_action = "Emergency department or same-day cardiology consult"
                monitoring = "Continuous ECG monitoring recommended"
            elif risk_level == "Moderate" or risk_level == "Moderate-High":
                urgency = "URGENT OUTPATIENT REFERRAL"
                recommended_action = "Schedule cardiology appointment within 1-2 weeks"
                monitoring = "Holter monitor for 24-48 hours recommended"
            else:
                urgency = "ROUTINE FOLLOW-UP"
                recommended_action = "Continue regular primary care"
                monitoring = "Annual ECG screening sufficient"
            
            # Store in session
            analysis_result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient": {"name": patient_name, "age": patient_age, "gender": patient_gender},
                "diagnosis": class_index,
                "diagnosis_name": clinical_data["name"],
                "confidence": confidence,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "signal_quality": signal_quality,
                "amplitude_range": amplitude_range,
                "zero_crossings": zero_crossings,
                "skewness": signal_skewness,
                "symptoms": symptoms,
                "urgency": urgency,
                "recommended_action": recommended_action,
                "monitoring": monitoring,
                "clinical_advice": clinical_data["clinical_advice"],
                "treatment": clinical_data["treatment"],
                "follow_up": clinical_data["follow_up"]
            }
            
            st.session_state.current_patient = analysis_result
            st.session_state.consultation_history.append(analysis_result)
            
            st.success("✅ Clinical analysis complete!")
            st.balloons()

# Display results
if st.session_state.current_patient:
    result = st.session_state.current_patient
    clinical_info = class_labels[result["diagnosis"]]
    
    with col_right:
        # Risk Assessment Card
        risk_color = {"Low": "risk-low", "Moderate": "risk-moderate", "Moderate-High": "risk-moderate", "High": "risk-high", "Uncertain": "risk-moderate"}
        st.markdown(f"""
        <div class="{risk_color[result['risk_level']]}">
            <h3>⚠️ Risk Assessment</h3>
            <h1>{result['risk_level']} Risk</h1>
            <p>Risk Score: {result['risk_score']}/100</p>
            <p>Confidence: {result['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Urgency
        st.markdown(f"""
        <div class="metric-card-clinical" style="border-left-color: #E74C3C;">
            <strong>🚨 URGENCY:</strong><br>
            {result['urgency']}
        </div>
        """, unsafe_allow_html=True)
    
    # Main results area
    st.markdown("---")
    
    # Three column layout for diagnosis
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(f"""
        <div class="metric-card-clinical" style="border-left-color: {clinical_info['color']}">
            <h3>📋 Primary Diagnosis</h3>
            <h2>{clinical_info['name']}</h2>
            <p>{clinical_info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_d2:
        st.markdown(f"""
        <div class="metric-card-clinical" style="border-left-color: #3498DB">
            <h3>💊 Treatment Plan</h3>
            <p>{clinical_info['treatment']}</p>
            <br>
            <strong>Follow-up:</strong><br>
            {clinical_info['follow_up']}
        </div>
        """, unsafe_allow_html=True)
    
    with col_d3:
        st.markdown(f"""
        <div class="metric-card-clinical" style="border-left-color: #27AE60">
            <h3>📅 Recommended Action</h3>
            <p><strong>Immediate:</strong> {result['recommended_action']}</p>
            <p><strong>Monitoring:</strong> {result['monitoring']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Signal analysis and stats
    if show_detailed_stats:
        st.markdown("### 📊 Advanced Signal Analysis")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            quality_color = "🟢" if result['signal_quality'] > 0.7 else "🟡" if result['signal_quality'] > 0.4 else "🔴"
            st.metric("Signal Quality", f"{quality_color} {result['signal_quality']*100:.1f}%")
        with col_s2:
            st.metric("Amplitude Range", f"{result['amplitude_range']:.3f}")
        with col_s3:
            st.metric("Zero Crossings", f"{result['zero_crossings']}")
        with col_s4:
            st.metric="Signal Skewness", f"{result['skewness']:.3f}"
        
        # Quality assessment
        if result['signal_quality'] < 0.4:
            st.warning("⚠️ Poor signal quality detected. Consider repeating ECG with proper lead placement.")
    
    # Clinical recommendations
    st.markdown("### 🏥 Clinical Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("#### ✅ Clinical Advice")
        st.info(result['clinical_advice'])
        
        if symptoms:
            st.markdown("#### 📝 Symptom Correlation")
            for symptom in symptoms:
                st.markdown(f"- {symptom}")
    
    with rec_col2:
        st.markdown("#### 📋 Next Steps")
        st.markdown(f"""
        1. **{result['recommended_action']}**
        2. **{result['monitoring']}**
        3. **{clinical_info['follow_up']}**
        
        ### 📞 Emergency Warning Signs
        Seek immediate care if:
        - Chest pain or pressure
        - Severe dizziness or fainting
        - Shortness of breath at rest
        - Irregular heartbeat with symptoms
        """)
    
    # Export options
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        # Generate clinical report
        report_text = f"""
        CLINICAL ECG REPORT
        ===================
        Generated: {result['timestamp']}
        
        PATIENT INFORMATION
        -------------------
        Name: {result['patient']['name'] or 'Not specified'}
        Age: {result['patient']['age']}
        Gender: {result['patient']['gender']}
        Symptoms: {', '.join(result['symptoms']) if result['symptoms'] else 'None reported'}
        
        DIAGNOSTIC RESULTS
        ------------------
        Primary Diagnosis: {clinical_info['name']}
        Description: {clinical_info['desc']}
        Confidence: {result['confidence']:.1f}%
        Risk Level: {result['risk_level']} (Score: {result['risk_score']}/100)
        
        CLINICAL RECOMMENDATIONS
        -------------------------
        Urgency: {result['urgency']}
        Immediate Action: {result['recommended_action']}
        Monitoring: {result['monitoring']}
        Treatment Plan: {clinical_info['treatment']}
        Follow-up: {clinical_info['follow_up']}
        
        SIGNAL QUALITY METRICS
        -----------------------
        Signal Quality: {result['signal_quality']*100:.1f}%
        Amplitude Range: {result['amplitude_range']:.3f}
        Zero Crossings: {result['zero_crossings']}
        
        DISCLAIMER
        ----------
        This is an AI-assisted analysis. All clinical decisions should be made by qualified healthcare professionals.
        """
        
        st.download_button("📥 Download Clinical Report", report_text, f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with col_export2:
        # Save to history
        if st.button("💾 Save to Patient History"):
            st.success("Saved to consultation history")
    
    with col_export3:
        if st.button("🔄 New Consultation"):
            st.session_state.current_patient = {}
            st.rerun()

# Patient history sidebar
if st.session_state.consultation_history:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📜 Consultation History")
        for i, consult in enumerate(reversed(st.session_state.consultation_history[-5:])):
            with st.expander(f"Consultation {consult['timestamp']}"):
                st.markdown(f"""
                **Diagnosis:** {consult['diagnosis_name']}
                **Risk:** {consult['risk_level']}
                **Confidence:** {consult['confidence']:.1f}%
                """)

# Disclaimer
st.markdown("---")
st.caption("⚠️ **Clinical Decision Support System** - This AI tool assists but does not replace clinical judgment. All diagnoses should be verified by qualified healthcare professionals.")
