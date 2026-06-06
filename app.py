import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Class labels with descriptions
class_labels = {
    0: {"name": "Normal (N)", "color": "#2Ecc71", "desc": "إيقاع قلب طبيعي", "advice": "لا توجد علامات خطورة، يفضل متابعة نمط حياة صحي"},
    1: {"name": "Supraventricular (S)", "color": "#F39C12", "desc": "انقباض فوق بطيني", "advice": "راجع طبيب القلب، قد تحتاج لمراقبة أو علاج بسيط"},
    2: {"name": "Ventricular (V)", "color": "#E74C3C", "desc": "انقباض بطيني", "advice": "استشارة عاجلة لطبيب القلب، متابعة دقيقة مطلوبة"},
    3: {"name": "Fusion (F)", "color": "#9B59B6", "desc": "انقباض مدمج", "advice": "مراجعة طبية لتحديد السبب والعلاج المناسب"},
    4: {"name": "Unknown (Q)", "color": "#95A5A6", "desc": "نمط غير معروف", "advice": "يوصى بإعادة الفحص أو استشارة أخصائي"}
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
        
        /* Confidence bar */
        .confidence-bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
            transition: width 0.5s ease;
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
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }
        
        /* Button hover */
            .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
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
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #888;
            font-size: 0.9rem;
        }
        
        /* Dark mode styles */
        .dark-mode {
            background-color: #1a1a2e;
            color: #eee;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    
    # Theme toggle
    theme = st.selectbox("🎨 المظهر", ["فاتح", "داكن"], index=0)
    
    # Model info
    st.markdown("---")
    st.markdown("### 📊 معلومات النموذج")
    st.info("""
    - **النوع**: CNN (التفاف عصبي)
    - **المدخلات**: 187 قيمة ECG
    - **الدقة**: >95%
    """)
    
    # About
    st.markdown("---")
    st.markdown("### 💡 عن التطبيق")
    st.markdown("""
    هذا التطبيق يستخدم الذكاء الاصطناعي لتصنيف نبضات القلب 
    من بيانات تخطيط القلب (ECG) إلى 5 فئات مختلفة.
    """)

# Title
st.markdown('<div class="gradient-title">💓 Heartbeat Diagnosis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">تشخيص ذكي بنبضات القلب باستخدام الذكاء الاصطناعي</div>', unsafe_allow_html=True)

# Two columns for main content
col1, col2 = st.columns([2, 1])

with col1:
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["📁 رفع ملف CSV", "✍️ إدخال يدوي", "🎛️ إنشاء بيانات تجريبية"])
    
    with tab1:
        uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"], help="يجب أن يحتوي الملف على 187 قيمة رقمية")
        
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
                    st.error(f"⚠️ الملف يحتوي على {len(values)} قيمة، ولكن يجب أن يكون 187 قيمة بالضبط")
                else:
                    st.success(f"✅ تم رفع {len(values)} قيمة بنجاح")
                    
                    # Preview data
                    with st.expander("📊 معاينة البيانات"):
                        st.write("أول 10 قيم:")
                        st.write(values[:10])
                        st.write(f"**المدى**: {values.min():.3f} - {values.max():.3f}")
                        st.write(f"**المتوسط**: {values.mean():.3f}")
                        st.write(f"**الانحراف المعياري**: {values.std():.3f}")
                    
                    if st.button("🔍 تحليل ECG", use_container_width=True):
                        with st.spinner("جاري تحليل النبضات..."):
                            reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                            prediction = model.predict(reshaped)
                            class_index = int(np.argmax(prediction))
                            confidence = float(np.max(prediction)) * 100
                            
                            # Store in session state
                            st.session_state['prediction'] = class_index
                            st.session_state['confidence'] = confidence
                            st.session_state['values'] = values
                            
                            st.success("تم التحليل بنجاح!")
                            
            except Exception as e:
                st.error(f"❌ خطأ في قراءة الملف: {e}")
    
    with tab2:
        st.markdown("أدخل 187 قيمة مفصولة بفواصل أو مسافات:")
        user_input = st.text_area("القيم:", height=150, 
                                 placeholder="مثال: 0.5, 0.7, 0.3, ...")
        
        if st.button("🔍 تحليل الإدخال", use_container_width=True):
            try:
                cleaned = user_input.strip().replace("\t", ",").replace("\n", ",").replace(" ", ",")
                values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
                
                if len(values) != 187:
                    st.error(f"⚠️ أدخلت {len(values)} قيمة، ولكن يجب أن تكون 187 قيمة بالضبط")
                else:
                    with st.spinner("جاري التحليل..."):
                        reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                        prediction = model.predict(reshaped)
                        class_index = int(np.argmax(prediction))
                        confidence = float(np.max(prediction)) * 100
                        
                        st.session_state['prediction'] = class_index
                        st.session_state['confidence'] = confidence
                        st.session_state['values'] = values
                        
                        st.success("تم التحليل بنجاح!")
            except Exception as e:
                st.error(f"❌ خطأ: {e}")
    
    with tab3:
        st.markdown("### إنشاء بيانات ECG تجريبية")
        
        col_noise, col_generate = st.columns([2, 1])
        with col_noise:
            noise_level = st.slider("مستوى التشويش:", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🎲 إنشاء نبضة عشوائية", use_container_width=True):
            # Generate synthetic ECG-like signal
            t = np.linspace(0, 4*np.pi, 187)
            synthetic = np.sin(t) + 0.3*np.sin(3*t) + np.random.normal(0, noise_level, 187)
            synthetic = synthetic / np.max(np.abs(synthetic))
            
            st.session_state['values'] = synthetic
            st.success("تم إنشاء بيانات تجريبية!")
            
            with st.expander("📊 معاينة البيانات المُنشأة"):
                st.write(f"**المدى**: {synthetic.min():.3f} - {synthetic.max():.3f}")
                st.write(f"**المتوسط**: {synthetic.mean():.3f}")
            
            if st.button("🔍 تحليل البيانات المُنشأة", use_container_width=True):
                with st.spinner("جاري التحليل..."):
                    reshaped = np.array(synthetic, dtype=np.float32).reshape(1, 187, 1)
                    prediction = model.predict(reshaped)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction)) * 100
                    
                    st.session_state['prediction'] = class_index
                    st.session_state['confidence'] = confidence
                    
                    st.success("تم التحليل بنجاح!")

with col2:
    # Display results if available
    if 'prediction' in st.session_state:
        pred_idx = st.session_state['prediction']
        confidence = st.session_state['confidence']
        
        # Prediction Card
        st.markdown(f"""
        <div class="prediction-card" style="border-left-color: {class_labels[pred_idx]['color']}">
            <h3 style="margin:0; color:{class_labels[pred_idx]['color']}">التشخيص:</h3>
            <h2 style="margin:10px 0;">{class_labels[pred_idx]['name']}</h2>
            <p style="color:#666;">{class_labels[pred_idx]['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown("### 📊 نسبة الثقة")
        st.progress(confidence/100)
        st.markdown(f"<center><b>{confidence:.1f}%</b></center>", unsafe_allow_html=True)
        
        # Color-coded confidence
        if confidence >= 80:
            st.success("✅ ثقة عالية جداً")
        elif confidence >= 60:
            st.warning("⚠️ ثقة متوسطة")
        else:
            st.error("❌ ثقة منخفضة - يوصى بإعادة الفحص")
        
        # Medical advice
        st.markdown("### 💊 النصيحة الطبية")
        st.info(class_labels[pred_idx]['advice'])
        
        # Plot ECG signal
        if 'values' in st.session_state:
            st.markdown("### 📈 شكل الإشارة")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state['values'],
                mode='lines',
                line=dict(color=class_labels[pred_idx]['color'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(class_labels[pred_idx]['color'][1:3],16)},{int(class_labels[pred_idx]['color'][3:5],16)},{int(class_labels[pred_idx]['color'][5:7],16)},0.2)"
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="الوقت (نقطة)",
                yaxis_title="السعة",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        if st.button("📥 تحميل نتيجة التشخيص", use_container_width=True):
            report_df = pd.DataFrame({
                'الخاصية': ['التشخيص', 'الثقة', 'النصيحة'],
                'القيمة': [class_labels[pred_idx]['name'], f"{confidence:.2f}%", class_labels[pred_idx]['advice']]
            })
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="💾 تحميل كـ CSV",
                data=csv,
                file_name="ecg_diagnosis_report.csv",
                mime="text/csv"
            )
    else:
        st.info("👈 قم بتحميل ملف أو إدخال بيانات لبدء التحليل")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🤖 Powered by Deep Learning CNN | 📊 دقة النموذج >95% | 💡 للاستخدام الطبي المرجعي فقط</p>
    <p>© 2024 Heartbeat Diagnosis AI - تم التطوير باستخدام TensorFlow و Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Additional features
with st.expander("ℹ️ معلومات إضافية"):
    st.markdown("""
    ### كيف يعمل التطبيق؟
    1. يقوم النموذج بتحليل 187 نقطة زمنية من إشارة ECG
    2. يستخدم شبكة CNN (Convolutional Neural Network) للتعرف على الأنماط
    3. يصنف النبضة إلى واحدة من 5 فئات:
       - **N**: نبضة طبيعية
       - **S**: نبضة فوق بطينية
       - **V**: نبضة بطينية
       - **F**: نبضة مدمجة
       - **Q**: نبضة غير معروفة
    
    ### نصائح للاستخدام:
    - تأكد من أن ملف CSV يحتوي على 187 قيمة رقمية بالضبط
    - القيم يجب أن تكون طبيعية (Normalized) بين -1 و 1 تقريباً
    - النموذج مصمم للاستخدام التشخيصي المساعد فقط
    """)

# Run with: streamlit run app.py
