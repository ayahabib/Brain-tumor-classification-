# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid
import cv2
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Brain Tumor Detector", layout="wide", page_icon="🧠")


MODEL_PATH = "best_model.keras"
IMAGE_SIZE = (224, 224)
CLASS_TYPES = ['BENIGN', 'MALIGNANT', 'NORMAL']

# Create outputs folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)



@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model at '{MODEL_PATH}': {e}")
    st.stop()


def preprocess_pil(pil_img, target_size=IMAGE_SIZE):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is not None and isinstance(out_shape, (tuple, list)) and len(out_shape) == 4:
            return layer.name
    for layer in reversed(model.layers):
        if "conv" in layer.__class__.__name__.lower() or "conv" in layer.name:
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def make_gradcam(img_array, model, last_conv_layer_name=None, eps=1e-8):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max() if heatmap.max() != 0 else eps
    heatmap /= max_val
    return heatmap  # values 0..1


def overlay_heatmap_on_pil(pil_img, heatmap, alpha=0.4, resize_to=IMAGE_SIZE):
    orig = np.array(pil_img.convert("RGB").resize(resize_to))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (orig.shape[1], orig.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)
    # combine side-by-side for visibility
    combined = np.hstack([orig, overlay])
    return Image.fromarray(combined)


def make_report_pdf(pred_class, confidence, overlay_pil, lang="en"):
    pdf = FPDF()
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_font("Arial", 'B', 16)
    if lang == "en":
        pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align='C')
    else:
        pdf.cell(200, 10, txt="تقرير كشف أورام الدماغ", ln=True, align='C')
    pdf.ln(10)

    # Date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", size=10)
    if lang == "en":
        pdf.cell(200, 10, txt=f"Report Date: {now}", ln=True)
    else:
        pdf.cell(200, 10, txt=f"تاريخ التقرير: {now}", ln=True)
    pdf.ln(5)

    # Prediction
    pdf.set_font("Arial", 'B', 12)
    conf_text = f"{confidence:.2f}%"
    if lang == "en":
        pdf.cell(200, 10, txt=f"Prediction: {pred_class}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {conf_text}", ln=True)
    else:
        pdf.cell(200, 10, txt=f"التصنيف: {pred_class}", ln=True)
        pdf.cell(200, 10, txt=f"نسبة الثقة: {conf_text}", ln=True)
    pdf.ln(10)

    # Recommendation
    pdf.set_font("Arial", size=12)
    if lang == "en":
        if pred_class == "MALIGNANT":
            rec = "This MRI scan indicates a malignant tumor. Immediate medical evaluation is advised."
        elif pred_class == "BENIGN":
            rec = "This MRI scan suggests a benign tumor. Follow-up imaging is recommended."
        else:
            rec = "This MRI scan appears normal with no tumor detected."
        pdf.multi_cell(0, 10, txt=f"Recommendation: {rec}")
    else:
        if pred_class == "MALIGNANT":
            rec = "تشير الصورة إلى وجود ورم خبيث. يُنصح بمراجعة الطبيب فورًا."
        elif pred_class == "BENIGN":
            rec = "تشير الصورة إلى وجود ورم حميد. يفضّل إجراء متابعة طبية دورية."
        else:
            rec = "الصورة تبدو سليمة ولا يظهر وجود ورم."
        pdf.multi_cell(0, 10, txt=f"التوصية: {rec}")
    pdf.ln(10)

    # Disclaimer
    pdf.set_font("Arial", 'I', 10)
    if lang == "en":
        pdf.multi_cell(0, 10,
                       txt="Disclaimer: This report is generated by an AI model and is for informational purposes only. It is not a substitute for professional medical advice. Please consult a qualified healthcare provider for diagnosis and treatment.")
    else:
        pdf.multi_cell(0, 10,
                       txt="تنويه: هذا التقرير تم إنشاؤه بواسطة نموذج ذكاء اصطناعي وهو لأغراض معلوماتية فقط. لا يُعتبر بديلاً عن الاستشارة الطبية المهنية. يرجى استشارة مقدم رعاية صحية مؤهل للتشخيص والعلاج.")
    pdf.ln(10)

    # Add overlay image
    overlay_bytes = io.BytesIO()
    overlay_pil.save(overlay_bytes, format="PNG")
    overlay_bytes.seek(0)
    pdf.image(overlay_bytes, x=10, y=None, w=180)
    pdf.ln(5)
    if lang == "en":
        pdf.cell(200, 10, txt="Grad-CAM Overlay: Highlights regions influencing the decision.", ln=True, align='C')
    else:
        pdf.cell(200, 10, txt="تراكب Grad-CAM: يبرز المناطق التي أثرت على القرار.", ln=True, align='C')

    # Output to bytes
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output


# -------------------------
# UI
# -------------------------
st.title("🧠 Brain Tumor Detector")
st.markdown(
    "Upload an MRI image (JPG/PNG) to detect brain tumors. The app provides prediction, confidence, and a Grad-CAM heatmap for explainability.")

with st.sidebar:
    st.header("Model Information")
    st.write(f"**Input Size:** {IMAGE_SIZE}")
    st.write(f"**Classes:** {', '.join(CLASS_TYPES)}")
    st.write("**Note:** This is an AI-assisted tool. Always consult a medical professional for accurate diagnosis.")

    lang = st.selectbox("Select Language / اختر اللغة", ["English", "العربية"])
    lang_code = "ar" if lang == "العربية" else "en"

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Choose an MRI image / اختر صورة MRI", type=["png", "jpg", "jpeg"])
    run_predict = st.button("Predict / تنبؤ", disabled=uploaded is None)

with col2:
    if uploaded is not None:
        st.info("Image uploaded successfully. Click 'Predict' to analyze.")
    else:
        st.info("Please upload an image to proceed.")

# state containers
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# Predict and produce outputs
if run_predict and uploaded is not None:
    with st.spinner("Analyzing image... / جاري التحليل..."):
        try:
            pil_img = Image.open(uploaded)
            arr = preprocess_pil(pil_img, IMAGE_SIZE)
            preds = model.predict(arr)[0]
            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_TYPES[pred_idx]
            confidence = float(preds[pred_idx] * 100.0)

            # Grad-CAM
            heatmap = make_gradcam(arr, model)
            overlay = overlay_heatmap_on_pil(pil_img, heatmap, alpha=0.4, resize_to=IMAGE_SIZE)

            # prepare files for download
            uid = uuid.uuid4().hex[:8]
            base = f"{uid}"
            orig_bytes = io.BytesIO()
            pil_img.convert("RGB").save(orig_bytes, format="PNG")
            orig_bytes.seek(0)

            overlay_bytes = io.BytesIO()
            overlay.save(overlay_bytes, format="PNG")
            overlay_bytes.seek(0)

            # Generate PDF report
            report_pdf = make_report_pdf(pred_class, confidence, overlay, lang=lang_code)

            # store in session state
            st.session_state["last_result"] = {
                "pred_class": pred_class,
                "confidence": confidence,
                "orig_pil": pil_img,  # Store PIL for display
                "overlay_pil": overlay,  # Store PIL for display
                "orig_bytes": orig_bytes,
                "overlay_bytes": overlay_bytes,
                "report_pdf": report_pdf
            }

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

# show results if available
if st.session_state["last_result"] is not None:
    res = st.session_state["last_result"]

    # Display prediction
    if lang_code == "en":
        st.success(f"**Prediction:** {res['pred_class']} — **Confidence:** {res['confidence']:.2f}%")
    else:
        st.success(f"**التصنيف:** {res['pred_class']} — **نسبة الثقة:** {res['confidence']:.2f}%")

    # Tabs for results
    tab1, tab2, tab3 = st.tabs(["Images / الصور", "Report / التقرير", "Downloads / التحميلات"])

    with tab1:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(res["orig_pil"], caption="Uploaded Image / الصورة المرفوعة", width=300)
        with col_img2:
            st.image(res["overlay_pil"],
                     caption="Grad-CAM Overlay (Original | Heatmap) / تراكب Grad-CAM (الأصلي | الخريطة الحرارية)",
                     width=300)
        st.caption(
            "Grad-CAM highlights regions that influenced the classifier's decision — not a precise segmentation. Always consult clinicians for diagnosis.")

    with tab2:
        # Display PDF preview or text
        st.markdown("**PDF Report Preview:**")
        # Since Streamlit can't directly display PDF, show a download link or embed if possible
        # For simplicity, show text version
        if lang_code == "en":
            st.text_area("Report Preview / معاينة التقرير",
                         f"Prediction: {res['pred_class']}\nConfidence: {res['confidence']:.2f}%\n\nRecommendation: {'Immediate medical evaluation advised.' if res['pred_class'] == 'MALIGNANT' else 'Follow-up imaging recommended.' if res['pred_class'] == 'BENIGN' else 'No tumor detected.'}",
                         height=150)
        else:
            st.text_area("معاينة التقرير / Report Preview",
                         f"التصنيف: {res['pred_class']}\nنسبة الثقة: {res['confidence']:.2f}%\n\nالتوصية: {'يُنصح بمراجعة الطبيب فورًا.' if res['pred_class'] == 'MALIGNANT' else 'يفضّل إجراء متابعة طبية دورية.' if res['pred_class'] == 'BENIGN' else 'لا يظهر وجود ورم.'}",
                         height=150)

    with tab3:
        # Overlay Image download
        if st.button("Prepare Overlay Image for Download / تجهيز صورة التراكب"):
            overlay_bytes = io.BytesIO()
            res["overlay_pil"].save(overlay_bytes, format="PNG")
            overlay_bytes.seek(0)
            st.session_state["overlay_download"] = overlay_bytes
            st.success("Overlay Image Ready!")

        if "overlay_download" in st.session_state:
            st.download_button(
                label="Download Overlay Image (PNG) / تحميل صورة التراكب (PNG)",
                data=st.session_state["overlay_download"].getvalue(),
                file_name=f"gradcam_{res['pred_class']}.png",
                mime="image/png"
            )

        # PDF Report download
        if st.button("Prepare PDF Report for Download / تجهيز التقرير PDF"):
            report_pdf = make_report_pdf(res['pred_class'], res['confidence'], res['overlay_pil'], lang=lang_code)
            st.session_state["pdf_download"] = report_pdf
            st.success("PDF Report Ready!")

        if "pdf_download" in st.session_state:
            st.download_button(
                label="Download Report (PDF) / تحميل التقرير (PDF)",
                data=st.session_state["pdf_download"].getvalue(),
                file_name=f"report_{res['pred_class']}.pdf",
                mime="application/pdf"
            )

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice.")
