
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyse de Flottation", layout="wide")
st.title("🧪 Analyse d'efficacité de la flottation à partir d'une image")

uploaded_file = st.file_uploader("📷 Charger une image de la mousse", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Image originale", use_column_width=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    st.subheader("🎨 Analyse de la couleur moyenne (HSV)")
    st.markdown(f"**H**: {mean_h:.1f}, **S**: {mean_s:.1f}, **V**: {mean_v:.1f}")

    if mean_v > 180:
        couleur_eval = "✅ Mousse claire → Flottation probablement efficace"
    elif mean_v > 130:
        couleur_eval = "🟠 Mousse intermédiaire → Flottation à surveiller"
    else:
        couleur_eval = "❌ Mousse foncée → Flottation probablement inefficace"
    st.info(couleur_eval)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [c for c in contours if cv2.contourArea(c) > 5]
    nb_bulles = len(bubbles)
    tailles = [cv2.contourArea(c) for c in bubbles]
    taille_moy = np.mean(tailles) if tailles else 0

    st.subheader("🔘 Analyse des bulles")
    st.markdown(f"**Nombre de bulles détectées**: {nb_bulles}")
    st.markdown(f"**Taille moyenne des bulles**: {taille_moy:.1f} pixels")

    if nb_bulles > 100 and taille_moy > 20:
        st.success("✅ Bonne répartition des bulles → Flottation efficace")
    elif nb_bulles > 30:
        st.warning("🟠 Répartition moyenne → À surveiller")
    else:
        st.error("❌ Peu de bulles → Flottation inefficace")

    img_contours = img_rgb.copy()
    cv2.drawContours(img_contours, bubbles, -1, (0, 255, 0), 2)

    st.subheader("📸 Visualisation des bulles détectées")
    st.image(img_contours, use_column_width=True)
