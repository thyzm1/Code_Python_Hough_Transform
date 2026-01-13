import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Exploration Hough Transform")

st.title("🔬 Dissection de la Transformée de Hough")
st.markdown("""
Cette interface permet de visualiser étape par étape l'algorithme décrit dans le rapport.
Elle implémente la logique de vote, la normalisation et la reprojection.
""")

# --- 1. CHARGEMENT ET PRÉTRAITEMENT ---
st.sidebar.header("1. Paramètres d'Entrée")
upload = st.sidebar.file_uploader("Charger une image", type=["png", "jpg", "jpeg"])

if upload is not None:
    # Lecture de l'image
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
else:
    # Image synthétique par défaut (Ligne diagonale et un point bruit)
    image_gray = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(image_gray, (50, 50), (150, 150), 255, 2)
    cv2.line(image_gray, (20, 150), (100, 150), 255, 2) # Ligne horizontale
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

# Détection de contours (Canny)
st.sidebar.subheader("Détection de Contours")
th1 = st.sidebar.slider("Seuil Canny 1", 50, 200, 50)
th2 = st.sidebar.slider("Seuil Canny 2", 100, 300, 150)
edges = cv2.Canny(image_gray, th1, th2)

col1, col2 = st.columns(2)
with col1:
    st.image(image_gray, caption="Image Originale (Greyscale)", use_container_width=True)
with col2:
    st.image(edges, caption="Image Binaire (Contours)", use_container_width=True)

# --- 2. CALCUL DE L'ACCUMULATEUR (Cœur de l'algo) ---
st.markdown("---")
st.header("2. L'Espace de Hough (Accumulateur)")

# Paramètres de résolution
st.sidebar.subheader("Paramètres Hough")
step_theta = st.sidebar.slider("Pas de Theta (degrés)", 0.5, 5.0, 1.0)
use_strict_implementation = st.sidebar.checkbox("Utiliser la contrainte stricte du rapport (Rho >= 0)", value=True)

@st.cache_data
def compute_hough_accumulator(edge_image, step_theta_deg, strict_mode):
    """
    Implémentation Python de l'algorithme décrit en pseudo-code/C.
    Note: Optimisé avec NumPy pour éviter que l'interface ne gèle, 
    mais respecte strictement la logique mathématique décrite.
    """
    H, W = edge_image.shape
    # Diagonale pour la taille maximale de rho
    diag_len = int(np.ceil(np.sqrt(H**2 + W**2)))
    
    # Définition des axes
    thetas_deg = np.arange(0, 360, step_theta_deg) # 0 à 359
    thetas_rad = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)
    
    # Nombre de rhos
    # Si mode strict (comme le code C fourni), rho va de 0 à W (ou Diag)
    # Si mode robuste, rho va de -Diag à +Diag pour capturer toutes les droites
    if strict_mode:
        rhos_range = diag_len # On suppose une taille arbitraire suffisante positive
        offset_rho = 0
    else:
        rhos_range = 2 * diag_len
        offset_rho = diag_len # Pour centrer 0 au milieu de l'array

    # Initialisation Accumulateur (Lignes = Rho, Colonnes = Theta)
    accumulator = np.zeros((rhos_range, len(thetas_deg)), dtype=np.uint64)
    
    # Récupération des points de contour (y, x) car numpy est (row, col)
    y_idxs, x_idxs = np.nonzero(edge_image) 
    
    # Vote (Version vectorisée des boucles imbriquées du rapport)
    # Pour chaque point de contour...
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # Calculer rho pour tous les thétas d'un coup
        # Formule du rapport : rho = x * cos(theta) + y * sin(theta)
        rho_vals = x * cos_t + y * sin_t
        
        # Remplissage
        for t_idx, rho in enumerate(rho_vals):
            rho_int = int(round(rho)) + offset_rho
            
            # Condition du code C : if (ro >= 0 && ro < NCOL)
            if 0 <= rho_int < rhos_range:
                accumulator[rho_int, t_idx] += 1
                
    return accumulator, thetas_deg, diag_len, offset_rho

# Exécution
accumulator, thetas, diag_len, offset = compute_hough_accumulator(edges, step_theta, use_strict_implementation)

# Normalisation pour affichage (Algorithme "Normalisation" du rapport)
max_vote = np.max(accumulator)
if max_vote > 0:
    acc_vis = (accumulator / max_vote) * 255.0
else:
    acc_vis = accumulator
acc_vis = acc_vis.astype(np.uint8)

# Affichage interactif avec Matplotlib (plus précis pour les axes)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(acc_vis, cmap='jet', aspect='auto', extent=[thetas[0], thetas[-1], accumulator.shape[0]-offset, -offset])
ax.set_title("Visualisation de l'Accumulateur (Heatmap)")
ax.set_xlabel("Theta (degrés)")
ax.set_ylabel("Rho (pixels)")
st.pyplot(fig)

st.info(f"Max de votes dans l'accumulateur : {max_vote} votes (intersection de sinusoïdes).")

# --- 3. COMPRENDRE LE VOTE (INTERACTIF) ---
st.markdown("---")
st.subheader("3. Comprendre le vote : Dualité Point-Sinusoïde")
st.markdown("Sélectionnez un point (X, Y) ci-dessous pour voir sa courbe sinusoïdale correspondante dans l'espace de Hough.")

col_inter_1, col_inter_2 = st.columns([1, 2])

with col_inter_1:
    # Sliders pour simuler la sélection d'un point
    x_sel = st.number_input("Coordonnée X du point", 0, edges.shape[1]-1, value=50)
    y_sel = st.number_input("Coordonnée Y du point", 0, edges.shape[0]-1, value=50)
    
    # Création d'une image juste pour montrer le point sélectionné
    img_pt = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
    cv2.circle(img_pt, (x_sel, y_sel), 5, (0, 0, 255), -1)
    st.image(img_pt, caption="Point sélectionné (Rouge)", use_container_width=True)

with col_inter_2:
    # Calcul de la courbe pour ce point spécifique
    # rho = x cos(t) + y sin(t)
    thetas_rad_plot = np.deg2rad(thetas)
    rhos_plot = x_sel * np.cos(thetas_rad_plot) + y_sel * np.sin(thetas_rad_plot)
    
    fig_sin, ax_sin = plt.subplots(figsize=(10, 4))
    ax_sin.plot(thetas, rhos_plot, color='red', linewidth=2)
    ax_sin.set_title(f"Courbe de vote pour le point ({x_sel}, {y_sel})")
    ax_sin.set_xlabel("Theta (degrés)")
    ax_sin.set_ylabel("Rho")
    ax_sin.grid(True)
    st.pyplot(fig_sin)
    st.caption("Dans l'accumulateur, chaque point blanc de l'image de gauche 'dessine' cette courbe. Là où les courbes se croisent, il y a un pic (une droite).")

# --- 4. RECONSTRUCTION ET TRACÉ (L'étape finale) ---
st.markdown("---")
st.header("4. Extraction et Tracé des Droites")

threshold_vote = st.sidebar.slider("Seuil de vote (Minimum)", 10, int(max_vote) if max_vote > 0 else 100, int(max_vote * 0.5))

# Récupération des pics > Seuil
# np.where retourne les indices (row, col) -> (rho_idx, theta_idx)
rho_idxs, theta_idxs = np.where(accumulator > threshold_vote)

res_image = image_bgr.copy()

st.write(f"Nombre de droites détectées : {len(rho_idxs)}")

# Algorithme de tracé (Traduction du pseudo-code "Algorithme de dessin de droite")
for i in range(len(rho_idxs)):
    r_idx = rho_idxs[i]
    t_idx = theta_idxs[i]
    
    rho = r_idx - offset # On rétablit le vrai rho (négatif possible si mode robuste)
    theta_deg = thetas[t_idx]
    theta_rad = np.deg2rad(theta_deg)
    
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    
    x0 = a * rho
    y0 = b * rho
    
    # On génère deux points très éloignés pour tracer la ligne sur toute l'image
    # x1 = x0 + 1000 * (-b)
    # y1 = y0 + 1000 * (a)
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    
    cv2.line(res_image, pt1, pt2, (0, 0, 255), 2)

col_final_1, col_final_2 = st.columns(2)
with col_final_1:
    st.image(acc_vis, caption="Accumulateur Normalisé", use_container_width=True)
with col_final_2:
    st.image(res_image, caption="Résultat : Droites détectées (Rouge)", use_container_width=True)

# --- 5. CODE RAW (Pour référence) ---
with st.expander("Voir le code Python brut de la boucle de remplissage"):
    st.code("""
    # Traduction littérale des boucles
    for x in range(n_lignes):
        for y in range(n_cols):
            if image[x,y] > 0:
                for theta_deg in range(0, 360):
                    theta = theta_deg * PI / 180
                    rho = x * cos(theta) + y * sin(theta)
                    # Note: offset nécessaire en Python pour indices négatifs
                    rho_idx = int(rho) + offset 
                    if 0 <= rho_idx < max_rho:
                        acc[rho_idx, theta_deg] += 1
    """, language='python')