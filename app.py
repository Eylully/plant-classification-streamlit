import streamlit as st
from PIL import Image, ImageOps  
import numpy as np
import joblib
import urllib.request
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
import requests
from io import BytesIO


st.set_page_config(
    page_title="Bitki Sınıflandırma",
    layout="wide"
)


IMG_SIZE = (96, 96)
SELECTED_CLASSES = ['aloevera', 'kale', 'corn', 'peperchili', 'curcuma']
CLASS_INFO = {
    'aloevera': 'Aloe Vera',
    'kale': 'Kale (Su ıspanağı)',
    'corn': 'Mısır',
    'peperchili': 'Acı Biber',
    'curcuma': 'Zerdeçal'
}

# Model ve Veri Yükleme 
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

@st.cache_data
def load_test_data():
    if os.path.exists("X_test.npy") and os.path.exists("y_test.npy"):
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
        return X_test, y_test
    return None, None

# --- Özellik Çıkarma Fonksiyonu ---
def extract_features_optimized(image):
    try:
        img = image.convert("RGB").resize(IMG_SIZE)
        img_np = np.array(img, dtype=np.float32) / 255.0

        hist_features = []
        for ch in range(3):
            h, _ = np.histogram(img_np[:, :, ch], bins=12, range=(0,1))
            hist_features.extend(h / (h.sum() + 1e-8))

        img_hsv = np.array(img.convert("HSV")) / 255.0
        for ch in range(3):
            h, _ = np.histogram(img_hsv[:, :, ch], bins=8, range=(0,1))
            hist_features.extend(h / (h.sum() + 1e-8))

        gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
        hog_features = hog(gray, orientations=12, pixels_per_cell=(16,16),
                           cells_per_block=(2,2), feature_vector=True, channel_axis=None)

        color_stats = []
        for ch in range(3):
            ch_data = img_np[:, :, ch].flatten()
            color_stats.extend([ch_data.mean(), ch_data.std(),
                                np.median(ch_data), np.percentile(ch_data,25), np.percentile(ch_data,75)])

        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        texture_features = [grad_x.mean(), grad_x.std(), grad_y.mean(), grad_y.std()]

        return np.concatenate([hist_features, hog_features, color_stats, texture_features])
    except Exception as e:
        st.error(f"Özellik çıkarma hatası: {e}")
        return None

model = load_model()
X_test, y_test = load_test_data()

if model is None:
    st.error("'model.pkl' dosyası bulunamadı! Lütfen proje klasörüne ekleyin.")
    st.stop()

# Sidebar
st.sidebar.title("🌿 Menü")
page = st.sidebar.radio(
    "Sayfa Seçin:",
    ["Ana Sayfa", "Tahmin Yap", "Model Performansı"]
)

#  ANA SAYFA
if page == "Ana Sayfa":
    st.title("Bitki Sınıflandırma")
    st.subheader("Makine Öğrenmesi - Random Forest")
    
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Tipi", "Random Forest") 
    with col2:
        st.metric("Sınıf Sayısı", len(SELECTED_CLASSES))
    with col3:
        st.metric("Test Verisi Sayısı", len(y_test) if y_test is not None else 0)
    
    st.write("---")

    st.subheader("Veri Seti Bilgisi")
    st.write("**Kaynak:** Kaggle - Plants Classification Dataset")
    st.write("**Link:** https://www.kaggle.com/datasets/marquis03/plants-classification/data")
    
    st.write("---")
    
    st.subheader(" Seçilen Bitki Türleri")
    
    satir1 = st.columns(3)
    
    satir2 = st.columns(3)
    
    tum_kolonlar = satir1 + [satir2[0], satir2[1]]
    
    HEDEF_BOYUT = (100, 100)

    for i, cls in enumerate(SELECTED_CLASSES):
        with tum_kolonlar[i]:
            with st.container(border=True):
                
                img_path_jpg = os.path.join("resimler", f"{cls}.jpg")
                
                final_img_path = None
                if os.path.exists(img_path_jpg):
                    final_img_path = img_path_jpg
                
                if final_img_path:
                    img = Image.open(final_img_path)
                    
                    img_resized = ImageOps.fit(img, HEDEF_BOYUT, Image.Resampling.LANCZOS)
                    
                    st.image(img_resized, use_container_width=True)
                else:
                    st.warning(f"Görsel bulunamadı: resimler/{cls}.jpg")
               
                bitki_adi = CLASS_INFO[cls]
                st.markdown(f"<h4 style='text-align: center; margin-top: 10px;'>{bitki_adi}</h4>", unsafe_allow_html=True)
    
    st.write("---")

    st.subheader("Veri Seti İstatistikleri")
    
    class_counts = {
        'aloevera': 700,
        'kale': 700,
        'corn': 700,
        'peperchili': 700,
        'curcuma': 700
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Eğitim Seti Dağılımı")
        df_train = pd.DataFrame({
            'Bitki Türü': [CLASS_INFO[c] for c in class_counts.keys()],
            'Görsel Sayısı': list(class_counts.values())
        })
        st.dataframe(df_train, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Test Seti Dağılımı")
        if y_test is not None:
            test_counts = pd.Series(y_test).value_counts().sort_index()
            df_test = pd.DataFrame({
                'Bitki Türü': [CLASS_INFO[SELECTED_CLASSES[i]] for i in test_counts.index],
                'Görsel Sayısı': test_counts.values
            })
            st.dataframe(df_test, hide_index=True, use_container_width=True)
        else:
            st.warning("Test verisi bulunamadı.")
    
    st.write("---")
    
    st.subheader("Detaylar")
    
    col_tek1, col_tek2 = st.columns(2)
    
    with col_tek1:
        st.markdown("**Kullanılan Özellikler**")
        st.write("• **Renk Histogramları** (RGB + HSV)")
        st.write("• **HOG** (Histogram of Oriented Gradients)")
        st.write("• **Doku Özellikleri** (Gradyan)")
    
    with col_tek2:
        st.markdown("**Model Parametreleri**")
        st.write("• **Algoritma:** Random Forest Classifier")
        st.write("• **Ağaç Sayısı:** 200")
        st.write("• **Görsel Boyutu:** 96x96 piksel")

# TAHMİN YAP
elif page == "Tahmin Yap":
    st.title("Bitki Türü Tahmini")
    
    # Toggle (Anahtar) 
    st.write("Yükleme Yöntemi:")
    url_mode = st.toggle("İnternetten URL ile yüklemek için tıklayın")
    
    if url_mode:
        st.info("İnternetten bir görsel linki yapıştırın.")
        url_input = st.text_input("URL' yi giriniz:", placeholder="https://...")
        
        if url_input and st.button("Tahmin Yap (URL)", key="predict_url"):
            try:
                response = requests.get(url_input, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(img, caption="URL Görseli", use_container_width=True)
                    with col2:
                        features = extract_features_optimized(img)
                        if features is not None:
                            features = features.reshape(1, -1)
                            pred_idx = model.predict(features)[0]
                            pred_class = SELECTED_CLASSES[pred_idx]
                            probabilities = model.predict_proba(features)[0]
                            
                            st.success(f"### Tahmin: **{CLASS_INFO[pred_class]}**")
                            
                            # Grafik
                            st.write("---")
                            st.write("#### Olasılık Dağılımı")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['#2E7D32' if c == pred_class else '#81C784' for c in SELECTED_CLASSES]
                            ax.barh([CLASS_INFO[c] for c in SELECTED_CLASSES], probabilities * 100, color=colors)
                            ax.set_xlabel('Olasılık (%)')
                            ax.set_xlim(0, 100)
                            st.pyplot(fig)
                else:
                    st.error("Resim indirilemedi.")
            except Exception as e:
                st.error(f"Hata: {e}")
                
    else:
        #  DOSYA YÜKLEME 
        uploaded_file = st.file_uploader("Bilgisayardan bir bitki görseli seçin", type=["jpg","jpeg","png"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="Yüklenen Resim Dosyası", use_container_width=True)
            
            with col2:
                if st.button("Tahmin Yap", key="predict_file"):
                    with st.spinner("Analiz ediliyor..."):
                        features = extract_features_optimized(img)
                        
                        if features is not None:
                            features = features.reshape(1, -1)
                            pred_idx = model.predict(features)[0]
                            pred_class = SELECTED_CLASSES[pred_idx]
                            probabilities = model.predict_proba(features)[0]
                            
                            st.success(f"### Tahmin: **{CLASS_INFO[pred_class]}**")
                            
                            # Grafik
                            st.write("---")
                            st.write("#### Olasılık Dağılımı")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['#2E7D32' if c == pred_class else '#81C784' for c in SELECTED_CLASSES]
                            ax.barh([CLASS_INFO[c] for c in SELECTED_CLASSES], probabilities * 100, color=colors)
                            ax.set_xlabel('Olasılık (%)')
                            ax.set_xlim(0, 100)
                            st.pyplot(fig)
    
    # ========================================
    # TOPLU TAHMİN VE ANALİZ BÖLÜMÜ (AYNI SAYFADA)
    # ========================================
    
    st.write("---")
    st.write("---")
    st.header("Toplu Tahmin ve Performans Analizi")
    st.write("Birden fazla görsel yükleyerek toplu tahmin yapın ve detaylı analiz grafikleri görüntüleyin.")
    
    # Çoklu dosya yükleme
    uploaded_files = st.file_uploader(
        "Birden fazla bitki görseli seçin",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} görsel yüklendi")
        
        # Her görsel için gerçek etiket seçimi
        st.subheader("Gerçek Etiketleri Seçin")
        st.write("Her görsel için gerçek bitki türünü seçin:")
        
        true_labels = []
        cols_per_row = 4
        
        # Görselleri grid halinde göster
        for idx in range(0, len(uploaded_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, file in enumerate(uploaded_files[idx:idx+cols_per_row]):
                with cols[col_idx]:
                    img = Image.open(file)
                    st.image(img, use_container_width=True)
                    label = st.selectbox(
                        f"Görsel {idx+col_idx+1}",
                        options=[CLASS_INFO[c] for c in SELECTED_CLASSES],
                        key=f"label_{idx+col_idx}"
                    )
                    true_labels.append(label)
        
        st.write("---")
        
        # Tahmin butonu
        if st.button("Toplu Tahmin Yap ve Analiz Et", type="primary"):
            with st.spinner("Tüm görseller analiz ediliyor..."):
                
                # Tahminleri topla
                predictions = []
                probabilities_list = []
                true_indices = []
                
                progress_bar = st.progress(0)
                
                for idx, (file, true_label) in enumerate(zip(uploaded_files, true_labels)):
                    img = Image.open(file)
                    features = extract_features_optimized(img)
                    
                    if features is not None:
                        features = features.reshape(1, -1)
                        pred_idx = model.predict(features)[0]
                        probabilities = model.predict_proba(features)[0]
                        
                        predictions.append(pred_idx)
                        probabilities_list.append(probabilities)
                        
                        # Gerçek etiketi index'e çevir
                        true_idx = [i for i, c in enumerate(SELECTED_CLASSES) if CLASS_INFO[c] == true_label][0]
                        true_indices.append(true_idx)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                
                # NumPy array'e çevir
                y_true = np.array(true_indices)
                y_pred = np.array(predictions)
                y_proba = np.array(probabilities_list)
                
                st.success(f"{len(predictions)} görsel başarıyla analiz edildi!")
                
                # SONUÇLAR
                st.write("---")
                st.subheader("Analiz Sonuçları")
                
                # Metrikler
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1: 
                    st.metric("Doğruluk (Accuracy)", f"%{accuracy*100:.1f}")
                    st.caption("Doğru tahmin oranı")

                with col2:
                    st.metric("Kesinlik (Precision)", f"%{precision*100:.1f}")
                    st.caption("Pozitif tahminlerin doğruluğu")

                with col3:
                    st.metric("Duyarlılık (Recall)", f"%{recall*100:.1f}")
                    st.caption("Gerçek pozitifleri bulma oranı")

                with col4:
                    st.metric("F1-Score", f"%{f1*100:.1f}")
                    st.caption("Precision ve Recall dengesi")
                
                st.write("---")
                
                # GRAFİKLER
                col_g1, col_g2 = st.columns(2)
                
                # Confusion Matrix
                with col_g1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=range(len(SELECTED_CLASSES)))
                    
                    fig, ax = plt.subplots(figsize=(7, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=[CLASS_INFO[c] for c in SELECTED_CLASSES], 
                                yticklabels=[CLASS_INFO[c] for c in SELECTED_CLASSES],
                                cbar_kws={'label': 'Sayı'})
                    plt.xlabel('Tahmin Edilen')
                    plt.ylabel('Gerçek')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # ROC Eğrileri
                with col_g2:
                    st.subheader("ROC Eğrileri")
                    
                    y_true_bin = label_binarize(y_true, classes=np.arange(len(SELECTED_CLASSES)))
                    
                    fig, ax = plt.subplots(figsize=(7, 6))
                    
                    # Uyarıları bastır
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        for i, cls in enumerate(SELECTED_CLASSES):
                            if y_true_bin[:, i].sum() > 0:  # Bu sınıftan örnek varsa
                                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                                roc_auc = auc(fpr, tpr)
                                ax.plot(fpr, tpr, label=f'{CLASS_INFO[cls]} (AUC={roc_auc:.2f})', linewidth=2)
                    
                    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Rastgele (AUC=0.50)')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curves - Tüm Sınıflar')
                    ax.legend(loc='lower right', fontsize=9)
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.write("---")
                
                # Sınıf Bazlı Detaylı Rapor
                st.subheader("Sınıf Bazlı Detaylı Rapor")
                
                # Eğer birden fazla sınıf varsa rapor göster
                unique_classes = len(np.unique(np.concatenate([y_true, y_pred])))
                
                if unique_classes > 1:
                    report = classification_report(y_true, y_pred, 
                                                  labels=range(len(SELECTED_CLASSES)),
                                                  target_names=[CLASS_INFO[c] for c in SELECTED_CLASSES], 
                                                  output_dict=True,
                                                  zero_division=0)
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
                else:
                    st.info("Detaylı rapor için en az 2 farklı sınıftan örnek yükleyin.")
                
                st.write("---")
                
                # Tahmin Detayları
                st.subheader("Tahmin Detayları")
                
                results_data = []
                for idx, (file, true_idx, pred_idx) in enumerate(zip(uploaded_files, y_true, y_pred)):
                    results_data.append({
                        'Görsel': file.name,
                        'Gerçek': CLASS_INFO[SELECTED_CLASSES[true_idx]],
                        'Tahmin': CLASS_INFO[SELECTED_CLASSES[pred_idx]],
                        'Durum': 'Doğru' if true_idx == pred_idx else 'Yanlış'
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Yanlış tahminleri göster
                wrong_predictions = results_df[results_df['Durum'] == 'Yanlış']
                if len(wrong_predictions) > 0:
                    st.warning(f"{len(wrong_predictions)} yanlış tahmin yapıldı")
                    with st.expander("Yanlış Tahminleri Göster"):
                        st.dataframe(wrong_predictions, use_container_width=True, hide_index=True)
                else:
                    st.success("Tüm tahminler doğru!")

#  MODEL PERFORMANSI
elif page == "Model Performansı":
    st.title("Model Performans Analizi")
    
    if X_test is None or y_test is None:
        st.warning("Test verileri eksik.")
    else:
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        st.subheader("Genel Performans Metrikleri")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1: 
            st.metric("Doğruluk (Accuracy)", f"%{accuracy*100:.2f}")
            st.markdown(r"$\frac{TP + TN}{TP + TN + FP + FN}$")

        with col2:
            st.metric("Kesinlik (Precision)", f"%{precision*100:.2f}")
            st.markdown(r"$\frac{TP}{TP + FP}$")

        with col3:
            st.metric("Duyarlılık (Recall)", f"%{recall*100:.2f}")
            st.markdown(r"$\frac{TP}{TP + FN}$")

        with col4:
            st.metric("F1-Score", f"%{f1*100:.2f}")
            st.markdown(r"$2 \cdot \frac{Prec \cdot Recall}{Prec + Recall}$")

        st.write("---")
        
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=[CLASS_INFO[c] for c in SELECTED_CLASSES], 
                        yticklabels=[CLASS_INFO[c] for c in SELECTED_CLASSES],
                        cbar=False) 
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)

        with col_g2:
            st.subheader("ROC Eğrileri")
            
            y_test_bin = label_binarize(y_test, classes=np.arange(len(SELECTED_CLASSES)))
            
            
            fig, ax = plt.subplots(figsize=(5, 4))
            
            for i, cls in enumerate(SELECTED_CLASSES):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{CLASS_INFO[cls]} (AUC={roc_auc:.2f})', linewidth=1.5)
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=9)
            ax.set_ylabel('True Positive Rate', fontsize=9)
            ax.legend(loc='lower right', fontsize=8) 
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        st.write("---")
        
        # RAPOR TABLOSU 
        st.subheader("Sınıf Bazlı Detaylı Rapor")
        report = classification_report(y_test, y_pred, target_names=SELECTED_CLASSES, output_dict=True)
        
     
        report_df = pd.DataFrame(report).transpose()
        yeni_index = {c: CLASS_INFO[c] for c in SELECTED_CLASSES}
        report_df = report_df.rename(index=yeni_index)
        
        st.dataframe(report_df.round(3), use_container_width=True)