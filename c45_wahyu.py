import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from math import log2
from graphviz import Digraph
import os

# Judul aplikasi dan informasi mahasiswa
st.set_page_config(layout="wide", page_title="Klasifikasi Tingkat Permintaan Produk Laut Berdasarkan Musim Menggunakan Decision Tree C4.5")

st.title("KLASIFIKASI TINGKAT PERMINTAAN PRODUK LAUT BERDASARKAN MUSIM MENGGUNAKAN DECISION TREE C4.5")
st.markdown("**Nama Mahasiswa   : WAHYU CAVIN GUNAWAN**")
st.markdown("**NPM              : 212350112**")
st.markdown("**Universitas Harapan Medan**")
st.markdown("""
    <style>
        * {
            color: #000000 !important;
        }
        
        .dataframe-container {
            width: 100%;
            overflow-x: auto;
            margin: 1em 0;
            border: 2px solid #000000;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        
        .dataframe thead th {
            background-color: #2c3e50 !important;
            color: white !important;
            font-weight: bold;
            padding: 12px 15px;
            text-align: center;
            border-bottom: 2px solid #000000;
            position: sticky;
            top: 0;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f8f9fa;
        }
        
        .dataframe tbody tr:last-of-type {
            border-bottom: 2px solid #000000;
        }
        
        .dataframe tbody tr:hover {
            background-color: #e9ecef;
        }
        
        .dataframe td {
            padding: 12px 15px;
            text-align: center;
            border-right: 1px solid #dddddd;
        }
        
        .dataframe td:last-child {
            border-right: none;
        }
        
        .highlight {
            font-weight: bold !important;
            background-color: #ffff99 !important;
        }
        
        .stText, .stMarkdown, .stAlert, label,
        h1, h2, h3, h4, h5, h6,
        .st-bb, .st-at, .st-ae, .st-af, .st-ag {
            color: #000000 !important;
        }
        
        .stTextInput input, .stNumberInput input,
        .stSelectbox select, .stSlider div {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

def df_to_html_table(df, highlight_max=False, add_index=False, show_attributes=False):
    html = f"""
    <div class="dataframe-container">
        <table class="dataframe">
            <thead>
                <tr>
    """

    if add_index:
        html += "<th>No</th>"
    if show_attributes:
        html += "<th>Atribut</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for i, row in df.iterrows():
        html += "<tr>"
        if add_index:
            html += f"<td>{i + 1}</td>"
        if show_attributes:
            html += f"<td>{row.name}</td>"
        for col in df.columns:
            cell_value = row[col]

            if isinstance(cell_value, float):
                cell_value = f"{cell_value:.4f}"
            if highlight_max and col in ["Information Gain", "Gain"] and cell_value == df[col].max():
                html += f'<td class="highlight">{cell_value}</td>'
            else:
                html += f"<td>{cell_value}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    
    return html


def custom_encoder(df, columns):
    le_dict = {}
    df_encoded = df.copy()
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df_encoded, le_dict

# Fungsi utama
def main():
    st.sidebar.header("Unggah Data")
    file_diupload = st.sidebar.file_uploader("Pilih file CSV/Excel", type=["csv", "xlsx"])

    if file_diupload is not None:
        try:
            if file_diupload.name.endswith('.csv'):
                data = pd.read_csv(file_diupload)
            else:
                data = pd.read_excel(file_diupload)
            
            st.header("Data Awal")
            st.markdown(df_to_html_table(data, add_index=True), unsafe_allow_html=True)

            # Drop kolom yang tidak digunakan
            if 'Produk' in data.columns:
                data = data.drop('Produk', axis=1)
                st.write("Kolom 'Produk' dihapus karena tidak digunakan dalam perhitungan")

            # Validasi kolom
            required_columns = ['Musim', 'Kategori_Penjualan', 'Tingkat_Permintaan']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Data harus mengandung kolom: {', '.join(required_columns)}")
                return

            # Encoding data untuk perhitungan model
            data_encoded, label_encoders = custom_encoder(data, ['Musim', 'Kategori_Penjualan', 'Tingkat_Permintaan'])
            
            # Simpan mapping untuk interpretasi hasil
            musim_mapping = dict(zip(label_encoders['Musim'].classes_, label_encoders['Musim'].transform(label_encoders['Musim'].classes_)))
            penjualan_mapping = dict(zip(label_encoders['Kategori_Penjualan'].classes_, 
                                      label_encoders['Kategori_Penjualan'].transform(label_encoders['Kategori_Penjualan'].classes_)))
            permintaan_mapping = dict(zip(label_encoders['Tingkat_Permintaan'].classes_, 
                                       label_encoders['Tingkat_Permintaan'].transform(label_encoders['Tingkat_Permintaan'].classes_)))

            # ===============================
            # Perhitungan Information Gain
            # ===============================
            st.header("Perhitungan Information Gain")

            def entropi(kolom_target):
                elemen, jumlah = np.unique(kolom_target, return_counts=True)
                return -np.sum([(jumlah[i]/np.sum(jumlah)) * log2(jumlah[i]/np.sum(jumlah)) 
                              for i in range(len(elemen))])

            def info_gain(data, nama_atribut, nama_target="Tingkat_Permintaan"):
                entropi_total = entropi(data[nama_target])
                nilai, jumlah = np.unique(data[nama_atribut], return_counts=True)

                entropi_terbobot = np.sum([
                    (jumlah[i]/np.sum(jumlah)) * entropi(data[data[nama_atribut] == nilai[i]][nama_target])
                    for i in range(len(nilai))
                ])

                return entropi_total - entropi_terbobot

            atribut = ['Musim', 'Kategori_Penjualan']
            daftar_gain = {a: info_gain(data, a, 'Tingkat_Permintaan') for a in atribut}
            tabel_gain = pd.DataFrame.from_dict(daftar_gain, orient='index', columns=['Information Gain'])
            
            # Tampilkan tabel 
            st.markdown(df_to_html_table(tabel_gain, highlight_max=True, show_attributes=True), unsafe_allow_html=True)
            
            atribut_terbaik = max(daftar_gain, key=daftar_gain.get)
            st.success(f"Atribut dengan Gain tertinggi: **{atribut_terbaik}**")

            # ============================
            # Proses Pembangunan Pohon
            # ============================
            st.header("Proses Pembangunan Pohon Keputusan")

            def hitung_gain_subset(sub_data, atribut_terpakai, target="Tingkat_Permintaan"):
                sisa_atribut = [a for a in atribut if a != atribut_terpakai]
                return {a: info_gain(sub_data, a, target) for a in sisa_atribut}

            st.subheader(f"Cabang Pertama - Berdasarkan {atribut_terbaik}")
            nilai_atribut = data[atribut_terbaik].unique()

            for nilai in nilai_atribut:
                with st.expander(f"Node {atribut_terbaik} = {nilai}"):
                    subset_data = data[data[atribut_terbaik] == nilai]
                    if len(subset_data['Tingkat_Permintaan'].unique()) == 1:
                        st.info(f"Semua data berlabel: {subset_data['Tingkat_Permintaan'].iloc[0]} (Node Daun)")
                    else:
                        gain_subset = hitung_gain_subset(subset_data, atribut_terbaik)
                        tabel_gain_subset = pd.DataFrame.from_dict(gain_subset, orient='index', columns=['Gain'])
                        
                        # Tampilkan tabel 
                        st.markdown(df_to_html_table(tabel_gain_subset, highlight_max=True, show_attributes=True), unsafe_allow_html=True)
                        
                        terbaik_selanjutnya = max(gain_subset, key=gain_subset.get)
                        st.success(f"Atribut terbaik selanjutnya: **{terbaik_selanjutnya}**")

            # ====================
            # Pelatihan dan Evaluasi Model
            # ====================
            st.header("Pelatihan dan Evaluasi Model")

            # Bagi data
            X = data_encoded[['Musim', 'Kategori_Penjualan']]
            y = data_encoded['Tingkat_Permintaan']
            X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.3, random_state=42)

            kol1, kol2 = st.columns(2)
            with kol1:
                st.subheader("Data Latih (70%)")
                st.write(f"Jumlah data latih: {len(y_latih)}")
                distribusi_latih = pd.Series(y_latih).map({v:k for k,v in permintaan_mapping.items()}).value_counts().reset_index()
                distribusi_latih.columns = ['Permintaan', 'Jumlah']
                distribusi_latih = distribusi_latih[['Permintaan', 'Jumlah']]  # Reorder columns
                st.markdown(df_to_html_table(distribusi_latih), unsafe_allow_html=True)

            with kol2:
                st.subheader("Data Uji (30%)")
                st.write(f"Jumlah data uji: {len(y_uji)}")
                distribusi_uji = pd.Series(y_uji).map({v:k for k,v in permintaan_mapping.items()}).value_counts().reset_index()
                distribusi_uji.columns = ['Permintaan', 'Jumlah']
                distribusi_uji = distribusi_uji[['Permintaan', 'Jumlah']]  # Reorder columns
                st.markdown(df_to_html_table(distribusi_uji), unsafe_allow_html=True)

            # Latih model pohon keputusan
            pohon = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=5)
            pohon.fit(X_latih, y_latih)
            prediksi = pohon.predict(X_uji)

            # Evaluasi hasil model
            st.subheader("Evaluasi Model")
            
            # Matriks kebingungan dengan label asli
            st.write("Matriks Kebingungan (Confusion Matrix)")
            cm = confusion_matrix(y_uji, prediksi)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_encoders['Tingkat_Permintaan'].classes_, 
                        yticklabels=label_encoders['Tingkat_Permintaan'].classes_, ax=ax)
            st.pyplot(fig)

            # Laporan klasifikasi dengan label asli
            st.write("Laporan Klasifikasi")
            y_uji_asli = pd.Series(y_uji).map({v:k for k,v in permintaan_mapping.items()})
            prediksi_asli = pd.Series(prediksi).map({v:k for k,v in permintaan_mapping.items()})
            laporan = classification_report(y_uji_asli, prediksi_asli, output_dict=True)
            
            # Tampilkan laporan klasifikasi dengan HTML
            laporan_df = pd.DataFrame(laporan).transpose()
            st.markdown(df_to_html_table(laporan_df), unsafe_allow_html=True)

            akurasi_cv = cross_val_score(pohon, X, y, cv=5, scoring='accuracy')
            st.write(f"Cross Validation (5-fold) - Rata-rata Akurasi: {np.mean(akurasi_cv):.4f}")

            # ==========================
            # Visualisasi Pohon Keputusan
            # ==========================
            st.header("Visualisasi Pohon Keputusan")

            def bangun_pohon(data, atribut, target):
                label = data[target]
                if len(np.unique(label)) == 1:
                    return label.iloc[0]
                if len(atribut) == 0:
                    return label.mode()[0]
                daftar_gain = [info_gain(data, a, target) for a in atribut]
                indeks_terbaik = np.argmax(daftar_gain)
                atribut_terbaik = atribut[indeks_terbaik]
                gain_terbaik = daftar_gain[indeks_terbaik]
                if gain_terbaik == 0:
                    return label.mode()[0]
                pohon = {f"{atribut_terbaik}\n(Gain={gain_terbaik:.4f})": {}}
                for nilai in data[atribut_terbaik].unique():
                    subset = data[data[atribut_terbaik] == nilai]
                    if subset.empty:
                        pohon[f"{atribut_terbaik}\n(Gain={gain_terbaik:.4f})"][nilai] = label.mode()[0]
                    else:
                        atribut_baru = [a for a in atribut if a != atribut_terbaik]
                        subtree = bangun_pohon(subset, atribut_baru, target)
                        pohon[f"{atribut_terbaik}\n(Gain={gain_terbaik:.4f})"][nilai] = subtree
                return pohon

            def visualisasi_pohon(pohon, graph=None, node_id=0, induk=None, label_cabang=None):
                if graph is None:
                    graph = Digraph(format='png')
                    graph.attr(rankdir='TB')
                id_saat_ini = str(node_id)
                if isinstance(pohon, dict):
                    akar = next(iter(pohon))
                    graph.node(id_saat_ini, akar, shape='box', style='filled', fillcolor='lightblue')
                    if induk is not None:
                        graph.edge(induk, id_saat_ini, label=label_cabang)
                    cabang = pohon[akar]
                    for label, anak in cabang.items():
                        node_baru = len(graph.body)
                        visualisasi_pohon(anak, graph, node_baru, id_saat_ini, str(label))
                else:
                    graph.node(id_saat_ini, str(pohon), shape='ellipse', style='filled', fillcolor='lightgreen')
                    if induk is not None:
                        graph.edge(induk, id_saat_ini, label=label_cabang)
                return graph

            struktur_pohon = bangun_pohon(data, atribut, target="Tingkat_Permintaan")
            graf_pohon = visualisasi_pohon(struktur_pohon)

            lokasi_file = "pohon_keputusan_permintaan"
            graf_pohon.render(lokasi_file, format="png", cleanup=True)

            st.image(f"{lokasi_file}.png", caption="Visualisasi Pohon Keputusan Berdasarkan Information Gain")

            # =====================
            # Interpretasi Hasil
            # =====================
            st.header("Interpretasi Hasil")
            
            st.markdown("""
            **Keterangan Encoding:**
            - **Musim**: 
              - {}
            - **Kategori Penjualan**: 
              - {}
            - **Tingkat Permintaan**: 
              - {}
            """.format(
                "\n              - ".join([f"{k} = {v}" for k,v in musim_mapping.items()]),
                "\n              - ".join([f"{k} = {v}" for k,v in penjualan_mapping.items()]),
                "\n              - ".join([f"{k} = {v}" for k,v in permintaan_mapping.items()])
            ))

            st.markdown("""
            **Insight Bisnis:**
            1. Atribut paling berpengaruh: **{}**
            2. Pola permintaan tertinggi cenderung terjadi ketika:
               - {}
            3. Rekomendasi manajemen stok:
               - {}
            """.format(
                atribut_terbaik,
                "Kondisi musim {} dengan penjualan {}".format(
                    list(musim_mapping.keys())[np.argmax([pohon.feature_importances_[0]])],
                    list(penjualan_mapping.keys())[np.argmax([pohon.feature_importances_[1]])]
                ),
                "Tingkatkan stok saat musim {} dan kurangi saat {}".format(
                    list(musim_mapping.keys())[0], list(musim_mapping.keys())[-1]
                )
            ))

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.info("Silakan unggah file data CSV atau Excel untuk memulai analisis.")

if __name__ == "__main__":
    main()
