import streamlit as st
import numpy as np
import os
import io
import base64
import tempfile
import pandas as pd


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input

from PIL import Image, ImageDraw, ImageFont

grid=[(20,20),(100,20),(180,20),(20,40),(140,80),(40,100),(100,150),(40,190),(120,220)]
font_sizes = [15, 20, 13, 35, 15, 38, 30, 35, 40] 

def sample_to_image(sample, save_directory):
    img = Image.new('RGB', (255, 255), color='black')
    draw = ImageDraw.Draw(img)
    for f, s, g in zip(font_sizes, sample, grid):
        font = ImageFont.truetype("FreeMono Bold.ttf", size=f)
        draw.text(g, str(s), font=font, fill='white')
    return img

def get_binary_file_downloader_html(bin_file, file_label='File', button_label='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">{button_label}</a>'
    return href

# Fungsi untuk memuat model dari file H5
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi untuk mengklasifikasikan gambar
def classify_image(model, image):
    image = tf.image.resize(image, (224, 224))  # Sesuaikan dengan ukuran input model Anda
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

if __name__ == '__main__':
    st.header('Implementasi SuperTML untuk Klasifikasi Genre Musik Indonesia')
    st.write('Aplikasi ini merupakan aplikasi untuk mengklasifikasikan Genre musik berdasarkan data audio analysis yang akan diklasifikasikan dari sebuah citra/gambar')
    st.write('Silakan Lihat terlebih dahulu di menu "Contoh data Audio Analysis dan Genre-nya" untuk melihat gambaran data')
    st.write('Kemudian bisa mencoba menginputkan nilai audio analysis pada menu "Konversi Tabular to Image" untuk mengubah data tabular menjadi citra')
    st.write('Jangan lupa untuk mengunduh citra, kemudian silakan coba klasifikasikan gambar yang sudah diunduh pada menu "Klasifikasi Genre Musik" ')
    apps = ['--Select--', 'Contoh data Audio Analysis dan Genre-nya', 'Konversi Tabular to Image', 'Klasifikasi Genre Musik']
    app_options = st.selectbox('Select application', apps)
    if app_options == 'Contoh data Audio Analysis dan Genre-nya':
        df = pd.read_csv('data.csv')
        st.dataframe(df)

    
    elif app_options == apps[2]:
        st.subheader('Pada bagian ini silakan inputkan nilai dari audio analysis sebagai berikut:')
        st.caption('1. Danceability: Danceability menggambarkan seberapa cocok suatu lagu untuk menari berdasarkan kombinasi elemen musik termasuk tempo, stabilitas ritme, kekuatan ketukan, dan keteraturan secara keseluruhan. Nilai 0,0 adalah yang paling tidak dapat ditarikan dan 1,0 adalah yang paling dapat ditarikan.')
        st.caption('2. Energy: Energi adalah ukuran dari 0,0 hingga 1,0 dan mewakili ukuran persepsi intensitas dan aktivitas. Biasanya, trek yang energik terasa cepat, keras, dan berisik. Misalnya, death metal memiliki energi yang tinggi, sedangkan pendahuluan Bach mendapat skor rendah pada skalanya.')
        st.caption('3. Loudness: Kenyaringan keseluruhan trek dalam desibel (dB). Nilai kenyaringan dirata-ratakan di seluruh trek dan berguna untuk membandingkan kenyaringan relatif trek. Kenyaringan adalah kualitas suara yang merupakan korelasi psikologis utama dengan kekuatan fisik (amplitudo). Nilai biasanya berkisar antara -60 dan 0 db.')
        st.caption('4. Tempo: Perkiraan tempo keseluruhan suatu lagu dalam detak per menit (BPM). Dalam terminologi musik, tempo adalah kecepatan atau laju suatu karya tertentu dan diturunkan langsung dari durasi ketukan rata-rata.')
        st.caption('5. Duration MS: Durasi trek dalam milidetik.')
        st.caption('6. Speechiness: Speechiness mendeteksi keberadaan kata-kata yang diucapkan dalam sebuah trek. Semakin eksklusif rekaman yang menyerupai ucapan (misalnya acara bincang-bincang, buku audio, puisi), semakin mendekati 1,0 nilai atributnya')
        st.caption('7. Instumentalness: Memprediksi apakah suatu lagu tidak mengandung vokal. Bunyi "Ooh" dan "aah" dianggap instrumental dalam konteks ini. Lagu rap atau kata-kata yang diucapkan jelas bersifat "vokal".')
        st.caption('8. Liveness: Mendeteksi keberadaan penonton dalam rekaman. Nilai keaktifan yang lebih tinggi menunjukkan peningkatan kemungkinan bahwa lagu tersebut ditampilkan secara langsung.')
        st.caption('9. Valence: Ukuran dari 0,0 hingga 1,0 yang menggambarkan kepositifan musik yang disampaikan oleh sebuah lagu.')

        with st.form('Feature Audio Analysis'):
            st.write('Input Features')
            danceability = st.text_input('Danceability: ')
            energy = st.text_input('Energy: ')
            loudness = st.text_input('Loudness: ')
            tempo = st.text_input('Tempo: ')
            duration_ms = st.text_input('Duration MS: ')
            speechiness = st.text_input('Speechiness: ')
            instrumentalness = st.text_input('Instrumentalness: ')
            liveness = st.text_input('Liveness: ')
            valence = st.text_input('Valence: ')

            save_directory = "output_images"
            os.makedirs(save_directory, exist_ok=True)

            input_values = [danceability, energy, loudness, tempo, duration_ms, speechiness, instrumentalness, liveness, valence]
            image_converted = sample_to_image(input_values, save_directory)
        
            st.form_submit_button('Generate')

        img_bytes = io.BytesIO()
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, "generated_image.png")
        image_converted.save(temp_file_path, format="PNG")
        st.image(image_converted, use_column_width=True, caption='generated image')

        file_name = 'generated_image.png'
        st.markdown("### Download Image")
        if st.button("Klik di sini untuk mengunduh gambar"):
            st.markdown(get_binary_file_downloader_html(temp_file_path, file_name), unsafe_allow_html=True)

    elif app_options == apps[3]:
        st.markdown('Klasifikasi Genre Musik')   
        st.title('Aplikasi Klasifikasi Gambar')
        model_path = 'mobilenet_sf_3.h5'
        model = load_model(model_path)

        uploaded_image = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
        # Tampilkan gambar yang diunggah
            image_uploaded = Image.open(uploaded_image)
            st.image(image_uploaded, caption="Gambar yang diunggah", use_column_width=True)

            classification_dict = {
            0: 'indonesian pop',
            1: 'indonesian folk',
            2: 'classic indonesian rock',
            3: 'dangdut',
            4: 'indonesian indie'
}
            # Klasifikasikan gambar jika ada gambar yang diunggah
            if st.button('Klasifikasikan'):
                image_uploaded = np.array(image_uploaded)
                predictions = classify_image(model, image_uploaded)  # Anda harus menentukan bagaimana Anda mendapatkan prediksi ini
                highest_prediction = np.argmax(predictions)  # Mengambil indeks prediksi tertinggi
                result = classification_dict.get(highest_prediction, 'Klasifikasi tidak ditemukan')  # Mengambil label klasifikasi

                st.write('Hasil Klasifikasi:')
                st.write(result)
            


