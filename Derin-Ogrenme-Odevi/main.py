import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

os.makedirs("outputs", exist_ok=True)

print("CNN FASHION-MNIST PROJESİ BAŞLIYOR")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

print("Veri hazırlama tamamlandı.")



def cnn_modeli_olustur():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



model = cnn_modeli_olustur()

model.summary()

baslangic = time.perf_counter()

gecmis = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

egitim_suresi = time.perf_counter() - baslangic


test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test kaybı:", test_loss)
print("Test doğruluğu:", test_accuracy)
print("Eğitim süresi:", egitim_suresi, "saniye")



plt.figure(figsize=(8, 5))

plt.plot(gecmis.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(gecmis.history['val_accuracy'], label='Doğrulama Doğruluğu')

plt.title("Model Doğruluk Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.legend()
plt.grid()

plt.savefig("outputs/accuracy.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(gecmis.history['loss'], label='Eğitim Kaybı')
plt.plot(gecmis.history['val_loss'], label='Doğrulama Kaybı')

plt.title("Model Kayıp Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()
plt.grid()

plt.savefig("outputs/loss.png", dpi=300)
plt.show()

print("Eğitim grafikleri kaydedildi.")


def deney_modeli_olustur(filtre_sayisi):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            filtre_sayisi, (3, 3), activation='relu'
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu'
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model_8 = deney_modeli_olustur(8)

baslangic = time.perf_counter()

gecmis_8 = model_8.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

sure_8 = time.perf_counter() - baslangic

kayip_8, test_dogruluk_8 = model_8.evaluate(
    x_test, y_test, verbose=0
)


model_16 = deney_modeli_olustur(16)

baslangic = time.perf_counter()

gecmis_16 = model_16.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

sure_16 = time.perf_counter() - baslangic

kayip_16, test_dogruluk_16 = model_16.evaluate(
    x_test, y_test, verbose=0
)



karsilastirma = pd.DataFrame({
    "Ölçüm": [
        "Eğitim Doğruluğu",
        "Doğrulama Doğruluğu",
        "Test Doğruluğu",
        "Test Kaybı",
        "Eğitim Süresi (saniye)"
    ],

    "8 Filtre": [
        gecmis_8.history["accuracy"][-1] * 100,
        gecmis_8.history["val_accuracy"][-1] * 100,
        test_dogruluk_8 * 100,
        kayip_8,
        sure_8
    ],

    "16 Filtre": [
        gecmis_16.history["accuracy"][-1] * 100,
        gecmis_16.history["val_accuracy"][-1] * 100,
        test_dogruluk_16 * 100,
        kayip_16,
        sure_16
    ]
})

print(karsilastirma.to_string(index=False))

karsilastirma.to_csv(
    "outputs/filtre_karsilastirmasi.csv",
    index=False
)

print("Filtre karşılaştırması kaydedildi.")



siniflar = [
    "Tişört", "Pantolon", "Kazak", "Elbise", "Mont",
    "Sandalet", "Gömlek", "Spor Ayakkabı", "Çanta", "Bot"
]

tahminler = model.predict(x_test[:10])

plt.figure(figsize=(15, 7))

for i in range(10):
    tahmin_edilen = np.argmax(tahminler[i])
    gercek = y_test[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")

    durum = "Doğru" if tahmin_edilen == gercek else "Yanlış"

    plt.title(
        f"Tahmin: {siniflar[tahmin_edilen]}\n"
        f"Gerçek: {siniflar[gercek]}\n"
        f"{durum}"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/tahmin_ornekleri.png", dpi=300)
plt.show()

print("Tahmin görselleri kaydedildi.")



conv_katmani = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

ornek_girdi = tf.random.normal((1, 128, 128, 1))
conv_cikti = conv_katmani(ornek_girdi)

agirliklar, biaslar = conv_katmani.get_weights()

print("\nCONV2D KATMANI İNCELEMESİ")
print("Giriş boyutu:", ornek_girdi.shape)
print("Çıkış boyutu:", conv_cikti.shape)
print("Ağırlıkların boyutu:", agirliklar.shape)
print("Bias boyutu:", biaslar.shape)
print("İlk filtrenin ağırlıkları:\n", agirliklar[:, :, 0, 0])



plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(agirliklar[:, :, 0, i], cmap="gray")
    plt.title(f"Filtre {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/conv2d_filtreleri.png", dpi=300)
plt.show()

print("Conv2D filtre görselleri kaydedildi.")



girdi = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 8, 3],
    [4, 1, 5, 9]
])

pool_girdi = girdi.reshape(1, 4, 4, 1)

pooling = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)

pool_cikti = pooling(pool_girdi)

print("\nMAX POOLING İNCELEMESİ")
print("Giriş matrisi:\n", girdi)
print("Max Pooling çıkışı:\n", pool_cikti.numpy()[0, :, :, 0])
print("Giriş boyutu:", pool_girdi.shape)
print("Çıkış boyutu:", pool_cikti.shape)



plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(girdi, cmap="gray")
plt.title("Orijinal Girdi (4x4)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pool_cikti.numpy()[0, :, :, 0], cmap="gray")
plt.title("Max Pooling (2x2)")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/pooling_sonucu.png", dpi=300)
plt.show()


relu_girdi = tf.constant([-5, -2, -1, 0, 2, 4, 6], dtype=tf.float32)
relu = tf.keras.layers.ReLU()
relu_cikti = relu(relu_girdi)

print("\nRELU İNCELEMESİ")
print("Giriş:", relu_girdi.numpy())
print("Çıkış:", relu_cikti.numpy())


logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
softmax_cikti = softmax(logits)

print("\nSOFTMAX İNCELEMESİ")
print("Giriş:", logits.numpy())
print("Olasılıklar:", softmax_cikti.numpy())
print("Olasılıkların toplamı:", tf.reduce_sum(softmax_cikti).numpy())
print("Tahmin edilen sınıf:", tf.argmax(softmax_cikti, axis=1).numpy())
