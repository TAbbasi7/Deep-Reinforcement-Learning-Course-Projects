# گزارش پروژه‌های یادگیری عمیق

این مستند نتایج، چالش‌ها و یافته‌های فنی چهار پروژه آموزشی در حوزه یادگیری عمیق را مستند می‌کند. تمرکز بر مشاهدات قابل تکرار و تحلیل رفتار واقعی مدل‌ها است.

---

## خلاصه پروژه‌ها

| پروژه | معماری | دیتاست | Epochs | نتیجه نهایی | وضعیت |
|-------|---------|---------|--------|-------------|-------|
| **MLP-MNIST** | Multi-Layer Perceptron | MNIST (60k train) | - | - | ✅ موفق |
| **Siamese-MNIST** | Twin Network + Contrastive Loss | MNIST (pairs) | - | Test Acc: **96.23%** | ✅ موفق (پس از تنظیم) |
| **CNN-CIFAR10** | Convolutional Neural Network | CIFAR-10 (50k train) | 100 | Val Acc: **81.08%**<br>Val Loss: **0.6321** | ✅ موفق |
| **Seq2Seq Chatbot** | LSTM Encoder-Decoder | Custom (5 samples) | - | خروجی بی‌معنی | ❌ شکست کامل |

---

## یافته‌های کلیدی

### 1️⃣ حساسیت شدید به Hyperparameter (Siamese Network)

**مشاهده تجربی:**

```python
# Configuration 1: ناپایدار
LR = 0.001
Dropout = 0.3
# نتیجه: Test Loss = nan (انفجار عددی)

# Configuration 2: پایدار
LR = 0.001  # بدون تغییر
Dropout = 0.5
# نتیجه: Test Accuracy = 96.23%
```

**تحلیل:**
- تغییر **تنها یک پارامتر** (Dropout: 0.3 → 0.5) تفاوت بین شکست کامل و موفقیت را ایجاد کرد
- این معماری به شدت به قدرت regularization در این نرخ یادگیری وابسته است
- نشان‌دهنده fragility مدل‌های Siamese در برابر تنظیمات hyperparameter

---

### 2️⃣ اهمیت LR Scheduling در آموزش طولانی (CIFAR-10)

**پیکربندی آموزش:**

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# با ImageDataGenerator برای data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

**شواهد از لاگ‌های آموزش:**

```
Epoch 97/100
LR: 1.5625e-05  # کاهش تدریجی از 0.001
val_accuracy: 0.8108
val_loss: 0.6321
```

**نتیجه‌گیری:**
- بدون LR scheduling + data augmentation، آموزش 100-epoch روی CIFAR-10 به overfitting شدید منجر می‌شد
- LR به‌طور پویا ۶۴ برابر کاهش یافت (0.001 → 1.5625e-05)

---

### 3️⃣ شکست کامل در غیاب داده (Seq2Seq)

**شرایط آزمایش:**

```python
# دیتاست
training_samples = 5  # فقط 5 جمله!

# معماری
Encoder: LSTM(256)
Decoder: LSTM(256)
```

**نتیجه واقعی:**

```
Input:  "how are you"
Output: "i am n"        # بی‌معنی کامل
```

**یافته:**
- معماری Encoder-Decoder بدون داده کافی هیچ توانایی generalization ندارد
- پیچیدگی معماری (LSTM layers) در غیاب data به هیچ‌وجه کمکی نمی‌کند
- این یک شکست مستند و قابل تکرار است (نه bug، بلکه محدودیت ذاتی)

---

## چالش‌های زیرساختی

### GPU Requirements

| پروژه | دلیل نیاز به GPU | زمان تقریبی (GPU) |
|-------|-------------------|-------------------|
| CIFAR-10 | 100 epochs × 50k images | چند ساعت |
| Seq2Seq | LSTM recurrent computations | متوسط |
| Siamese | Twin network forward passes | کم |
| MLP-MNIST | - | قابل اجرا روی CPU |

### موانع Platform

1. **Kaggle/Colab:** دسترسی به GPU های قوی‌تر (P100) نیازمند phone verification
2. **Session Timeout:** آموزش‌های طولانی در معرض قطع شدن ناگهانی
3. **Resource Limits:** محدودیت ساعات استفاده رایگان از GPU

---

## فایل‌های پروژه

```
📁 Deep Learning Projects
├── 📓 Image_Classification_on_MNIST_Tahere_ABBAsi_tutorial_mnist_siamese.ipynb
│   └── شبکه Siamese + تحلیل hyperparameter sensitivity
│
├── 📓 Tamrin_Image_Classification_on_CIFAR10_Tahere_ABBASi.ipynb
│   └── CNN با data augmentation و LR scheduling (100 epochs)
│
├── 📓 Tamrin_seq_chatbot_Tahereh_abbasi.ipynb
│   └── Seq2Seq chatbot (مستندسازی شکست با 5 samples)
│
└── 📓 Tamrin-1-10-4 MLP Image Classification on MNIST-Tahere-ABBAsi.ipynb.ipynb
    └── MLP ساده با کتابخانه tensorlayer
```

---

## وابستگی‌ها

### Core Dependencies

```python
tensorflow>=2.x      # فریم‌ورک اصلی (tf.keras)
numpy               # عملیات آرایه‌ای
scikit-learn        # metrics (confusion_matrix, classification_report)
plotly              # visualization
```

### Special Dependencies

```python
tensorlayer         # فقط برای پروژه MLP-MNIST
                    # نصب: pip install tensorlayer
```

### محیط اجرایی

- Google Colab (GPU runtime)
- Kaggle Notebooks (با GPU accelerator)

---

## نکات مهم برای تکرارپذیری

1. **Siamese Network:** حتماً `Dropout=0.5` استفاده شود (نه 0.3)
2. **CIFAR-10:** حداقل GPU با 4GB VRAM برای batch processing
3. **Seq2Seq:** این پروژه یک نمونه شکست است، نه یک مدل قابل استفاده
4. **همه پروژه‌ها:** دسترسی به GPU برای training time معقول ضروری است

---

## درس‌های آموخته‌شده

| # | یافته | اهمیت |
|---|-------|-------|
| 1 | تفاوت ۰.۲ در Dropout می‌تواند بین موفقیت و شکست کامل باشد | بالا |
| 2 | LR scheduling در آموزش‌های طولانی غیرقابل چشم‌پوشی است | بالا |
| 3 | معماری پیچیده بدون data کافی = شکست حتمی | بالا |
| 4 | دسترسی به GPU یک محدودیت واقعی است، نه صرفاً کمک‌کننده | متوسط |

---

