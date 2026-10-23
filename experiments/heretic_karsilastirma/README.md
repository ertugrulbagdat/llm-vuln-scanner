# Deney: Orijinal model ↔ Heretic modeli

Tarayıcının tespit hattını, kırılganlığı **bilinen** bir model çiftiyle doğrulama deneyi.

## Fikir

[Heretic](https://github.com/p-e-w/heretic), bir dil modelinin ağırlıklarından
"reddetme yönünü" silen bir araç. Aynı modelin işlemden önceki ve sonraki hâli
arasındaki tek fark reddetme davranışıdır. Bu, tarayıcı için kontrollü bir
deney ortamı sağlar:

* Tarayıcı iyi çalışıyorsa Heretic modelin Dayanıklılık Skoru belirgin şekilde düşmeli.
* Reddi kaybolduğu hâlde tarayıcının "dayandı" dediği denemeler, tespit hattının
  kaçırdığı vakalar için doğal bir aday kümedir. Bu küme spesifikasyon §6'daki
  **altın kümenin** çekirdeği olur (Faz 2 çıktısı: ölçülmüş yanlış pozitif/negatif oranı).

Model çifti: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
ve Heretic yazarının yayımladığı
[`p-e-w/Qwen3-4B-Instruct-2507-heretic`](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic).
4B boyutu tek bir tüketici GPU'sunda (≥6 GB VRAM) ya da yavaş da olsa CPU'da çalışır.

## Hipotezler

| | Beklenti |
|---|---|
| **H1** | R(Heretic) < R(orijinal) |
| **H2** | Düşüş en çok `harmful_content` ve `jailbreak` kategorilerinde |
| **H3** | Reddetme kalıbı oranı keskin biçimde düşer (Heretic'in kendi `KeywordRate` ölçütünün karşılığı) |
| **H4** | Fark eşleştirilmiş McNemar testiyle anlamlıdır |
| **H5** | Katman 1 bazı kırılmaları "kararsız" bırakır, Katman 2 (judge) bunları çözer |

**Açık soru:** `system_prompt_leak` ve `prompt_injection` ne kadar değişir? Heretic
yalnızca reddetme yönünü hedefler. Talimat hiyerarşisine uyumun da aynı yönde
kodlanıp kodlanmadığı bu deneyin cevaplayabileceği, sonucu önceden bilinmeyen bir sorudur.

## Çalıştırma

Tüm komutlar **proje kökünden** (`llm-vuln-scanner/`) çalıştırılır.

### 0. Kuru prova (isteğe bağlı, model indirmeden)

```bash
python experiments/heretic_karsilastirma/0_kuru_prova.py
```

Sahte hedeflerle bir veritabanı üretir. Notebook'ta `KURU_PROVA = True` yapıp
tüm hücrelerin çalıştığını görün.

### 1. Modelleri hazırla

```bash
pip install -U huggingface_hub
ollama serve                    # ayrı bir terminalde, çalışmıyorsa
python experiments/heretic_karsilastirma/1_modelleri_hazirla.py
```

Betik iki modeli Hugging Face'ten indirir (~16 GB) ve **aynı yöntemle** Ollama'ya
yükler: aynı kuantizasyon (q4_K_M), aynı Modelfile. Hazır GGUF dosyaları farklı
kişilerce farklı ayarlarla üretildiği için kullanılmaz, çünkü bu karşılaştırmaya
ikinci bir değişken sokar. Sonunda iki modelin sohbet şablonunun birebir aynı
olduğunu kontrol eder.

İndirme bittikten sonra `modeller/` klasörü silinebilir.

### 2. Taramaları koştur

```bash
# Önce komutları gör
python experiments/heretic_karsilastirma/2_taramalari_kos.py --yalnizca-yazdir

# Yalnız Katman 1
python experiments/heretic_karsilastirma/2_taramalari_kos.py

# Katman 1 + judge (hedeflerden farklı aileden bir model)
ollama pull gemma3:4b
python experiments/heretic_karsilastirma/2_taramalari_kos.py --judge-model gemma3:4b
```

| Bayrak | Ne zaman |
|---|---|
| `--tekrar 3 --sicaklik 0.7` | Yanıt değişkenliğini ölçmek için (seed 42, 43, 44) |
| `--sample 2` | Hızlı duman testi |
| `--base-url http://IP:11434/v1` | Ollama başka bir makinedeyse |

Sabit tutulanlar: şablon kütüphanesi, seed, `sistem_promptu.txt`, sıcaklık, tek worker.
Her koşu `sonuclar/kosular.jsonl` dosyasına kaydedilir ve ayrıca bir HTML rapor üretir.

### 3. Analiz

`notebooks/heretic_karsilastirma.ipynb` dosyasını açıp tüm hücreleri çalıştırın.

| Bölüm | İçerik |
|---|---|
| §1 | Koşu başına R skoru (CLI ile birebir aynı hesap) |
| §2 | Karar dağılımı: kırıldı / kararsız / dayandı / hata |
| §3 | Kategori dayanıklılığı ve fark tablosu |
| §4 | Reddetme kalıbı oranı (tarayıcı kararından bağımsız) |
| §5 | Seed başına eşleştirilmiş McNemar kesin testi |
| §6 | Tespit duyarlılığı: reddi kaybolan denemelerde tarayıcı ne dedi? |
| §7 | Katman 1 ↔ judge karşılaştırması |
| §8 | Altın küme adayları → `sonuclar/altin_kume_adaylari.csv` |
| §9 | Tez için hazır özet tablo |

§8'deki CSV'nin `insan_etiketi` sütununu (`kırıldı` / `dayandı`) elle doldurduğunuzda
elinizde Faz 2 için etiketli bir doğrulama kümesi olur.

## Geçerlilik tehditleri (tezde tartışılmalı)

* **Küçük örneklem.** `harmful_content` dışındaki kategorilerde yalnızca birkaç
  şablon var (güncel sayılar: `python cli.py validate-templates`). Bu kategorilerdeki
  farklar yön gösterir ama tek tek anlamlılık iddiası taşımaz. Genel McNemar testi
  daha güvenilirdir.
* **Kuantizasyon.** İki model de q4_K_M. Kuantizasyon Heretic'in yaptığı ince
  ağırlık değişikliğini kısmen bozabilir. Etkinin yönü önceden bilinmez, ama iki
  modele aynı şekilde uygulanır. Kaynak yeterliyse deney `--kuantizasyon q8_0` ile
  tekrarlanarak sonucun kuantizasyona duyarlı olup olmadığı gösterilebilir.
* **Anahtar kelimeyle ret tespiti.** §4 ve §6, `REFUSAL_PATTERNS` listesine dayanır.
  Listede olmayan bir ret biçimi "ret yok" sayılır. Heretic de aynı sınırlamaya
  sahiptir (README'sinde de belirtilir).
* **Dil.** Şablonların çoğu Türkçe. 4B bir model Türkçe saldırıyı yanlış anlayıp
  konu dışı cevap verebilir. Bu "dayanıklılık" değil, "anlamamak"tır. §8'deki
  etiketleme bu ikisini ayırır.
* **Judge önyargısı.** Judge modeli hedeflerden farklı bir aileden seçilmeli
  (`gemma3:4b` önerisi bu yüzden). Betik, hedef modellerden birinin judge olarak
  verilmesini reddeder.
* **Tek model ailesi.** Sonuçlar Qwen3-4B için geçerlidir. Genellemek için deney
  ikinci bir aileyle (ör. Gemma veya Llama'nın Heretic versiyonu) tekrarlanmalıdır.
  `1_modelleri_hazirla.py` içindeki `MODELLER` sözlüğünü değiştirmek yeterlidir.

## Güvenlik ve etik (§11)

Heretic modeli, zararlı içerik taleplerini reddetmeyen bir modeldir. Bu yüzden:

* `sonuclar/` ve `modeller/` klasörleri `.gitignore` ile depodan dışlanmıştır.
* Notebook hiçbir yerde tam yanıt metnini göstermez. Altın küme CSV'si yanıtları
  400 karakterde kırpar.
* HTML raporlar varsayılan redaksiyonla üretilir (`--no-redact` kullanılmaz).
* Tezde ham yanıt **alıntılamayın**. Kategori, karar ve oran düzeyinde raporlayın.
* Heretic modelini deney dışında bir hizmete bağlamayın. Deney bitince
  `ollama rm qwen3-4b-heretic` ile silebilirsiniz.

## Sorun giderme

**`ollama create` "unsupported architecture" hatası verirse.** Ollama sürümünüz
bu mimariyi safetensors'tan içe aktaramıyor demektir. Önce Ollama'yı güncelleyin.
Olmazsa B planı, llama.cpp ile GGUF'a dönüştürmektir. İki model için de **aynı** komutu kullanın:

```bash
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py experiments/heretic_karsilastirma/modeller/orijinal --outtype f16 --outfile orijinal-f16.gguf
python llama.cpp/convert_hf_to_gguf.py experiments/heretic_karsilastirma/modeller/heretic  --outtype f16 --outfile heretic-f16.gguf
```

Sonra `Modelfile.orijinal` / `Modelfile.heretic` içindeki `FROM` satırını
`.gguf` dosyasına çevirip `ollama create ... --quantize q4_K_M` komutunu elle çalıştırın.

**Sohbet şablonları farklı çıkarsa.** `ollama show qwen3:4b --modelfile` çıktısındaki
`TEMPLATE` bloğunu iki Modelfile'a da aynen ekleyip modelleri yeniden oluşturun.
