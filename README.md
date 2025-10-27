# CCTVAI

Davranış odaklı güvenlik analitiği için modüler bir Python kütüphanesi. Hazır modelleri ve kendi etiketli video verilerinizi kullanarak hırsızlık, sigara içme, bayılma, kaybolan çocuk gibi olayları öğrenip gerçek zamanlı uyarılar üretebilecek şekilde tasarlandı.

## Özellikler

- Çoklu kamera akışlarını (RTSP/IPTV/webcam) aynı anda izleyebilme
- YOLO tabanlı kişi tespiti
- DeepFace ile yaş/cinsiyet/duygu analizi
- HuggingFace video sınıflandırma modelleri ile davranış algılama
- Her 10 dakikada bir SQLite veritabanına toplu istatistik yazımı
- FastAPI tabanlı web arayüzü (kamera durumu, loglar, alarmlar)
- Typer CLI ile kolay kurulum ve kullanım

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> **Not:** `ultralytics`, `torch`, `deepface` gibi bağımlılıklar GPU ve CPU için farklı paketler gerektirebilir. Resmi talimatları inceleyerek uygun sürümleri kurduğunuzdan emin olun.

## Yapılandırma

Varsayılan ayarlar yerel bilgisayarınızdaki webcam (cihaz `0`) üzerinden sistemi başlatır. Kendi akışlarınızı tanımlamak için aşağıdaki gibi YAML dosyası oluşturabilirsiniz:

```yaml
streams:
  - name: Magaza Girişi
    url: rtsp://kullanici:sifre@192.168.1.10/stream1
    sampling_rate: 2
  - name: Kasa
    url: 0

detection:
  person_detector: yolov8m.pt
  behaviour_model: MCG-NJU/videomae-base-finetuned-kinetics

analytics:
  aggregation_interval_seconds: 600

storage:
  sqlite_path: data/cctvai.db
```

## Kullanım

### CLI

```bash
# Varsayılan ayarlarla kameraları izlemeye başla
cctvai run

# Özelleştirilmiş yapılandırma ile çalıştır
cctvai run --config config.yaml

# Yalnızca web arayüzünü başlat
cctvai dashboard --config config.yaml

# Son istatistikleri terminalde göster
cctvai stats --config config.yaml --limit 20

# Son alarmları görüntüle
cctvai alerts --config config.yaml --limit 20
```

### Web Arayüzü

`cctvai dashboard` komutu ile çalışan FastAPI uygulaması `http://localhost:8080` adresinde statik kontrol paneli sunar. Kameraların durumu, üretilen alarmlar ve toplanan istatistikleri gerçek zamanlı takip edebilirsiniz.

## Etiketli Veri ve Model Eğitimi

- Etiketli video segmentlerini `config.py` içindeki `BehaviourLabel` kayıtlarına ekleyerek özelleştirebilirsiniz.
- `VideoBehaviourClassifier`, HuggingFace üzerindeki herhangi bir `video-classification` modelini kullanabilir. Kendi verinizle yeniden eğittiğiniz modeli HuggingFace Hub'a yükleyerek burada referans gösterebilirsiniz.
- Alternatif olarak Ultralytics HUB, Roboflow gibi platformlardan hazır davranış tespit modelleri indirip `behaviour_model` parametresi olarak kullanabilirsiniz.

## Geliştirme

```bash
pip install -e .[web]
```

Kod stilini korumak için `ruff`, `black` gibi araçları isteğe bağlı ekleyebilirsiniz.

## Uyarılar

- Canlı sistemlerde kullanmadan önce yasal gereklilikleri, KVKK/GDPR gibi regülasyonları kontrol ediniz.
- Gerçek zamanlı analiz için güçlü GPU ve optimize edilmiş modeller gerekebilir. Örnek kod CPU üzerinde temel doğrulama içindir.
