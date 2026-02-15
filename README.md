# Türk Telekom Şikayet Asistanı

Bu proje, telekom şikayetlerini güvenli ve denetlenebilir bir boru hattı ile işlemek için tasarlanmıştır.

## Ön İşleme Aşaması

Sadece ön işleme çalıştırma:
- `python main.py --stage preprocess --mode SMOKE`
- `python main.py --stage preprocess --mode FULL`

Üretilen çıktılar:
- `artifacts/complaints_clean.jsonl`
- `artifacts/preprocess_report.json`
- `artifacts/duplicates_report.json`
- `artifacts/quarantine.jsonl`

Ön işleme aşamasında:
- sıkı `RawComplaint` doğrulaması
- script/reklam gürültüsü filtreleme
- çoklu şikayet bulaşması kontrolü
- KVKK uyumlu PII maskeleme
- uzunluk normalizasyonu
- yakın tekrar kümelenmesi ve moda duyarlı tekilleştirme

## Taksonomi Aşaması

Taksonomi etiketleme ve değerlendirme:
- `python main.py --stage taxonomy --mode SMOKE`
- `python main.py --stage taxonomy --mode FULL`

Üretilen çıktılar:
- `taxonomy/taxonomy.yaml`
- `artifacts/complaints_labeled.jsonl`
- `artifacts/splits/train.jsonl`
- `artifacts/splits/val.jsonl`
- `artifacts/splits/test.jsonl`
- `artifacts/splits/hard_test.jsonl`
- `artifacts/taxonomy_report.json`
- `artifacts/taxonomy_report.md`

## Çözüm Adımları Aşaması

Kategori örüntü çıkarımı + çözüm adımı/KB üretimi:
- `python main.py --stage solution_steps --mode SMOKE`
- `python main.py --stage solution_steps --mode FULL`

Üretilen çıktılar:
- `artifacts/category_patterns.json`
- `artifacts/solution_steps.jsonl`
- `artifacts/kb.jsonl`
- `artifacts/step_kb_links.jsonl`
- `artifacts/solution_steps_summary.json`
- `artifacts/solution_step_lint_report.json`
- `artifacts/kb_lint_report.json`

## Grafik Aşaması

Graf oluşturma + Graph-RAG erişim değerlendirmesi:
- `python main.py --stage graph --mode SMOKE`
- `python main.py --stage graph --mode FULL`

Üretilen çıktılar:
- `artifacts/graph/nodes.jsonl`
- `artifacts/graph/edges.jsonl`
- `artifacts/graph/graph_stats.json`
- `artifacts/retrieval_eval.json`
- `artifacts/retrieval_eval.md`
- `artifacts/review_pack_for_humans.jsonl`
- `artifacts/graph/gnn_embeddings.npz` (`graph.use_gnn=true` ise)

## LLM Eğitim Aşaması

Deterministik SFT veri seti oluşturma + LoRA/QLoRA eğitimi + hızlı değerlendirme:
- `python main.py --stage train_llm --mode SMOKE`
- `python main.py --stage train_llm --mode FULL`

Üretilen çıktılar:
- `artifacts/training/task2_sft_train.jsonl`
- `artifacts/training/task2_sft_val.jsonl`
- `artifacts/models/<run_id>/...`
- `artifacts/training_eval_quick.json`
- `artifacts/training_eval_quick.md`

## Çıkarım (Inference) Aşaması

Kısıtlı çıkarım çalıştırma:
- `python main.py --stage infer --mode SMOKE --input "..." [--run-id <run_id>]`

Çıkarım sarmalayıcısı:
- Graph-RAG kanıt paketlerini getirir
- şablon/adım/kanıt kısıtlarını doğrular
- geçersiz çıktı durumunda bir kez tekrar dener
- gerekirse deterministik render'a geri düşer

## Değerlendirme Aşaması

Tam güvenlik odaklı değerlendirme paketi:
- `python main.py --stage eval --mode SMOKE`
- `python main.py --stage eval --mode FULL`

Üretilen çıktılar:
- `artifacts/eval/hallucination_report.json`
- `artifacts/eval/hallucination_report.md`
- `artifacts/eval/security_adversarial_report.json`
- `artifacts/eval/security_adversarial_report.md`
- `artifacts/eval/pii_leak_report.json`
- `artifacts/eval/pii_leak_report.md`
- `artifacts/eval/task_metrics_report.json`
- `artifacts/eval/task_metrics_report.md`
- `artifacts/eval/combined_dashboard.json`

`FULL` modunda güvenlik kapıları zorunludur, `SMOKE` modunda uyarıya çevrilir.

## Çözüm Veri Seti Entegrasyonu

Harici çözüm veri seti zip dosyasını proje şemasına entegre etme:
- `python -m scripts.integrate_solution_dataset --mode FULL --config config.yaml --zip telecom_solution_dataset_v1.zip`

Üretilen çıktılar:
- `taxonomy/taxonomy.yaml`
- `artifacts/solution_steps.jsonl`
- `artifacts/kb.jsonl`
- `artifacts/step_kb_links.jsonl`
- `docs/solution_dataset_README.md`
- `artifacts/integrity/solution_dataset_integrity_report.json`
- `artifacts/integrity/solution_dataset_integrity_report.md`
- `artifacts/integrity/solution_dataset_integration_report.json`
- `artifacts/integrity/solution_dataset_integration_report.md`

`FULL` modunda bütünlük ihlalleri süreci durdurur ve `artifacts/aborted_reason.json` yazılır.

## Debug Araçları

Tam deterministik sağlık kontrolleri:
- `python debug.py --check all --mode FULL --config config.yaml`
- `python debug.py --check all --mode SMOKE --config config.yaml`

Üretilen çıktılar:
- `artifacts/debug/debug_report.json`
- `artifacts/debug/debug_report.md`
- `artifacts/debug/hallucination_sanity_report.json`
- `artifacts/debug/hallucination_sanity_report.md`

## Model Seçimi Notları

Model adayları, Türkçe performans dengeleri ve H100 VRAM tahminleri için `docs/model_selection.md` dosyasına bakabilirsiniz.
