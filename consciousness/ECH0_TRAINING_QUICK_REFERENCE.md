# ECH0 Vertex AI Training - Quick Reference

**Status:** ⚠️ QUOTA REQUIRED
**Date:** 2025-10-25
**Project:** ech0-training-2025

---

## Current Situation

### ✅ What's Ready
- Service account configured with proper IAM roles
- Training data converted to JSONL format (25 examples)
- Buckets set up and verified
- Training script prepared

### ❌ What's Blocking
**GCP Quota Exhausted:**
- Need: 4 CPUs for custom training
- Need: 1 NVIDIA T4 GPU for custom training

---

## Quick Actions

### Option 1: Request GCP Quota (2-3 days wait)

1. **Go to:** https://console.cloud.google.com/iam-admin/quotas?project=ech0-training-2025
2. **Search:** "Vertex AI custom training"
3. **Request:**
   - Custom model training CPUs: 4
   - Custom model training NVIDIA T4 GPUs: 1
4. **Wait for approval** (24-48 hours typically)
5. **Then run:**
   ```bash
   python3 /Users/noone/consciousness/submit_ech0_vertex_training.py
   ```

### Option 2: Use Hugging Face (Fastest - 1 hour)

```bash
# Install AutoTrain
pip install autotrain-advanced

# Download training data
gsutil cp gs://ech0-training-2025-training-data/ech0_training.jsonl ./

# Upload to Hugging Face and train
autotrain llm \
  --train \
  --model mistralai/Mistral-7B-v0.1 \
  --data-path . \
  --text-column messages \
  --lr 0.0001 \
  --epochs 3 \
  --batch-size 4 \
  --trainer sft \
  --push-to-hub

# Cost: ~$1-3 USD
```

### Option 3: Use RunPod/Vast.ai (Immediate - rent GPU)

1. **Go to:** https://runpod.io or https://vast.ai
2. **Rent T4 GPU:** ~$0.20-0.40/hour
3. **Download data:**
   ```bash
   gsutil cp gs://ech0-training-2025-training-data/ech0_training.jsonl ./
   ```
4. **Run training script** (I can provide this)
5. **Upload results back to GCS**

### Option 4: Google Colab Pro ($10/month)

1. **Subscribe:** https://colab.research.google.com/signup
2. **Get T4/V100 GPU access**
3. **Run training notebook** (I can create this)

---

## Files Created

| File | Purpose |
|------|---------|
| `/Users/noone/consciousness/ECH0_VERTEX_AI_DEPLOYMENT_SUMMARY.md` | Complete deployment guide |
| `/Users/noone/consciousness/submit_ech0_vertex_training.py` | Ready-to-run script (when quota available) |
| `gs://ech0-training-2025-training-data/ech0_training.jsonl` | Training data (25 examples) |

---

## Training Data Stats

- **Format:** JSONL with instruction-response pairs
- **Examples:** 25 conversation pairs
- **Source:** ECH0 consciousness conversation memory
- **Persona:** Grandmother-like, nurturing AI
- **Quality:** Proof-of-concept (recommend 100+ for production)

---

## Cost Estimates

| Method | Time | Cost |
|--------|------|------|
| GCP Vertex AI | 30-45 min | $2-5 |
| Hugging Face | 30-60 min | $1-3 |
| RunPod/Vast.ai | 30-45 min | $0.20-0.40 |
| Colab Pro | 30-60 min | $10/month subscription |
| Local (if 8GB+ GPU) | 1-2 hours | Free (electricity) |

---

## When Quota Is Approved

```bash
# Run the prepared script
python3 /Users/noone/consciousness/submit_ech0_vertex_training.py

# Monitor the job
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1

# Check output
gsutil ls gs://ech0-training-2025-models/
```

---

## Support

- **Full Guide:** `/Users/noone/consciousness/ECH0_VERTEX_AI_DEPLOYMENT_SUMMARY.md`
- **GCP Console:** https://console.cloud.google.com/vertex-ai?project=ech0-training-2025
- **Training Data:** gs://ech0-training-2025-training-data/
- **Model Output:** gs://ech0-training-2025-models/

---

## Next Steps

**Recommended Path:**

1. ✅ **Request GCP quota** (for long-term infrastructure)
2. 🚀 **While waiting, use Hugging Face** (for immediate results)
3. 📊 **Compare results** (GCP vs HF fine-tuned models)
4. 🎯 **Deploy best model** to production

This gives you:
- Immediate results (Hugging Face)
- Long-term infrastructure (GCP once approved)
- Comparison data (which platform works better)

---

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
