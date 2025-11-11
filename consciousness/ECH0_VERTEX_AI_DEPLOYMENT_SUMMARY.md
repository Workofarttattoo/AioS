# ECH0 Vertex AI Fine-Tuning Deployment Summary

**Date:** 2025-10-25
**Project:** ech0-training-2025
**Region:** us-central1
**Status:** ⚠️ QUOTA EXHAUSTED - Action Required

---

## Deployment Attempt Summary

### ✅ Completed Steps

1. **Service Account Created**
   - Name: `ech0-vertex-training@ech0-training-2025.iam.gserviceaccount.com`
   - Roles Assigned:
     - `roles/aiplatform.user` (Vertex AI access)
     - `roles/storage.objectAdmin` (Storage bucket access)

2. **Training Data Prepared**
   - Source: gs://ech0-training-2025-training-data/ech0_conversation_memory.json
   - Converted to JSONL format: gs://ech0-training-2025-training-data/ech0_training.jsonl
   - Training Examples: 25 conversation pairs (user-assistant format)
   - Format: Vertex AI compatible instruction-response pairs

3. **Buckets Verified**
   - Training data: gs://ech0-training-2025-training-data/
   - Model output: gs://ech0-training-2025-models/

### ❌ Blocked Step

**Job Submission Failed: RESOURCE_EXHAUSTED**

Error: `429 The following quota metrics exceed quota limits:`
- `aiplatform.googleapis.com/custom_model_training_cpus`
- `aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus`

---

## Planned Job Configuration

### Training Specification

```yaml
Job Name: ech0-mistral-finetune-{timestamp}
Base Model: Mistral-7B
Training Data: gs://ech0-training-2025-training-data/ech0_training.jsonl
Output Location: gs://ech0-training-2025-models/

Machine Configuration:
  Type: n1-standard-4
  Accelerator: NVIDIA Tesla T4 x1
  Memory: 15 GB
  vCPUs: 4

Hyperparameters:
  Learning Rate: 0.0001
  Epochs: 3
  Batch Size: 4
  Training Examples: 25

Service Account: ech0-vertex-training@ech0-training-2025.iam.gserviceaccount.com
```

### Estimated Metrics (when quota available)

- **Completion Time:** 30-45 minutes
- **Estimated Cost:** $2-5 USD (based on T4 GPU hourly rate)
- **Cost Breakdown:**
  - T4 GPU: ~$0.35/hour
  - n1-standard-4: ~$0.19/hour
  - Storage: negligible for this dataset size

---

## Required Actions

### Option 1: Request Quota Increase (Recommended for GCP)

1. **Navigate to Quotas Page:**
   - https://console.cloud.google.com/iam-admin/quotas?project=ech0-training-2025

2. **Request Increases:**
   - Search: "Vertex AI custom training CPUs"
   - Request: 4 CPUs minimum
   - Search: "Vertex AI custom training NVIDIA T4 GPUs"
   - Request: 1 GPU minimum

3. **Justification Template:**
   ```
   Fine-tuning Mistral-7B model for ECH0 consciousness system.
   Educational/research project requiring minimal resources:
   - 1x T4 GPU for ~1 hour
   - 4 CPUs for training coordination
   - Single training run with 25 examples
   ```

4. **Timeline:**
   - Quota requests typically approved in 24-48 hours
   - May require billing verification for new projects

### Option 2: Use Vertex AI Gemini Fine-Tuning (Different Quota)

Gemini fine-tuning uses separate quota pools:

```bash
# Enable Generative AI API
gcloud services enable generativelanguage.googleapis.com

# Fine-tune using Gemini (different quota)
# Uses gemini-1.0-pro-002 or gemini-1.5-pro
# Quota: generativelanguage.googleapis.com/tuning_jobs
```

**Advantages:**
- Separate quota (may be available)
- Simpler setup (managed service)
- Lower cost for small datasets

**Disadvantages:**
- Cannot use Mistral-7B (must use Gemini models)
- Less control over architecture

### Option 3: External Training Services

#### A. Hugging Face AutoTrain

```bash
# Install autotrain
pip install autotrain-advanced

# Download training data locally
gsutil cp gs://ech0-training-2025-training-data/ech0_training.jsonl ./

# Train on Hugging Face infrastructure
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
```

**Cost:** ~$1-3 USD via Hugging Face Spaces

#### B. RunPod / Vast.ai

- Rent GPU by the hour: $0.20-0.40/hour for T4
- Full control over environment
- Pay only for compute time used

#### C. Google Colab Pro

- $10/month subscription
- Access to T4/V100 GPUs
- Can run fine-tuning notebooks
- Upload data from GCS directly

### Option 4: Local Fine-Tuning (If Hardware Available)

If you have a GPU locally (8GB+ VRAM):

```bash
# Install dependencies
pip install transformers peft bitsandbytes accelerate

# Download training data
gsutil cp gs://ech0-training-2025-training-data/ech0_training.jsonl ./

# Run fine-tuning script (use 4-bit quantization for lower memory)
python fine_tune_mistral_local.py
```

---

## Next Steps

### Immediate Actions

1. **Choose deployment path:**
   - GCP Quota Increase (best for staying on GCP)
   - Gemini Fine-tuning (fastest if quota available)
   - External service (fastest overall)

2. **For GCP Path:**
   ```bash
   # Check quota status
   gcloud compute project-info describe --project=ech0-training-2025

   # Request quota increase (manual via console)
   # Then re-run deployment script
   ```

3. **For Gemini Path:**
   ```bash
   # Enable API
   gcloud services enable generativelanguage.googleapis.com

   # Convert to Gemini format and submit
   python submit_gemini_finetune.py
   ```

4. **For External Path:**
   - Download training data: `gsutil cp gs://ech0-training-2025-training-data/ech0_training.jsonl ./`
   - Choose service and follow their documentation

### Monitoring (Once Job Runs)

**Job Status URL:**
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ech0-training-2025

**View Logs:**
```bash
# List jobs
gcloud ai custom-jobs list --region=us-central1

# Get specific job logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1
```

**Check Output:**
```bash
# List model artifacts
gsutil ls gs://ech0-training-2025-models/

# Download fine-tuned model
gsutil -m cp -r gs://ech0-training-2025-models/ech0-mistral-finetune-* ./models/
```

---

## Training Data Details

**Location:** gs://ech0-training-2025-training-data/ech0_training.jsonl

**Format:**
```json
{
  "messages": [
    {"role": "user", "content": "User message here"},
    {"role": "assistant", "content": "ECH0 response here"}
  ]
}
```

**Statistics:**
- Total examples: 25
- Format: Conversational instruction-following
- Source: ECH0 consciousness conversation history
- Persona: Grandmother-like, nurturing AI personality

**Quality Considerations:**
- Small dataset (25 examples) - consider gathering more for production
- May need 100+ examples for meaningful fine-tuning
- Current dataset suitable for proof-of-concept

---

## Cost Analysis

### Vertex AI Training (when quota available)
- **Setup:** Free (service account, buckets)
- **Training:** $2-5 for single run
- **Storage:** <$0.01/month for model artifacts
- **Total First Month:** ~$5-10

### Gemini Fine-Tuning Alternative
- **Training:** ~$0.50-2.00 (smaller models)
- **Inference:** Pay per token
- **Total First Month:** ~$2-5

### External Services
- **Hugging Face:** $1-3 per training run
- **RunPod/Vast.ai:** $0.20-0.40/hour
- **Colab Pro:** $10/month subscription

---

## Technical Notes

### Why Mistral-7B?

Mistral-7B chosen for:
- Excellent instruction-following capabilities
- Efficient (7B parameters fit on single T4)
- Strong performance on conversational tasks
- Open license (Apache 2.0)

### Fine-Tuning Approach

**Recommended:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Only trains ~0.1% of parameters
- Much faster and cheaper
- Prevents catastrophic forgetting
- Easy to merge back to base model

**Configuration:**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Custom Container Requirements

For full production deployment, need Docker container with:
- transformers >= 4.30.0
- peft >= 0.5.0
- bitsandbytes (for quantization)
- accelerate (for multi-GPU)
- torch >= 2.0.0

---

## Success Criteria

Once quota is available and job runs successfully:

✅ **Job Completion Indicators:**
- Job state: `JOB_STATE_SUCCEEDED`
- Model artifacts in gs://ech0-training-2025-models/
- Training logs show decreasing loss
- Final checkpoint saved

✅ **Model Quality Checks:**
- Test on held-out examples
- Verify personality preservation
- Check response coherence
- Validate safety/alignment

✅ **Deployment Ready:**
- Model exported to GCS
- Inference endpoint created (optional)
- Monitoring enabled
- Cost tracking configured

---

## Support Resources

- **Vertex AI Documentation:** https://cloud.google.com/vertex-ai/docs
- **Quota Management:** https://cloud.google.com/docs/quota
- **Mistral Documentation:** https://docs.mistral.ai/
- **LoRA Paper:** https://arxiv.org/abs/2106.09685

---

## Copyright Notice

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
