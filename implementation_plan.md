# Cloud-Native MLOps Pipeline for Predictive Analytics - Implementation Plan

## Overview

This implementation plan details the construction of an end-to-end, production-ready MLOps pipeline on AWS for customer churn prediction. The pipeline automates model training, deployment, and monitoring while achieving the following targets:

**Performance Goals:**
- **Model Accuracy:** 84% with F1-score of 0.79
- **Scalability:** Handle 5,000+ requests/hour
- **Availability:** 99% uptime
- **Deployment Speed:** Reduce from 2 days to ~4 hours
- **Auto-retraining:** Trigger when performance drops below 80%

**Technology Stack:**
- **Cloud Platform:** AWS (SageMaker, Lambda, S3, EKS, CloudWatch, ECR)
- **ML Framework:** TensorFlow 2.x
- **Containerization:** Docker
- **Orchestration:** Kubernetes (EKS), AWS Step Functions
- **CI/CD:** Jenkins, AWS CodePipeline
- **Programming:** Python 3.9+

---

## User Review Required

> [!IMPORTANT]
> **AWS Infrastructure Costs:** This implementation will utilize multiple AWS services including EKS, SageMaker, Lambda, S3, and CloudWatch. Estimated monthly cost ranges from $500-$2000 depending on data volume and request traffic.

> [!IMPORTANT]
> **Customer Data Requirements:** This plan assumes access to customer churn data with 50,000+ records. If you don't have real customer data, we can use publicly available datasets (e.g., Telco Customer Churn Dataset from Kaggle).

> [!WARNING]
> **Multi-Account Strategy:** For production deployments, AWS recommends separate accounts for dev, staging, and production environments. This plan assumes a single AWS account but can be extended.

> [!CAUTION]
> **EKS Cluster Provisioning:** Setting up EKS clusters can take 15-20 minutes per cluster. Consider keeping development clusters running to avoid repeated provisioning delays.

---

## Proposed Changes

### Phase 1: Project Foundation & Infrastructure Setup

#### [NEW] Project Structure
```
mlops-pipeline/
├── data/                          # Data management
│   ├── raw/
│   ├── processed/
│   └── scripts/
│       ├── data_ingestion.py
│       ├── feature_engineering.py
│       └── data_validation.py
├── models/                        # ML models
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── churn_model.py
│   └── artifacts/
├── deployment/                    # Deployment configs
│   ├── docker/
│   │   ├── Dockerfile.train
│   │   ├── Dockerfile.inference
│   │   └── requirements.txt
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── hpa.yaml
│   │   └── ingress.yaml
│   └── terraform/                # IaC for AWS resources
│       ├── main.tf
│       ├── sagemaker.tf
│       ├── eks.tf
│       ├── s3.tf
│       └── lambda.tf
├── pipelines/                     # MLOps pipelines
│   ├── sagemaker_pipeline.py
│   ├── step_functions/
│   │   └── training_workflow.json
│   └── jenkins/
│       └── Jenkinsfile
├── monitoring/                    # Monitoring & drift detection
│   ├── drift_detection.py
│   ├── model_monitor.py
│   └── cloudwatch_alarms.py
├── api/                          # Inference API
│   ├── app.py
│   ├── predictor.py
│   └── requirements.txt
├── tests/                        # Testing
│   ├── unit/
│   ├── integration/
│   └── performance/
├── notebooks/                    # Jupyter notebooks for exploration
├── config/
│   ├── config.yaml
│   └── secrets.yaml.example
└── README.md
```

#### [NEW] AWS Infrastructure Components

**S3 Buckets:**
- `mlops-pipeline-data-{env}` - Raw and processed data
- `mlops-pipeline-models-{env}` - Model artifacts and checkpoints
- `mlops-pipeline-monitoring-{env}` - Monitoring logs and reports

**IAM Roles:**
- `SageMakerExecutionRole` - For SageMaker training and deployment
- `LambdaExecutionRole` - For Lambda functions
- `EKSWorkerNodeRole` - For EKS worker nodes
- `JenkinsEC2Role` - For Jenkins CI/CD server

**VPC Configuration:**
- Custom VPC with public and private subnets across 3 AZs
- NAT Gateways for private subnet internet access
- Security groups for SageMaker, EKS, and Lambda

---

### Phase 2: Data Pipeline & Feature Engineering

#### [NEW] [data/scripts/data_ingestion.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/data/scripts/data_ingestion.py)

**Purpose:** Ingest customer data from multiple sources and store in S3

**Key Features:**
- S3 event-triggered Lambda for automatic data ingestion
- Data validation and schema enforcement
- Incremental data loading support
- Data versioning with DVC (Data Version Control)

**Implementation:**
- AWS Lambda function triggered by S3 PUT events
- Data quality checks (null values, data types, ranges)
- Publish CloudWatch metrics for data pipeline health

#### [NEW] [data/scripts/feature_engineering.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/data/scripts/feature_engineering.py)

**Purpose:** Transform raw customer data into ML-ready features

**Key Features:**
- Customer demographic features (age, location, tenure)
- Behavioral features (usage patterns, service interactions)
- Aggregated features (average monthly charges, total services)
- Temporal features (seasonality, trends)
- Feature scaling and encoding

**Implementation:**
- SageMaker Processing Job using scikit-learn
- Feature store integration (AWS Feature Store)
- Automated feature validation

---

### Phase 3: ML Model Development (TensorFlow Churn Prediction)

#### [NEW] [models/training/churn_model.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/models/training/churn_model.py)

**Purpose:** TensorFlow model architecture for customer churn prediction

**Architecture:**
```
Input Layer (# features) 
    ↓
Dense(128, activation='relu', kernel_regularizer=l2(0.001))
    ↓
BatchNormalization()
    ↓
Dropout(0.3)
    ↓
Dense(64, activation='relu', kernel_regularizer=l2(0.001))
    ↓
BatchNormalization()
    ↓
Dropout(0.3)
    ↓
Dense(32, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')  # Binary classification
```

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Class weight balancing for imbalanced data
- Early stopping and model checkpointing

#### [NEW] [models/training/train.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/models/training/train.py)

**Purpose:** SageMaker-compatible training script

**Key Features:**
- Hyperparameter tuning support
- Experiment tracking with SageMaker Experiments
- Model artifact saving to S3
- Training metrics logging to CloudWatch

#### [NEW] [models/training/evaluate.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/models/training/evaluate.py)

**Purpose:** Model evaluation and performance validation

**Metrics:**
- Accuracy: Target 84%+
- F1-Score: Target 0.79+
- Precision/Recall curves
- Confusion matrix
- ROC AUC score

---

### Phase 4: Containerization with Docker

#### [NEW] [deployment/docker/Dockerfile.train](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/docker/Dockerfile.train)

**Purpose:** Docker image for model training

**Base Image:** `tensorflow/tensorflow:2.13.0-gpu`

**Key Components:**
- TensorFlow 2.13+
- SageMaker Training Toolkit
- Custom training dependencies
- Optimized for multi-GPU training

#### [NEW] [deployment/docker/Dockerfile.inference](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/docker/Dockerfile.inference)

**Purpose:** Lightweight Docker image for model serving

**Base Image:** `tensorflow/serving:2.13.0`

**Key Components:**
- TensorFlow Serving
- REST API endpoint
- Health check endpoints
- Minimal image size for fast deployment

**Images stored in:** Amazon ECR (Elastic Container Registry)

---

### Phase 5: Kubernetes Orchestration on EKS

#### [NEW] [deployment/kubernetes/deployment.yaml](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/kubernetes/deployment.yaml)

**Purpose:** Kubernetes deployment for inference service

**Configuration:**
- **Replicas:** 3 (for high availability)
- **Resource Requests:** 
  - CPU: 500m
  - Memory: 1Gi
- **Resource Limits:**
  - CPU: 2000m
  - Memory: 4Gi
- **Rolling Update Strategy:** MaxSurge=1, MaxUnavailable=0
- **Readiness/Liveness Probes:** HTTP health checks

#### [NEW] [deployment/kubernetes/hpa.yaml](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/kubernetes/hpa.yaml)

**Purpose:** Horizontal Pod Autoscaler for automatic scaling

**Configuration:**
- **Min Replicas:** 3
- **Max Replicas:** 20
- **Target CPU Utilization:** 70%
- **Target Memory Utilization:** 80%
- **Custom Metrics:** Requests per second (5000+ req/hour capability)

#### [NEW] [deployment/kubernetes/service.yaml](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/kubernetes/service.yaml)

**Purpose:** Kubernetes Service with LoadBalancer

**Configuration:**
- Type: LoadBalancer (AWS NLB)
- Port: 8080 (HTTP), 8443 (HTTPS)
- Session Affinity: ClientIP (optional)

#### [NEW] [deployment/terraform/eks.tf](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/deployment/terraform/eks.tf)

**Purpose:** Infrastructure as Code for EKS cluster

**Configuration:**
- **Kubernetes Version:** 1.28+
- **Node Groups:**
  - General: t3.large (2-5 nodes)
  - ML Inference: c5.xlarge (2-10 nodes)
- **Karpenter:** For dynamic node provisioning
- **Cluster Autoscaler:** For automatic scaling
- **AWS Load Balancer Controller:** For ingress
- **EBS CSI Driver:** For persistent storage

---

### Phase 6: SageMaker MLOps Pipeline

#### [NEW] [pipelines/sagemaker_pipeline.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/pipelines/sagemaker_pipeline.py)

**Purpose:** Automated end-to-end training and deployment pipeline

**Pipeline Stages:**

1. **Data Processing**
   - Input: Raw data from S3
   - Processor: SageMaker Processing Job
   - Output: Processed training/validation/test sets

2. **Hyperparameter Tuning**
   - Strategy: Bayesian Optimization
   - Objective: Maximize F1-Score
   - Parameters: learning_rate, dropout_rate, layer_sizes

3. **Model Training**
   - Estimator: TensorFlow Estimator
   - Instance Type: ml.p3.2xlarge (GPU)
   - Training Input: Processed data from S3

4. **Model Evaluation**
   - Evaluation Script: evaluate.py
   - Condition: Register model only if accuracy >= 84% and F1 >= 0.79

5. **Model Registration**
   - Registry: SageMaker Model Registry
   - Version: Automatic semantic versioning
   - Metadata: Training metrics, hyperparameters

6. **Conditional Deployment**
   - Condition: If model approved in registry
   - Target: SageMaker Endpoint (staging first)

#### [NEW] [pipelines/step_functions/training_workflow.json](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/pipelines/step_functions/training_workflow.json)

**Purpose:** AWS Step Functions orchestration for complex workflows

**Workflow Steps:**
1. Data Validation (Lambda)
2. Feature Engineering (SageMaker Processing)
3. Model Training (SageMaker Training)
4. Model Evaluation (Lambda)
5. Performance Check (Choice State)
6. Model Registration (SageMaker)
7. Deployment Notification (SNS)

---

### Phase 7: Monitoring & Automated Retraining

#### [NEW] [monitoring/model_monitor.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/monitoring/model_monitor.py)

**Purpose:** SageMaker Model Monitor for continuous monitoring

**Monitoring Types:**

1. **Data Quality Monitoring**
   - Track feature distributions
   - Detect missing values
   - Identify outliers

2. **Model Quality Monitoring**
   - Real-time prediction accuracy
   - F1-Score tracking
   - Confusion matrix updates

3. **Bias Drift Monitoring**
   - Fairness metrics
   - Demographic parity

4. **Feature Attribution Drift**
   - SHAP value tracking

**Baseline Creation:**
- Generated from training dataset
- Stored in S3
- Updated quarterly

**Monitoring Schedule:**
- Frequency: Hourly
- Instance Type: ml.m5.xlarge

#### [NEW] [monitoring/drift_detection.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/monitoring/drift_detection.py)

**Purpose:** Statistical drift detection algorithms

**Methods:**
- **Kolmogorov-Smirnov Test:** For continuous features
- **Chi-Square Test:** For categorical features
- **Population Stability Index (PSI):** Overall distribution shift

**Thresholds:**
- Warning: PSI > 0.1
- Critical: PSI > 0.25
- Action: Trigger retraining

#### [NEW] [monitoring/cloudwatch_alarms.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/monitoring/cloudwatch_alarms.py)

**Purpose:** CloudWatch alarms for automated retraining triggers

**Alarms:**

1. **Model Performance Degradation**
   - Metric: ModelAccuracy
   - Threshold: < 80%
   - Action: Trigger EventBridge → Step Functions → Retraining

2. **Data Drift Detected**
   - Metric: DataDriftScore
   - Threshold: > 0.25
   - Action: SNS notification + retraining

3. **Inference Latency**
   - Metric: ModelLatency
   - Threshold: > 200ms (p99)
   - Action: Scale up pods / optimize model

4. **Error Rate**
   - Metric: 5XXErrors
   - Threshold: > 1%
   - Action: Alert + rollback

**Automated Retraining Pipeline:**
```
CloudWatch Alarm → EventBridge Rule → Lambda Function → Step Functions → 
SageMaker Pipeline → Model Training → Evaluation → Auto-deployment (if approved)
```

---

### Phase 8: CI/CD with Jenkins

#### [NEW] [pipelines/jenkins/Jenkinsfile](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/pipelines/jenkins/Jenkinsfile)

**Purpose:** Automated testing and deployment pipeline

**Pipeline Stages:**

1. **Source Code Checkout**
   - SCM: Git (GitHub/CodeCommit)
   - Branch Strategy: GitFlow (develop, staging, main)

2. **Code Quality**
   - Linting: pylint, flake8
   - Type Checking: mypy
   - Security Scan: bandit

3. **Unit Tests**
   - Framework: pytest
   - Coverage: > 80%
   - Test Docker builds

4. **Build Docker Images**
   - Build training and inference images
   - Push to Amazon ECR
   - Tag with Git commit SHA and semantic version

5. **Integration Tests**
   - Test SageMaker training job
   - Validate model artifacts
   - Test inference endpoint

6. **Deploy to Staging**
   - Update EKS deployment (staging namespace)
   - Run smoke tests
   - Performance testing (load test)

7. **Manual Approval Gate**
   - Human review required for production

8. **Deploy to Production**
   - Blue/Green Deployment strategy
   - Gradual traffic shift (10% → 50% → 100%)
   - Automatic rollback on errors

**Rollback Capabilities:**
- Store last 5 model versions in registry
- One-click rollback via Jenkins parameter
- Automatic rollback on alarm triggers

#### [NEW] Infrastructure for Jenkins

**EC2 Instance:**
- Instance Type: t3.large
- OS: Ubuntu 22.04
- Plugins: AWS, Docker, Kubernetes, SageMaker
- Backup: Daily AMI snapshots

---

### Phase 9: Inference API

#### [NEW] [api/app.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/api/app.py)

**Purpose:** Flask REST API for model inference

**Endpoints:**

- `POST /predict` - Single prediction
  - Input: JSON with customer features
  - Output: Churn probability + prediction

- `POST /batch-predict` - Batch predictions
  - Input: Array of customer records
  - Output: Array of predictions

- `GET /health` - Health check
  - Output: Model version, status, uptime

- `GET /metrics` - Prometheus metrics
  - Output: Request count, latency, errors

**Features:**
- Request validation with Pydantic
- Response caching with Redis
- Rate limiting
- Authentication (API keys)

#### [NEW] [api/predictor.py](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/api/predictor.py)

**Purpose:** Model loading and prediction logic

**Optimizations:**
- Model warmup on startup
- TF Serving gRPC for low latency
- Batch inference support
- Model version management

---

### Phase 10: Testing Infrastructure

#### [NEW] [tests/unit/](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/tests/unit/)

**Purpose:** Unit tests for all components

**Coverage:**
- Data processing functions
- Feature engineering
- Model training/evaluation
- API endpoints
- Monitoring utilities

#### [NEW] [tests/integration/](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/tests/integration/)

**Purpose:** Integration tests

**Tests:**
- SageMaker pipeline execution
- Docker container builds
- Kubernetes deployments
- Model registry operations
- CloudWatch alarm triggers

#### [NEW] [tests/performance/](file:///C:/Users/ksrid/Desktop/Cloud-Native%20MLOps%20Pipeline%20for%20Predictive%20Analytics/tests/performance/)

**Purpose:** Performance and load testing

**Tools:** Locust, JMeter

**Tests:**
- Load test: 5000+ requests/hour
- Latency: p50 < 50ms, p99 < 200ms
- Concurrent users: 100+
- Autoscaling validation

---

## Verification Plan

### Automated Tests

1. **Unit Tests**
   ```bash
   cd mlops-pipeline
   pytest tests/unit/ --cov=. --cov-report=html
   ```
   - **Success Criteria:** 80%+ code coverage, all tests pass

2. **Integration Tests**
   ```bash
   pytest tests/integration/ -v
   ```
   - **Success Criteria:** SageMaker pipeline executes successfully, model registers in Model Registry

3. **Performance Tests**
   ```bash
   locust -f tests/performance/load_test.py --host=http://[EKS-ENDPOINT]
   ```
   - **Success Criteria:** 
     - Handle 5000+ requests/hour
     - p99 latency < 200ms
     - 0% error rate under normal load

4. **Docker Build Tests**
   ```bash
   docker build -f deployment/docker/Dockerfile.train -t churn-train:test .
   docker build -f deployment/docker/Dockerfile.inference -t churn-inference:test .
   ```
   - **Success Criteria:** Both images build without errors, size < 2GB each

### AWS Infrastructure Validation

5. **SageMaker Pipeline Execution**
   ```bash
   python pipelines/sagemaker_pipeline.py --execute
   ```
   - **Success Criteria:** 
     - Pipeline completes all stages
     - Model achieves 84%+ accuracy, F1-score 0.79+
     - Model auto-registers in Model Registry

6. **EKS Deployment Validation**
   ```bash
   kubectl apply -f deployment/kubernetes/
   kubectl get pods -n mlops-production
   kubectl describe hpa churn-prediction-hpa -n mlops-production
   ```
   - **Success Criteria:** 
     - 3 pods running successfully
     - HPA configured correctly
     - LoadBalancer service gets external IP

7. **Model Monitoring Setup**
   ```bash
   python monitoring/model_monitor.py --create-baseline
   python monitoring/cloudwatch_alarms.py --create-alarms
   ```
   - **Success Criteria:** 
     - Baseline created in S3
     - CloudWatch alarms active
     - Monitoring schedule running

8. **CI/CD Pipeline Test**
   - Trigger Jenkins pipeline manually
   - **Success Criteria:** 
     - All stages pass (build, test, deploy to staging)
     - Approval gate functions correctly
     - Successful deployment to staging namespace

### Manual Verification

9. **Inference API Testing**
   - Send POST request to `/predict` endpoint with sample customer data
   - Verify response format and prediction accuracy
   - **Success Criteria:** 
     - API returns valid JSON response
     - Prediction confidence scores reasonable
     - Latency < 100ms for single prediction

10. **Drift Detection Testing**
    - Manually inject data drift (modify feature distributions)
    - Wait for hourly monitoring job
    - **Success Criteria:** 
      - Drift detected in CloudWatch metrics
      - Alarm triggers within expected timeframe
      - SNS notification received

11. **Auto-scaling Validation**
    - Run load test with increasing traffic
    - Monitor Kubernetes HPA and pod count
    - **Success Criteria:** 
      - Pods scale from 3 to 10+ under load
      - No request failures during scaling
      - Pods scale down after load decreases

12. **Automated Retraining Test**
    - Manually trigger CloudWatch alarm for model performance < 80%
    - **Success Criteria:** 
      - EventBridge triggers Step Functions workflow
      - SageMaker training job starts automatically
      - New model version appears in Model Registry

13. **Rollback Test**
    - Deploy a deliberately broken model version
    - Trigger rollback via Jenkins
    - **Success Criteria:** 
      - Previous working model version restored
      - Zero downtime during rollback
      - Inference continues successfully

14. **99% Uptime Validation**
    - Monitor CloudWatch metrics for 7 days
    - **Success Criteria:** 
      - Uptime > 99% (< 1.7 hours downtime per week)
      - No extended outages

---

## Implementation Timeline

**Phase 1-2 (Foundation & Data Pipeline):** 3-5 days  
**Phase 3-4 (Model Development & Docker):** 4-6 days  
**Phase 5 (Kubernetes/EKS):** 3-4 days  
**Phase 6 (SageMaker Pipeline):** 3-4 days  
**Phase 7 (Monitoring & Retraining):** 3-4 days  
**Phase 8 (CI/CD Jenkins):** 2-3 days  
**Phase 9-10 (API & Testing):** 2-3 days  

**Total Estimated Time:** 20-30 days for complete implementation

---

## Cost Estimation (Monthly)

- **EKS Cluster:** ~$75 (control plane) + $150-300 (worker nodes)
- **SageMaker:** ~$200-500 (training jobs + endpoints)
- **S3:** ~$50 (data storage)
- **Lambda:** ~$20 (invocations)
- **CloudWatch:** ~$30 (logs + metrics)
- **ECR:** ~$10 (image storage)
- **Data Transfer:** ~$50
- **Jenkins EC2:** ~$50

**Estimated Total:** $585-$1,135/month (varies with usage)

---

## Next Steps After Approval

1. Set up AWS account and configure credentials
2. Create initial S3 buckets and IAM roles
3. Initialize project repository with structure
4. Begin infrastructure provisioning with Terraform
5. Implement data pipeline components
