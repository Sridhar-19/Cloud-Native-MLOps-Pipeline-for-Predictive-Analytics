# S3 Buckets for MLOps Pipeline

resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-data-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-data"
    Environment = var.environment
    Purpose     = "Data Lake"
  }
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-models-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-models"
    Environment = var.environment
    Purpose     = "Model Artifacts"
  }
}

resource "aws_s3_bucket" "monitoring_bucket" {
  bucket = "${var.project_name}-monitoring-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-monitoring"
    Environment = var.environment
    Purpose     = "Monitoring Logs"
  }
}

# Enable versioning on buckets
resource "aws_s3_bucket_versioning" "data_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "model_versioning" {
  bucket = aws_s3_bucket.model_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "data_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_encryption" {
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "monitoring_encryption" {
  bucket = aws_s3_bucket.monitoring_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "data_public_access_block" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "model_public_access_block" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "monitoring_public_access_block" {
  bucket = aws_s3_bucket.monitoring_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policies
resource "aws_s3_bucket_lifecycle_configuration" "data_lifecycle" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    id     = "archive-old-data"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Outputs
output "data_bucket_name" {
  description = "Data bucket name"
  value       = aws_s3_bucket.data_bucket.id
}

output "model_bucket_name" {
  description = "Model bucket name"
  value       = aws_s3_bucket.model_bucket.id
}

output "monitoring_bucket_name" {
  description = "Monitoring bucket name"
  value       = aws_s3_bucket.monitoring_bucket.id
}
