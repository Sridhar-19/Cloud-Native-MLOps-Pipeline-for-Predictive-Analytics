# Amazon SageMaker Infrastructure

# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-sagemaker-execution-role"
  }
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Custom policy for S3 access
resource "aws_iam_policy" "sagemaker_s3_policy" {
  name        = "${var.project_name}-sagemaker-s3-policy"
  description = "Policy for SageMaker to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.data_bucket.arn}/*",
          "${aws_s3_bucket.data_bucket.arn}",
          "${aws_s3_bucket.model_bucket.arn}/*",
          "${aws_s3_bucket.model_bucket.arn}",
          "${aws_s3_bucket.monitoring_bucket.arn}/*",
          "${aws_s3_bucket.monitoring_bucket.arn}"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_policy_attachment" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

# SageMaker Model Registry
resource "aws_sagemaker_model_package_group" "churn_model_group" {
  model_package_group_name        = "churn-prediction-models"
  model_package_group_description = "Customer churn prediction model registry"

  tags = {
    Name = "churn-prediction-models"
  }
}

# SageMaker Domain for Studio (optional, for development)
resource "aws_sagemaker_domain" "studio_domain" {
  domain_name = "${var.project_name}-studio"
  auth_mode   = "IAM"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn
  }

  tags = {
    Name = "${var.project_name}-studio-domain"
  }
}

# Outputs
output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "model_registry_name" {
  description = "Model registry name"
  value       = aws_sagemaker_model_package_group.churn_model_group.model_package_group_name
}
