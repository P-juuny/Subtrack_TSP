# Install diagrams: pip install diagrams
from diagrams import Diagram, Cluster
from diagrams.aws.general import User
from diagrams.aws.network import InternetGateway, ELB, NatGateway, RouteTable
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.storage import S3, SimpleStorageService
from diagrams.aws.analytics import Cloudtrail
from diagrams.aws.management import Cloudwatch

with Diagram("SubTrack AWS Architecture", show=False, direction="LR", filename="subtrack_aws_architecture"):
    # Top-level AWS services
    s3 = S3("Static Assets (S3)")
    trail = Cloudtrail("CloudTrail")
    snap = SimpleStorageService("Snapshot Archive")
    cw = Cloudwatch("CloudWatch Metrics")

    with Cluster("VPC"):
        igw = InternetGateway("Internet Gateway")
        # Public Subnets
        with Cluster("Public Subnets"):
            rt_public = RouteTable("Public RT")
            alb = ELB("ALB")
            nat = NATGateway("NAT Gateway")
        # Private Subnets & DB
        with Cluster("Private Subnets"):
            # Web Tier
            with Cluster("Web Tier (AZ1)"):
                ec2_web1 = EC2("Web Server #1")
            with Cluster("Web Tier (AZ2)"):
                ec2_web2 = EC2("Web Server #2")
        with Cluster("DB Subnets"):
            # Multi-AZ RDS
            rds_primary = RDS("MySQL #1 (Primary)")
            rds_standby = RDS("MySQL #2 (Standby)")

        # Routing
        igw >> alb
        alb >> ec2_web1
        alb >> ec2_web2
        rt_public >> alb
        rt_public >> nat

        ec2_web1 >> rds_primary
        ec2_web2 >> rds_primary
        rds_primary >> rds_standby

        # NAT for outbound
        ec2_web1 >> nat
        ec2_web2 >> nat

    # Monitoring & Archival
    trail >> snap
    cw >> ec2_web1
    cw >> ec2_web2
    cw >> rds_primary
    cw >> rds_standby

    # External flows
    User() >> igw
    ec2_web1 >> s3
    ec2_web2 >> s3

