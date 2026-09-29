"""The operations side of the router: storage, database, drift, and the training job.

storage.py         DigitalOcean Spaces (S3-compatible) for the dataset and every model version
db.py              Managed PostgreSQL: the audit log of predictions, human corrections, training runs
drift.py           compares live traffic with the reference the training job recorded
train_pipeline.py  the retraining job that runs on the GPU droplet
"""
