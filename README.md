## Usage

```shell
docker network create mlflow
docker-compose up -d
# open mlflow folder in dev container and DEVELOP bruh
MLFLOW_S3_ENDPOINT_URL=http://artifacts:9000 MLFLOW_TRACKING_URI=http://tracking_server:5000 python train.py
```