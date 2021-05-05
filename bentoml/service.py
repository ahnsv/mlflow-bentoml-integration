import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.artifact import PickleArtifact
from bentoml.adapters import DataframeInput

@env(infer_pip_packages=True)
@artifacts([PickleArtifact('model')])
class LinearRegressionService(BentoService):
    """
    LR 모델을 MLFlow로 부터 서빙하는 BentoService
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)

if __name__ == '__main__':
    import boto3
    import pickle


    lr_svc = LinearRegressionService()
    # load picklized model from s3
    s3 = boto3.resource('s3', 
        endpoint_url='http://artifacts:9000', 
        config=boto3.session.Config(signature_version='s3v4')
    )
    pickle_file_response = s3.meta.client.get_object(Bucket='mlflow', Key='0/80501ba3f47943ffaf5a2512aadeff24/artifacts/model/model.pkl')
    pickle_file_body = pickle_file_response.get("Body").read()
    pickle_data = pickle.loads(pickle_file_body)
    # ALTERNATIVE: use mlflow https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.download_artifacts
    # pack the model into svc
    lr_svc.pack("model", pickle_data)
    # serve
    saved_path = lr_svc.save()
    print(saved_path)