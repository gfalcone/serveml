import os
import logging

from mlflow.tracking import MlflowClient
from mlflow.models import Model


class MlflowModelLoader(object):
    """
    Class for loading MLflow models
    """

    def __init__(self, tracking_uri):
        super().__init__()
        self.mlflow_client = MlflowClient(tracking_uri)

    def load_model(
            self,
            registered_model_name,
            model_directory,
            tmp_directory='/tmp',
            stages=["Production"]
    ):
        # get latest version
        model_versions = self.mlflow_client.get_latest_versions(
            registered_model_name,
            stages=stages
        )
        logging.debug(
            'Registered models retrieved : {}'.format(model_versions)
        )
        if len(model_versions) != 1:
            raise ValueError(
                'Did not retrieve exactly one model but got these {}'.format(
                    model_versions
                )
            )
        model_version = model_versions[0]

        # get artifact name
        run_id = model_version.run_id
        logging.info(
            'Run id associated to registered model : {}'.format(run_id)
        )
        artifacts = list(
            map(lambda x: x.path, self.mlflow_client.list_artifacts(run_id))
        )
        logging.debug(
            'Artifacts retrieved linked to run_id {} : {}'.format(
                run_id,
                artifacts
            )
        )

        if model_directory not in artifacts:
            raise ValueError(
                'Could not find an artifact named "model" linked to this run_id'  # NOQA
            )

        # downloading artifact to temporary directory
        logging.info(
            'Downloading artifact {} to {}'.format(
                model_directory,
                tmp_directory
            )
        )
        self.mlflow_client.download_artifacts(
            run_id,
            model_directory,
            tmp_directory
        )

        # loading model thanks to Mlflow
        model_path = os.path.join(tmp_directory, model_directory)
        logging.info('Loading model {}'.format(model_path))
        return Model().load(model_path)


if __name__ == '__main__':
    model = MlflowModelLoader(
        'http://localhost:5000',
    )
    print(model.load_model('my_model', 'model', '/tmp'))
    # print(model.mlflow_client.get_registered_model_details('my_model'))
