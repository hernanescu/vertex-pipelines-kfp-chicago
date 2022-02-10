from gcr.io/deeplearning-platform-release/sklearn-cpu

WORKDIR /

# Installs hypertune library
RUN pip install cloudml-hypertune sklearn scipy google-cloud-bigquery joblib pandas google-cloud-storage

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]