# Vertex AI: Serverless framework for MLOPs (ESP / ENG)

## Español

### Qué es esto?
Este repo contiene un pipeline end to end diseñado usando el SDK de Kubeflow Pipelines (KFP). En el contexto del uso de Vertex AI como solución, la idea es construir una arquitectura de machine learning lo más automatizada posible, integrando algunos de los principales servicios de Google Cloud Platform (GCP) tales como BigQuery (data warehousing), Google Cloud Storage (almacenamiento de objetos) y Container Registry (repositorio de inágenes de Docker).

### Cómo lo corro?
- Primero, ejecutar la notebook **pipeline_setup.ipynb**. Contiene la configuración de la infraestructura que será utilizada: se crean datasets en BigQuery y buckets en GCS y se instalan librerías necesarias. Además se crean imágenes de Docker y se pushea a Container Registry para los jobs de tuneos de hiperparámetros. 
- Segundo, dentro de la carpeta *components* se encuentra la notebook **components_definition.ipynb** que deberá ejecutarse para generar los .yamls que serán invocados en la notebook principal de ejecución. 
- Por último, seguir los pasos indicados en **pipeline_run.ipynb**. Algunos parámetros como la cantidad de trials de hiperparámetros o los tipos de máquina deseadas para algunos pasos pueden ser fácilmente modificables.

### TO-DO
agregar costo estimado
permisos

## English

### What is this?
This repo contains an end to end pipeline designed using Kubelow Pipelines SDK (KFP). Using Vertex AI as a main solution, the idea is to build a machine learning architecture as automated as possible, integrating some of the main Google Cloud Platform (GCP) services, such as BigQuery (data warehousing), Google Cloud Storage (storage system) and Container Registry (Docker images repository).

### How do I run it?
- First, execute **pipeline_setup.ipynb**. It contains the infraestructure configuration to be used: BigQuery datasets and GCS buckets are created and installs the necessary libraries. It also creates Docker images and pushes them to Container Registry in order to perform hyperparameter tuning jobs. 
- Second, in the *components* folder there's a notebook called **components_definition.ipynb** which should be executed to generate the .yamls to be invoked in the main notebook execution.
- Last, follow the steps in **pipeline_run.ipynb**. Some parameters, as hyperparameter trials or machine types for given steps of the process can be easily modified.

## To-do
estimated cost
roles