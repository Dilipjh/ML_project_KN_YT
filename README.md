# ML_project_KN_YT

Hands on end to end ML project

## Agenda

1. setting up the project with git hub for collaboration
2. structuring the project
3. data ingestion
4. Data transformation
5. Model training
6. model evaluation
7. model deployment - in AWS
8. CI/CD pipelines - github actions

### structuring the project

1. setting up project repo in Git
2. initiating the repo
3. setting up a environment using venv
4. requirements.txt file
5. setup.py
   - with functoin to read the requiremnts.txt and update it source file
   - packaging the entire project
   - building the package
6. components folder for modular code containing functions collated
   - data ingestion
   - data transforing (wrangling)
   - model trainer
   - model pusher ( to push model to cloud)
7. Pipeline folder containing
   - model training
   - model prediction
8. settign up logging in logger.py with custom log details
9. setting up custom exception handling
   - used inheritance
