# Project MLOps - Niels Ornelis

# Dataset & Project Description

---

## Dataset

For this project, I have chosen the "Fruits and Vegetables Image Recognition Dataset," which can be accessed [here](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition). Upon downloading the data, you will receive a folder named "Food." Inside this folder, there are three subfolders: "train," "test," and "validation." Within these subfolders, you'll find additional subfolders named after specific types of fruits or vegetables (the classes). In total there are 36 classes.

## ****Project Description****

The project involves a machine learning pipeline for classifying food images. The workflow includes data preparation, model training, and deployment. This is done using Azure Machine Learning and GitHub Actions for continuous integration and continuous deployment (CI/CD).

# Pipeline

---

## Intro

An Azure ML pipeline is defined in **`./pipelines/food-classification.yaml`**. This pipeline integrates all steps from data preparation to model registration. It includes tasks such as data preparation, data splitting, model training, and the registration of trained models.

- This is a picture of how the pipeline looks in Azure.
    
    ![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled.png)
    

- This is the pipeline file called **`food-classification.yaml`**
    
    ```yaml
    $schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
    
    name: food-classification
    type: pipeline
    display_name: Food Classification
    experiment_name: classification
    
    inputs:
      training_test_split_factor: 20
      epochs: 100
    outputs:
      model:
        type: uri_folder
    
    settings: 
      default_compute: azureml:cli-MLops-compute
    
    jobs:
      data_prep:
        type: command
        component: ../components/dataprep/dataprep.yaml
    
        inputs:
          data:
            type: uri_folder
            path: azureml:food:1
    
        outputs:
          output_data:
            mode: rw_mount
        
      data_split:
        type: command
        component: ../components/dataprep/data_split.yaml
    
        inputs:
          food_data: ${{parent.jobs.data_prep.outputs.output_data}}
          train_test_split_factor: ${{parent.inputs.training_test_split_factor}}
        outputs:
          training_data:
            mode: rw_mount
          testing_data:
            mode: rw_mount
    
      training:
        type: command
        component: ../components/training/training.yaml
    
        inputs:
          training_folder: ${{parent.jobs.data_split.outputs.training_data}}
          testing_folder: ${{parent.jobs.data_split.outputs.testing_data}}
          epochs: ${{parent.inputs.epochs}}
        outputs:
          output_folder:
            mode: rw_mount
    
      register_model:
        type: command
        component: azureml://registries/azureml/components/register_model/versions/0.0.9
    
        inputs:
          model_name: food-classification
          model_type: custom_model
          model_path: ${{parent.jobs.training.outputs.output_folder}}
        outputs:
          registration_details_folder: ${{parent.outputs.model}}
    ```
    

## Components

As seen above, there are four components in this pipeline. I will provide further explanation for each of these four components to give a clear understanding of their respective functionalities.

### Data Preparation

The first component reads and processes the data. As I mentioned before there is quite a folder structure, to make it easier I just combined all pictures. Otherwise I would’ve had a complicated structure in my pipeline file with a lot of inputs for the different classes. Now I just have 1 input folder instead of 36.  In the picture below you can see what I mean.

```yaml
jobs:
  data_prep:
    type: command
    component: ../components/dataprep/dataprep.yaml

    inputs:
      data:
        type: uri_folder
        path: azureml:food:1

    outputs:
      output_data:
        mode: rw_mount
```

So this component first it iterates trough the folders train, test, validation.  Then it iterates trough the class folders and reads the images. The component first resizes every image to 100x100 pixels. Then it flips every resized image for extra augmentation.  Later in the Train component there is an additional image augmentation.

### Data Split

Here's a breakdown of what the component does:

- The component iterates through each dataset (in my case there is only one dataset because I put everything together), subfolder, and class folder to gather a list of image paths.
- Images are randomly shuffled, and a portion is set aside for testing based on the specified split size.
- Then the selected testing and training images are copied to the respective output directories.
- Each image is renamed to include the class name to make it easier to train the model with class names.

### Train

This component creates a convolutional neural network. Here is a short breakdown on how it’s done:

- Loads image paths from the specified folders, shuffles them
- Parses image paths into features and targets, one-hot encodes the labels, and prepares the data for training.
- Builds and compiles a CNN model using the specified architecture and optimizer.
- Initializes an image data generator for data augmentation.
- Trains the model using the augmented data, saves the best model, and logs training metrics.
- Evaluates the trained model on the testing set and prints a classification report and confusion matrix.
- Saves the confusion matrix as a NumPy file in the output folder.
- The [utils.py](http://utils.py) contains utility functions for processing image data, encoding labels, and building the CNN model architecture.

### Register Model

The Register Model component is a pre-made Azure component that handles the deployment of the trained model to Azure.

# Github Actions

---

### **Workflow Overview**

The GitHub Actions workflow, named "Azure data preparation, training, and deployment," is designed to automate various tasks related to Azure Machine Learning. The workflow includes several jobs, each serving a specific purpose in the end-to-end ML process.

- This is the workflow file:
    
    ```yaml
    name: Azure data preperation, training and deployment
    
    on:
      workflow_dispatch:
        inputs:
          create_workspace_and_environment:
            description: 'Create workspace, environment, components'
            required: false
            type: boolean
            default: false
          create_compute:
            description: 'Create compute'
            required: false
            type: boolean
            default: true
          train_model:
            description: 'Train model'
            required: false
            type: boolean
            default: true
          delete_compute:
            description: 'Delete compute'
            required: false
            type: boolean
            default: true
          deploy_model:
            description: 'Deploy model'
            required: false
            type: boolean
            default: true
    
    env:
      GROUP: MLops
      WORKSPACE: ornelis-niels-ml
      LOCATION: westeurope
      CREATE_WORKSPACE_AND_ENVIRONMENT: ${{ github.event.inputs.create_workspace_and_environment }}
      CREATE_COMPUTE: ${{ github.event.inputs.create_compute }}
      DELETE_COMPUTE: ${{ github.event.inputs.delete_compute }}
      TRAIN_MODEL: ${{ github.event.inputs.train_model }}
      DEPLOY_MODEL: ${{ github.event.inputs.deploy_model }}
    
    jobs:
    
      check-and-create-environments:
        runs-on: ubuntu-latest
        steps:
          - name: 'Checkout repository'
            uses: actions/checkout@v4
    
          - name: 'Az CLI login'
            uses: azure/login@v1
            with:
              creds: ${{ secrets.AZURE_CREDENTIALS }}
    
          - name: 'Check if AML Workspace exists'
            if: ${{ inputs.create_workspace_and_environment }}
            id: check-aml-workspace
            run: |
              az extension add --name ml -y
              az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
              WORKSPACE_EXISTS=$(az ml workspace show --query 'id' -o tsv)
              echo "Workspace exists: $WORKSPACE_EXISTS"
              if [ -z "$WORKSPACE_EXISTS" ]; then
                echo "Creating Azure ML workspace..."
                az ml workspace create -n $WORKSPACE -g $GROUP
              fi
    
          - name: 'Check and Create Environments'
            uses: azure/CLI@v1
            id: check-and-create-environments
            if: ${{ steps.check-aml-workspace.outcome == 'success' }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name azure-cli-ml -y
                az ml folder attach -w $WORKSPACE -g $GROUP
    
                check_and_create_environment() {
                  ENV_NAME=$1
                  ENV_FILE=$2
    
                  if ! az ml environment show --name $ENV_NAME --query "name" -o tsv; then
                    echo "Creating Azure ML environment: $ENV_NAME with $ENV_FILE"
                    az ml environment create --file $ENV_FILE
    
                  else
                    echo "Azure ML environment $ENV_NAME with $ENV_FILE already exists."
                  fi
                }
    
                check_and_create_environment "aml-Pillow-cli-project" "./environments/pillow.yaml"
                check_and_create_environment "aml-Tensorflow-cli-project" "./environments/tensorflow.yaml"
    
          - name: 'Check and Create Components'
            uses: azure/CLI@v1
            if: ${{ steps.check-and-create-environments.outcome == 'success' }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name azure-cli-ml -y
                az ml folder attach -w $WORKSPACE -g $GROUP
    
                check_and_create_component() {
                  COMPONENT_NAME=$1
                  COMPONENT_FILE=$2
                  COMPONENT_VERSION=$3
    
                  if ! az ml component show --name $COMPONENT_NAME --version $COMPONENT_VERSION --query "name" -o tsv; then
                    echo "Creating Azure ML component: $COMPONENT_NAME with version $COMPONENT_VERSION and $COMPONENT_FILE"
                  else
                    echo "Azure ML component $COMPONENT_NAME with version $COMPONENT_VERSION and $COMPONENT_FILE already exists."
                  fi
                }
    
                check_and_create_component "data_prep_image_resize_augment" "./components/data_prep_image_resize_augment.yaml" "1.0.0"
                check_and_create_component "data_split" "./components/data_split.yaml" "0.1.0"
                check_and_create_component "training" "./components/training.yaml" "0.1.0"
    
      azure-pipeline:
        runs-on: ubuntu-latest
        needs: check-and-create-environments
        if: ${{ needs.check-and-create-environments.result == 'success' }} || ${{ needs.check-and-create-environments.result == 'skipped' }}
        steps:
          - name: 'Checkout repository'
            uses: actions/checkout@v4
    
          - name: 'Az CLI login'
            uses: azure/login@v1
            with:
              creds: ${{ secrets.AZURE_CREDENTIALS }}
          
          - name: 'Create compute'
            uses: azure/CLI@v1
            id: azure-ml-compute
            if: ${{ inputs.create_compute }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                az ml compute create --file ./environments/compute.yaml
    
          - name: 'Start compute'
            uses: azure/CLI@v1
            id: azure-ml-compute-start
            if: ${{ steps.azure-ml-compute.outcome == 'skipped' }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                az ml compute start --name cli-MLops-compute
            continue-on-error: true
          
          - name: 'Run pipeline'
            uses: azure/CLI@v1
            id: azure-ml-pipeline
            if: ${{ inputs.train_model }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                az ml job create --file ./pipelines/food-classification.yaml --set name=food-classification-${{ github.sha }}-${{ github.run_id }} --stream
                echo "Pipeline Done"
                VERSION=$( az ml model list -n food-classification --query '[0].version' )
                echo "Latest version of model is: $VERSION"
                echo "::set-output name=latest_version::$VERSION"
    
          - name: 'Cleanup Azure delete compute'
            uses: azure/CLI@v1
            id: azure-ml-compute-delete
            if: ${{ inputs.delete_compute }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                az ml compute delete --name cli-MLops-compute --yes
            continue-on-error: true
    
          - name: 'Cleanup Azure stop compute'
            uses: azure/CLI@v1
            id: azure-ml-compute-stop
            if: ${{ steps.azure-ml-compute-delete.outcome == 'skipped' }}
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                az ml compute stop --name cli-MLops-compute
            continue-on-error: true
      download:
        runs-on: ubuntu-latest
        needs: azure-pipeline
        if: ${{ needs.azure-pipeline.result == 'success' }} || ${{ needs.azure-pipeline.result == 'skipped' }} 
        steps:
    
          - name: 'Checkout repository'
            uses: actions/checkout@v4
    
          - name: 'AZ CLI login'
            uses: azure/login@v1
            with:
              creds: ${{ secrets.AZURE_CREDENTIALS }}
    
          - name: 'Download model'
            uses: azure/CLI@v1
            with:
              azcliversion: 2.53.0
              inlineScript: |
                az extension add --name ml -y
                az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
                VERSION=$(az ml model list -n food-classification --query '[0].version')
                version=$(echo $VERSION | tr -d '"')
                echo "Latest version of model is: $version"
                echo "::set-output name=latest_version::$version"
                az ml model download --name food-classification --download-path ./inference --version $version
          
          - name: 'Upload api code'
            uses: actions/upload-artifact@v2
            with:
              name: docker-config
              path: inference
          
      deploy:
        runs-on: ubuntu-latest
        needs: download
        if:  ${{ ( needs.download.result == 'success'  ||  needs.download.result == 'skipped'  ) &&  inputs.deploy_model  }} 
        steps:
          - name: 'AZ CLI login'
            uses: azure/login@v1
            with:
              creds: ${{ secrets.AZURE_CREDENTIALS }}
          
          - name: 'Gather Docker Meta Information'
            id: docker-metadata
            uses: docker/metadata-action@v3
            with:
              # list of Docker images to use as base name for tags
              images: |
                ghcr.io/ornelisniels/mlops-food-api
              tags: |
                type=ref,event=branch
                type=sha
    
          # Enter your GITHUB Token here!
          - name: Login to GHCR
            uses: docker/login-action@v1
            with:
              registry: ghcr.io
              username: ornelisniels
              password: ${{ secrets.TOKEN_GITHUB }}
    
          # Download artifacts
          - name: Download API code for Docker
            uses: actions/download-artifact@v2
            with:
              name: docker-config
              path: inference
    
          - name: 'Debug: List files in ./inference'
            run: ls ./inference
    
          - name: Docker Build and push
            id: docker_build
            uses: docker/build-push-action@v2
            with:
              context: ./inference
              push: true
              tags: ${{ steps.docker-metadata.outputs.tags }}
    ```
    

### **Triggers**

The workflow is triggered manually through the GitHub Actions using the **`workflow_dispatch`** event. Upon manual triggering, you have the option to set input parameters to do the following things:

- Create a workspace and environment
- Create compute (When not creating a compute, compute starts instead of creates)
- Train a model
- Delete compute
- Deploy a model.

### **Environment Variables**

The workflow utilizes environment variables to customize the workflow behaviour. These variables include Azure resource group (**`GROUP`**), workspace name (**`WORKSPACE`**), location (**`LOCATION`**), and flags for each step in the workflow.

## **Jobs**

### **1. Check and Create Workflow, Environments & Components**

This job serves as the foundational step for the entire workflow, focusing on the creation and verification of essential Azure ML elements.

- **Azure ML Workspace:**
    - Checks if the specified workspace exists.
    - Creates the workspace if not present.
- **Environments:**
    - If the workspace is successfully established:
        - Verifies the existence of ML environments.
        - Creates environments that are not yet defined.
- **Components:**
    - If the environments are ready:
        - Checks for the presence of ML components.
        - Creates missing components.

This sequential approach ensures the Azure ML workspace, environments, and components are in place for next workflow stages.

### **2. Azure ML Pipeline**

This job has the following steps:

- **Environment Setup:**
    - Utilizes the 'Checkout repository' action to retrieve the workflow.
    - Performs Azure CLI login using credentials stored in GitHub secrets.
- **Compute Creation and Activation:**
    - Checks if compute creation is required (**`create_compute`** input).
        - If true, creates the compute using the specified YAML file.
        - If false, starts the compute instead of creating it.
- **Pipeline Execution:**
    - Checks if training is true (**`train_model` input)**
    - If true, executes the Azure ML pipeline job by creating a job instance using the **`food-classification.yaml`**.
- **Compute Cleanup:**
    - Checks if compute deletion is true(**`delete_compute`** input).
        - If true, deletes the Azure ML compute instance after the pipeline execution.
        - If false, stops the compute instead of deleting it.

### **3. Model Download and API Code Upload**

This job focuses on downloading the latest version of the trained model from Azure ML and preparing the API code for deployment.

- **Repository and Azure CLI Login:**
    - Same as before
- **Download Latest Model:**
    - Retrieves the latest version of the model from the Azure ML workspace.
    - Downloads the model files to the local directory specified as **`./inference`**.
- **API Code Upload:**
    - Uses the 'Upload artifact' action to package the API code (located in the **`./inference`** directory) for next steps.
    - The packaged artifact is named 'docker-config' for easy reference in following job.

### **4. Model Deployment with Docker**

This job is responsible for deploying the trained model as an API using Docker. It relies on the artifacts prepared in the previous 'Model Download and API Code Upload' job.

- **Azure CLI Login:**
    - Same as before
- **Gather Docker Meta Information:**
    - Specifies the base name for Docker images and defines tags based on events (e.g., branch and commit SHA).
- **Login to GitHub Container Registry (GHCR):**
    - Employs 'docker/login-action' to authenticate with GitHub Container Registry (GHCR) using the GitHub token stored in secrets.
- **Download API Code for Docker:**
    - Uses the 'Download artifact' action to retrieve the packaged API code from the 'docker-config' artifact created in the previous job.
    - The downloaded content is stored in the **`./inference`** directory.
- **Docker Build and Push:**
    - Utilizes 'docker/build-push-action' to build a Docker image from the API code and push it to the specified container registry.
    - The image is tagged based on the gathered metadata.

# FastApi

---

### Dockerfile

- Defines the environment for the FastAPI project.
- Specifies dependencies in **`requirements.txt`**.
- Copies project files into the Docker image.
- Sets the command to run the FastAPI app using Uvicorn on host 0.0.0.0 and port 8000.

### docker-compose.yaml

- Composes the Docker deployment, specifying service details.
- Builds the Docker image using the Dockerfile.
- Maps port 8001 on the host to port 8000 in the container.
- Names the Docker image as **`nielsornelis/mlops-food-api`**.

### main.py

- Imports necessary libraries and modules.
- Creates a FastAPI app.
- Adds CORS middleware to handle cross-origin resource sharing.
- Defines a list of food categories (**`FOODS`**) for classification.

```python
FOODS = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
 'turnip', 'watermelon']
```

- Loads a pre-trained deep learning model using TensorFlow/Keras.

```python
model_path = os.path.join('food-classification', 'INPUT_model_path', 'food-cnn')
model = load_model(model_path)
```

- **Routes:**
    - **GET Route ("/"):**
        - Returns a simple message when the root endpoint is accessed.
    - **POST Route ("/upload/image"):**
        - Handles image uploads using FastAPI's **`UploadFile`**.
        - Processes the image for prediction by resizing it to 100x100px and converting image from png to jpg if needed.
        - Uses a pre-trained model to predict the food category.
        - Returns the predicted food category.
    - **GET Route ("/healthcheck"):**
        - Provides a health check endpoint returning a status message.

```python
@app.get("/")
async def root():
    return {"message": "Goeindag"}

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.resize((100, 100))
    # If image is png, convert it to jpg
    if original_image.mode == 'RGBA':
        original_image = original_image.convert('RGB')
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classification = predictions.argmax(axis=1)

    return FOODS[classification.tolist()[0]]

@app.get("/healthcheck")
def healthcheck():
    return {"status": "Healthy"}
```

## Fictional Company

My fictional company specializes in developing applications for food recognition and classification. The AI model described in this document, which performs food classification for fruits and vegetables, can be seamlessly integrated into various services and existing software within the company.

The model can be integrated into existing systems used by food retailers or suppliers. For example, the model can be incorporated into inventory management software to automatically classify and categorize fruits and vegetables based on their images.

# Automation examples

As mentioned before these are some automations:

- Create a workspace and environment → Automatically creating workspace, envs, components if necessary.
- Create compute → When not creating a compute, compute starts instead of creates
- Delete compute → When not deleting a compute, compute stops instead of deletes

For more information go back to the Job sections in Github Actions sections

# Additional screenshots

---

## Model

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%201.png)

Accuracy:

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%202.png)

## Environments

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%203.png)

## Jobs

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%204.png)

## Components

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%205.png)

## Github Actions

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%206.png)

Note: Here the model was already trained so pipeline didn’t take long to complete.

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%207.png)

## FastApi

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%208.png)

![Untitled](Project%20MLOps%20-%20Niels%20Ornelis%205e363c0249e64cdfadd1b4633c6e32fc/Untitled%209.png)