# Object Detection Using Tensorflow

## Installation
* Install Miniconda3 according to instruction for your OS which can be found 
[here](https://conda.io/docs/user-guide/install/index.html).
* Create a Virtual Environment with python 3.6
```` 
 conda create -n name_of_environemnt python=3.6
 
 ````
 * Activate environment
 ````
 source activate name_of_environment
 ````
  * Install Dependencies
  
    - For Training: <br/>
    [Tensorflow Object Detection Repository](https://github.com/tensorflow/models/tree/master/research/object_detection) <br/>
    And follow instruction for installing Tensorflow for GPU<br/>
    Clone Tensorflow Research Repository
    - For Detection <br/>
    We are going to use CPU Tesnsorflow.<br/>
```
# Upgrade pip first
pip install --upgrade pip
# Install requirements from requirements.txt
pip install -r requirements.txt
```

## Training 
     
  * Data Collection and Labeling<br/>
  We are going to need at least 1000 images of the object of in interest. <br/>
  I intend to use [LabelImg](https://github.com/tzutalin/labelImg) for all of the labeling.
  
  * Utilizing Transfer Learning <br/>
  We are going to use one of [these pretreated models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
  on our data set.
  
## Detection

  * Model <br/>
  
    - Frozen graph
    - Tensorflow serving
     
  * GUI <br/>
  
    - Kivy
    
    
 ## Usage of the current code
 
```
python detect_faces.py --video_file <path_to_the_video_file>
```