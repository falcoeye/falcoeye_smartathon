# Falcoeye Smartathon

## I.About:
#### Falcoeye Smartathon is a deeplearning based workflow meant for detecting and aggregating patholes severity in videos. 
#### This project is a submission for the [2023 SADAIA Smartathon Challenge Path 2](https://smartathon.hackerearth.com/). 
#### More information on [our team's project](put link here), [how to use the interface](put youtube video here).


## II.Usage:
#### Prerequisite:
>[python3.8](https://www.python.org/downloads/)


>[pip](https://pypi.org/project/pip/)


>[git](https://git-scm.com/downloads)


#### Dependencies:
***See [requirements.txt](https://github.com/falcoeye/falcoeye_smartathon/blob/main/python/requirements.txt)***


### To install the tool in your device:
##### 1- Clone the repository
##### 2- Create and activate virtual environment
##### 3- Install dependencies
##### 4- Run run.py

### 1- Clone this repository:
First make sure you should have [git](https://git-scm.com/downloads) installed.

on terminal or cmd:
Create or select path where this repository will be installed and clone the repository:
```
$ git clone git@github.com:falcoeye/falcoeye_smartathon.git
```

### 2- Create a virtual environment:

```
$ cd <cloning dir>/falcoeye_smartathon/python
$ python3.8 -m venv venv
$ source venv/bin/activate
```

### 3- Install all dependencies:
Using the `requirements.txt` file this command will install all Python libraries that the project depends on.
```
$ cd <cloning dir>/falcoeye_smartathon/python
$ pip install -r requirements.txt
```
#### ***Note: If you want to use the Inertial Stability feature, clone this two repositories; Mesh Reader:(https://github.com/p-hofmann/MeshReader.git), Voxlib: (https://github.com/p-hofmann/PyVoxelizer.git) into your environment.***

### 4- launch the tool:
Finally:
To run with video with GPS information provided in the frame
```
$ python run.py --checkpoints <checkpoint path> --file <mp4 or jpg file> --backbone <backbone name> --draw --video --output <where to store the csv data>
```

To run with video that doesn't have GPS information:
```
$ python run.py --checkpoints <checkpoint path> --file <mp4 or jpg file> --backbone <backbone name> --draw --video --output <where to store the csv data> --nogps
```
### 
