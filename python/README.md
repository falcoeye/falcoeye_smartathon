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

### 4- Download the pre-trained weights:

You need to download the pre-trained weights below and place them in the checkpoints folder
```
$ cd <cloning dir>/falcoeye_smartathon/python/
$ mkdir checkpoints
$ download and place the weights the checkpoints folder
```

The below pre-trained models were adapted from [Road Crack Detection Challange 2020](https://github.com/mahdi65/roadDamageDetection2020)


| Model                  	| Input Image Resolution 	| #params 
|------------------------	|------------------------	|---------
| D0 [checkpoint](https://drive.google.com/file/d/1TCs_snVnUcBAmovyuNfCc8pHk8uR1ENw/view?usp=sharing)      	| 512                    	| 3.9M
| D1 [checkpoint](https://drive.google.com/file/d/1iH0hp12yz80s7U1yZnr5Ac0l87RbGknv/view?usp=sharing)      	| 640                    	| 6.5M
| D2 [checkpoint](https://drive.google.com/file/d/1Nigyw8yvq5trj1P7IImgJK080jJOjkYK/view?usp=sharing)       	| 768                    	| 8M	|
| D3 [checkpoint](https://drive.google.com/file/d/1btcOiJ-Gz0uVFfawl8OJXWykYL9pdB3G/view?usp=sharing)       	| 796                    	| 11.9M	|
| D4 [checkpoint](https://drive.google.com/file/d/1IODGXThyH6dyahB2D3-bp73BebyxjMGc/view?usp=sharing)      	| 1024                   	| 20.5M
| D7 [checkpoint](https://drive.google.com/file/d/1FUk_cEyYX7hEeB_DfaEEKK-p6Ltquvdb/view?usp=sharing)  	    | 1536                   	| 51M

### 5- launch the tool:
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

