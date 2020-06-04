# Image Captioning

## Description
Our project aims at automatically generating a single sentence describing the content of
a previously unseen image. We run all the training and testing and validation on windows 10 with a GTX1060 GPU. 

## Requirements
For a windows platform, do the following:<br>
Install CUDA and CUDNN following the official instructions.<br>
Install MinGW following the official guide.<br>
Install Git following the official guide.<br>
Install Microsoft Visuall C++ Toolbox.<br>

## Code organization
demo.ipynb --Visualize the result sentence.<br>
EvalDemo.ipynb --Evaluate the result using bleu, meteor, rouge, cider, and spice metrics.<br>
build_vocab.py -- Generating vocab.pkl<br>
resize.py -- transform the original images to a common size.<br>
data_loader.py -- define the data structure.<br>
model.py -- define the encoder and the decoder.<br>
train.py -- Run the training process on the training set.<br>
val.py -- Run the test process and generate results on the validation set.<br>

## Guidance
    REMINDER: All the commands below should be run on cmd in the directory of WORKSPACE.

Here is an example:
>WORKSPACE
>>Image-Captioning
>>>build_cocab.py<br>
>>>resize.py<br>
>>>train.py<br>
>>>val.py<br>
>>>model.py<br>
>>>demo.ipynb<br>
>>>EvalCap.ipynb<br>
>>>data<br>
>>>>annotations<br>
>>>>>captions_train2014.json<br>
>>>>>captions_val2014.json<br>
>>>>vocab.pkl<br>
>>>>resized2014<br>
>>>>pretrained<br>

## Usage 

#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
```

To avoid issues on windows, do the following instead:
```
$ git clone https://github.com/philferriere/cocoapi.git
```
Then move on:
```
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone https://github.com/ufocjrufo/Image-Captioning.git
```

#### 2. Download the dataset

```bash
$ pip install -r requirements.txt --user
$ chmod +x download.sh
$ ./download.sh
```

#### 3. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
```

#### 4. Train the model

```bash
$ python train.py
```
Or you can specify the parameters such as:
```
$ python train.py --model_path=models/ --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.001 --num_epochs=3   
```

#### 5. Use the model to generate results

```bash
$ python val.py
```
Or you can specify the parameters such as:
```
$ python val.py --encoder_path=models3/encoder-3-3000.ckpt --decoder_path=models3/decoder-3-3000.ckpt --name=val3 
```

#### 5. Use metrics to evaluate the results

Run **EvalCap.ipynb**


## Run the demo
To directly run **demo.ipynb**, you need to finish the step 1 above first. Then do the following:<br><br>
Go to ***/Image-Captioning***,<br><br>
Check if ***encoder-5-3000.ckpt***(235MB) and ***decoder-5-3000.ckpt***(37MB) and ***data/annotations/captions_val2014.json***(32MB) are of the correct size. if not so, directly download ***encoder-5-3000.ckpt*** and ***decoder-5-3000.ckpt*** and ***captions_val2014.json*** from this website.( This is because sometimes downloading large files may fail.)<br><br>
Then run **demo.ipynb**.

## Run the EvalDemo
To directly run the **EvalDemo.ipynb**, you are required to install both java and python first. As we failed to install java on DSMLP, the result was actually run on our own PC. If you have both java and python, do the following to run this demo:<br><br> 
Go to ***/Image-Captioning***<br><br>
Check if ***val1_3_3000.json*** , ***val2.json*** , ***val3.json***, ***val4.json*** , ***val1_3_3000.json***  and ***val5.json*** are in the correct file path:  ***data/annotations/***<br><br>
Check if the ***/pycocoevalcap*** file contains all the required files. If not, (for Python 2)turn to the official coco eval github and clone the ***pycocoevalcap*** file. https://github.com/tylin/coco-caption. (for Python 3) You can refer to He Huang's ***coco-caption-py3***.
Then run the **demo.ipynb**


## Results

The scores are shown below:

| | Bleu_1 | Bleu_1  | Bleu_1 | Bleu_1  | METEOR | ROGUE_L | CIDEr | SPICE |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Train1 | 0.671 | 0.491 | 0.347 | 0.243 | 0.223 | 0.492 | 0.763 | 0.154 |
| Train2 | 0.677 | 0.491 | 0.343 | 0.239 | 0.223 | 0.491 | 0.765 | 0.152 |
| Train3 | 0.666 | 0.483 | 0.338 | 0.237 | 0.223 | 0.488 | 0.743 | 0.151 |
| Train4 | 0.662 | 0.481 | 0.338 | 0.237 | 0.222 | 0.487 | 0.736 | 0.149 |
| Train5 | 0.670 | 0.484 | 0.335 | 0.232 | 0.219 | 0.487 | 0.727 | 0.147 |
| Train6 | 0.677 | 0.491 | 0.343 | 0.239 | 0.223 | 0.491 | 0.765 | 0.152 |
| Train7 | 0.669 | 0.488 | 0.345 | 0.243 | 0.228 | 0.494 | 0.776 | 0.154 |



The configuration of models are shown below:

| | basic model | num_layers of LSTM | learning rate | dim of LSTM hidden states | num_epochs |
| - | - | - | - | - | - |
| Train1 | resnet152 | 1 | 0.001 | 512 | 3 |
| Train2 | resnet152 | 2 | 0.001 | 512 | 3 |
| Train3 | resnet101 | 1 | 0.001 | 512 | 3 |
| Train4 | resnet152 | 1 | 0.005 | 512 | 3 |
| Train5 | resnet152 | 1 | 0.001 | 256 | 3 |
| Train6 | resnet101 | 1 | 0.005 | 512 | 3 |
| Train1 | resnet152 | 1 | 0.001 | 512 | 5 |

## Reference

Some codes are borrowed from:<br>
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning.<br>
https://www.tensorflow.org/tutorials/text/image_captioning<br>
