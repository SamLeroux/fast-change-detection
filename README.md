# Fast-change-detection

## Requirements:

You will need to clone the tensorflow repository: https://github.com/tensorflow/models
Install the dependencies: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
Install the xElementTree dependency via pip

How to run:

Copy and paste the file generate_xml.py and visualization_utils.py into the research/object_detection/utils in the tensorflow repo.
Add your pre-treined model and label map into the 'graphs' folder.
Add the images you want to label into the images folder
Change the xml path in generate_xml.py to put your own local path.
Inside the auto_annotate folder run: python3 scripts/detection_images.py
If everything is ok you will see the inference results and the xml in your respective folders!
If you have trouble or doubt see my tutorial on medium: https://medium.com/@alvaroleandrocavalcante/auto-annotate-images-for-tensorflow-object-detection-19b59f31c4d9?sk=0a189a8af4874462c1977c6f6738d759

## Download:

### data <br/>
-Florida:https://drive.google.com/file/d/1dmw-K9116v5Wc679yCNY4YpGZhZRzI4m/view?usp=sharing <br/>
-Michigan:https://drive.google.com/drive/folders/1abREXd2J66pphjYikLCMC5fSf3bgzuvI?usp=sharing

### graphs  <br/>
link:https://drive.google.com/file/d/1owk-LgRfsa2dSB8_pCzcGW4bpeJNmnoS/view?usp=sharing

### faster-rcnn frozen model <br/>
link:https://drive.google.com/file/d/11WAVs53dwKzbiFlkTdUZYO83ade5VIOU/view?usp=sharing
