## Dataset Setup
Download [CULane](https://xingangpan.github.io/projects/CULane.html) and [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). 
Then extract them to `$CULANEROOT` and `$TUSIMPLEROOT`. 
The Tusimple directory should look like:
```
$TUSIMPLEROOT
|──clips
|──label_data_0313.json
|──label_data_0531.json
|──label_data_0601.json
|──test_tasks_0627.json
|──test_label.json
|──readme.md
```
The CULane directory should look like:
```
$CULANEROOT
|──driver_100_30frame
|──driver_161_90frame
|──driver_182_30frame
|──driver_193_90frame
|──driver_23_30frame
|──driver_37_30frame
|──laneseg_label_w16
|──list
```
Download our simulation data [WATO](https://drive.google.com/drive/folders/1ZPO_e_gMUXdWNgxMJWdAgn9iOOgV8qUW?usp=sharing)
and unzip to `$WATOROOT`.
The WATO directory should look like:
```
$WATOROOT
|──Town01_000000.jpg
|──Town01_000000.json
|──Town01_000001.jpg
|──Town01_000001.json
|──...
```

### TuSimple Class Label Setup
To setup the TuSimple dataset with classes, download the json files from this 
[Google Drive](https://drive.google.com/drive/folders/1sMA7pdknqwRunami5ZWdmZlAsSoM44Cv?usp=sharing) 
and place them in the TuSimple root folder. The class labels were downloaded from 
[TuSimple-lane-classes](https://github.com/fabvio/TuSimple-lane-classes) and converted to json using the given 
converter script.

### TuSimple/WATO Format Conversion

TuSimple has a unique structure for data and labels. There is also no segmentation labelling.
TuSimple must therefore be reformatted to have a similar structure to CULane data. Additionally,
segmentation labelling should be generated from the available data, to use segmentation loss.

This is accomplished by running the following script within the docker container:
```
python data/convert_tusimple_format.py --dataset TuSimple --root /datasets/TuSimple
```
This command must be run before any training or evaluation; the training and testing
scripts expect the TuSimple data to already be reformatted.

Similarly the WATO data also needs to be reformatted since it uses the same label format as TuSimple 
```
python data/convert_tusimple_format.py --dataset WATO --root /datasets/WATO_TuSimple
```
For image size that is not 720x1280, for example for training with CULane
```
python data/convert_tusimple_format.py --dataset WATO --root /datasets/WATO_CULane --res 590x1640
```