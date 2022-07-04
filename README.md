# Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training

This is the repository for source code of "L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training".

The source code consists of independent encoder, decoder module, header file, and patch file for NVIDIA DALI.

The current version is a prototype version, and many parts are not automated in various aspects:
* To encode the image, the data of R, G, and B channels of the raw image must be separately saved.
* Manually, the file to be encoded or decoded must be specified in the "encoder_main.cu" or "decoder_main.cu" file.
* Manually, the factor "N" must be defined to determine the number of patches in "l3.cuh".
* L3 format can encode and decode the RGB-formatted image only.

## Prepare
Before the using L3 encoder, the user needs to prepare the raw data of R, G, and B channel of image from
other image formats. For example, the user can use **PIL** and **numpy** package on Python3 by folloing commands:
```
>>> from PIL import Image
>>> import numpy as np

>>> im = Image.open("some_example_your_image.png")
>>> pixels = np.array(im)

# Save "R" channel data to file
>>> pix_r = pixels[:, :, 0]
>>> pix_r.tofile(open("pixel_r.dat", "wb"))

# Save "G" channel data to file
>>> pix_g = pixels[:, :, 1]
>>> pix_g.tofile(open("pixel_g.dat", "wb"))

# Save "B" channel data to file
>>> pix_b = pixels[:, :, 2]
>>> pix_b.tofile(open("pixel_b.dat", "wb"))

>>> im.close()
```

## Encoding
Before the encoding is started, the user makes sure that the input files (raw data of R, G, and B) and output file path are predefined in ***encoder/encoder_main.cu***, and factor ***N*** is defined in ***l3.cuh***
```
$ cd src/
$ nvcc encoder/encoder.cu encoder/encoder_main.cu -o encoder_test
$ ./encoder_test
```

## Decoding
Before the decoding is started, the user makes sure that the input files (L3-encoded data) is predefined in ***decoder/decoder_main.cu***.
```
$ cd src/
$ nvcc decoder/decoder.cu decoder/decoder_main.cu -o decoder_test
$ ./decoder_test
```

## L3 with NVIDIA DALI
We support Github patch file to use L3 decoder with DALI on version 1.1.0 (commit number: 25b99fa703e4971906321e9360e357d74975de6e).

To apply the patch file to DALI:
```
# Download the DALI version 1.1.0
$ git clone -b release_v1.1 https://github.com/NVIDIA/DALI.git
$ cd $DALI_HOME

# Test patch command
$ patch -p1 --dry-run < ${L3 directory}/src/nvidia-dali/l3-integrated-dali.patch

# Apply patch to DALI
$ patch -p1 < ${L3 directory}/src/nvidia-dali/l3-integrated-dali.patch
```

## Citation
Please cite the following paper if you use L3:

**L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training.** Jonghyun Bae, Woohyeon Baek, Tae Jun Ham, and Jae W. Lee. In _European Conference on Computer Vision_, October 2022.

~~~
@inproceedings {XXXXXX,
  author = {Jonghyun Bae and Woohyeon Baek and Tae Jun Ham and Jae W. Lee},
  title = {L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training},
  booktitle = {European Conference on Computer Vision ({ECCV} 22)},
  year = {2022},
  publisher = {European Computer Vision Association},
  month = oct,
  address = {Tel-Aviv, Israel},
}
~~~
