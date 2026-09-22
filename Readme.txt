## To use video broadcast program you need the following libraries


### Ubuntu

```
apt-get install libsdl2-dev
apt-get install libavdevice-dev
```

### CentOS

SDL:

```
dnf --enablerepo=powertools install SDL2
dnf --enablerepo=powertools install SDL2-devel
```

Libav:

Follows instructions from: https://www.benholcomb.com/ffmpeg-on-rhel8/

```
dnf -y install https://download.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm
subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms
sudo dnf install ffmpeg
sudo dnf install ffmpeg-devel
```


## Program behavior

The video server multicasts a video using a given multicast group while simultaneously showing it to the screen. The client program sets up a multicast segment and shows it repeatedly to the screen. If you run the program with the `-help` parameters, or with no parameters, a short description of how to use the program will appear.

The server can use either an MP4 or another FFmpeg-supported video file with `-video`, or a V4L2 webcam device with `-camera`. For example:

```
./video_broadcast -server -camera /dev/video0
```

Camera mode requests an MJPEG `1280x720` feed at 30 fps, which is supported by the default output frame size limits.

Camera frames use the same raw-frame multicast buffer as file frames, so clients require no changes.

## Inference overlays

The server runs OpenCV face detection and draws green boxes around detected faces before broadcast. The client independently runs MediaPipe palm detection through OpenCV DNN and draws cyan boxes around detected hands. These are separate inference workloads and do not add data to the multicast protocol.

The client requires `models/palm_detection_mediapipe_2023feb.onnx` beside the executable. It is sourced from the Apache-2.0 licensed OpenCV Zoo hand model collection.

Ubuntu dependencies:

```
sudo apt install libopencv-dev opencv-data
```

## Other considerations:

Sometimes SDL will struggle to access the screen when running as `root` on the computer, to solve this, try running it as a user instead.

When you access the computer using ssh, X-forwarding will need to be used to get the output of the screen from the remote computer to your local computer.

If SCIPrepareSegment fails, the reason is typically that the wrong adapter is specified. The default is 0 and it can be changed with the `-a` parameter.

