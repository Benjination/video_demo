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
./video_broadcast -speedClient
```

The IC780 camera at `/dev/video0` supports MJPEG `1920x1080` at 30 fps. Camera mode requests that full-HD input and scales it to the multicast output size. Preset `4` is `1536x864`, the largest 16:9 frame that fits in SISCI's 2 MiB segment limit on this system (a full `1920x1080` frame needs about 3.1 MB and makes segment creation fail with `Out of local resources`). Use the same size on the server and client. The default remains preset `2` (`854x480`) for compatibility with existing lower-resolution sessions.

Camera frames use the same raw-frame multicast buffer as file frames, so clients require no changes.

## Inference overlays

The server runs OpenCV face detection and draws green boxes around detected faces before broadcast. The client independently runs MediaPipe palm detection through OpenCV DNN and draws cyan boxes around detected hands. These are separate inference workloads and do not add data to the multicast protocol.

The client requires `models/palm_detection_mediapipe_2023feb.onnx` beside the executable. It is sourced from the Apache-2.0 licensed OpenCV Zoo hand model collection.

Ubuntu dependencies:

```
sudo apt install libopencv-dev opencv-data
```

## Badge text reader

The `-speedClient` mode reads printed badge text from the received video with Tesseract OCR and shows each recognized badge beside the video as name, company, and location. Reading runs on a background thread about twice per second, and a badge is saved only after it is read identically twice in a row, which filters out one-off misreads. Each unique badge (matched by name and company) is saved as a JSON file under `badges/visible/` containing the name, company, location, the raw lines that were read, and a timestamp. Clicking **Clear / Archive** moves the visible JSON records to `badges/stored/`. Existing records are loaded when the speed client starts. Use the mouse wheel to scroll the visible badge list.

The parser expects the badge layout used by the event badges tested so far: first name, full name, company (one to three lines), then `CITY, ST`, printed upright. Hold the badge upright, flat, and facing the camera. Other badge layouts may need changes to `parse_badge_lines` in `video_broadcast.cpp`.

Badges do not include email addresses or phone numbers, so saved records identify people by name and company only. Treat the saved files as personal information and keep them private.

Ubuntu dependencies:

```
sudo apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
```

## Other considerations:

Sometimes SDL will struggle to access the screen when running as `root` on the computer, to solve this, try running it as a user instead.

When you access the computer using ssh, X-forwarding will need to be used to get the output of the screen from the remote computer to your local computer.

If SCIPrepareSegment fails, the reason is typically that the wrong adapter is specified. The default is 0 and it can be changed with the `-a` parameter.
