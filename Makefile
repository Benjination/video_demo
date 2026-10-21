
.PHONY: clean

FILES = video_broadcast

WARNINGS = -W -Wall -Wextra -Wpedantic

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
TESSERACT_CFLAGS = $(shell pkg-config --cflags tesseract)
TESSERACT_LIBS = $(shell pkg-config --libs tesseract lept)

INCLUDES = -I/opt/DIS/include -I/opt/DIS/include/dis -I/usr/include/ffmpeg/ $(OPENCV_CFLAGS) $(TESSERACT_CFLAGS)

FLAGS = -lSDL2 -lSDL2_ttf -lavdevice -lavformat -lavcodec -lswscale -lavutil -lz -L/opt/DIS/lib64 -lsisci -pthread $(OPENCV_LIBS) $(TESSERACT_LIBS)

all: $(FILES)

video_broadcast: video_broadcast.o
	g++ $< -o $@ $(FLAGS)

%.o: %.cpp
	g++ -std=c++17 $(WARNINGS) $(INCLUDES) -D_REENTRANT -c $< -o $@

clean:
	rm -f $(FILES) *.o
