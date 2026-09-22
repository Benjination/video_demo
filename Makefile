
.PHONY: clean

FILES = video_broadcast

WARNINGS = -W -Wall -Wextra -Wpedantic

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

INCLUDES = -I/opt/DIS/include -I/opt/DIS/include/dis -I/usr/include/ffmpeg/ $(OPENCV_CFLAGS)

FLAGS = -lSDL2 -lavdevice -lavformat -lavcodec -lswscale -lavutil -lz -L/opt/DIS/lib64 -lsisci $(OPENCV_LIBS)

all: $(FILES)

video_broadcast: video_broadcast.o
	g++ $< -o $@ $(FLAGS)

%.o: %.cpp
	g++ $(WARNINGS) $(INCLUDES) -D_REENTRANT -c $< -o $@

clean:
	rm -f $(FILES) *.o
