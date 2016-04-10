CXXFLAGS   := -DNO_ECLIPSE -std=c++11 -I/usr/include/eigen3 -O2 -g -c -Winvalid-pch -fpch-preprocess
CXXFLAGS_H := -DNO_ECLIPSE -std=c++11 -I/usr/include/eigen3 -O2 -g
LDFLAGS    := -O2

HEADERS    := $(shell ls src/*.h)
OBJECTS    := $(shell ls src/*.cpp | sed -e 's,src/,build/,' -e 's,.cpp$$,.o,')

PROJECT    := eigen_versuche

.PHONY: all clean

all: build/${PROJECT} | build

clean:
	rm -f src/libs.h.gch
	rm -f -R build

build:
	mkdir -p build

build/%.o: src/%.cpp src/libs.h.gch ${HEADERS} | build
	g++ ${CXXFLAGS} $< -o $@

build/${PROJECT}: ${OBJECTS}
	g++ ${LDFLAGS} ${OBJECTS} -o build/${PROJECT}

src/libs.h.gch: src/libs.h
	g++ ${CXXFLAGS_H} src/libs.h -o src/libs.h.gch


