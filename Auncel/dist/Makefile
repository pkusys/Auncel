# Copyright (c) Zili Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

DIST_SRC=$(wildcard *.cpp)
DIST=$(DIST_SRC:.cpp=)


all: $(DIST)

clean:
	rm -f $(DIST)

%: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -lfaiss


.PHONY: all clean
