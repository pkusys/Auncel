# Copyright (c) Zili Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

EVAL_SRC=$(wildcard *.cpp)
EVAL=$(EVAL_SRC:.cpp=)


all: $(EVAL)

clean:
	rm -f $(EVAL)

%: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -lfaiss


.PHONY: all clean