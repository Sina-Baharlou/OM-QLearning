CXXFLAGS =	-std=c++0x -O2 -g -Wall -fmessage-length=0 -Wreorder -Wwrite-strings -Wsign-compare

OBJS =		OAgent.o QAgent.o QLearning.o RewardGenerator.o

LIBS =		-lopencv_core -lpthread -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

TARGET =	QLearning

LIBSDIR = 	

$(TARGET):	$(OBJS)
	$(CXX) $(LIBSDIR) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
