CC      = g++
SRC     = *.cpp
LIBS    = -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
FOLDER  = bin/
ROOT    = ../
NAME    = Koivisto
MINOR   = 19
MAJOR   = 4
EXE     = $(ROOT)$(FOLDER)$(NAME)_$(MAJOR).$(MINOR)
ifeq ($(OS),Windows_NT)
    PREFIX := windows
    SUFFIX := .exe
else
    PREFIX := linux
    SUFFIX := 
endif

WFLAGS = -std=c++17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -DNDEBUG -flto -march=native
RFLAGS = -O3 $(WFLAGS) -DNDEBUG -flto -static
PFLAGS = -O0 $(WFLAGS) -DNDEBUG -p -pg
DFLAGS = -O0 $(WFLAGS) -g
EFLAGS = -O3 $(WFLAGS) -DNDEBUG -flto -static

EFLAGS += -s EXPORT_NAME="Koivisto" -s ENVIRONMENT=web,worker,node -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=1
EFLAGS += -s EXIT_RUNTIME=0 -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall']" --pre-js pre.js
EFLAGS += -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 -s MAXIMUM_MEMORY=2147483648
EFLAGS += -s FILESYSTEM=0 --closure 1


POPCNTFLAGS = -DUSE_POPCNT -mpopcnt
PEXTFLAGS   = $(POPCNTFLAGS) -DUSE_PEXT -mbmi2
AVX2FLAGS   = -msse -msse3 -mpopcnt -mavx2 -msse4.1 -mssse3 -msse2

MAKROS      = -DMINOR_VERSION=$(MINOR) -DMAJOR_VERSION=$(MAJOR)

openbench:
	$(CC) $(CFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) -o $(EXE)

native:
	mkdir -p $(ROOT)$(FOLDER)
	$(CC) $(CFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) -o $(EXE)-x64-$(PREFIX)-native$(SUFFIX)
    
release:
	mkdir -p $(ROOT)$(FOLDER)
	$(CC) $(CFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) -o $(EXE)-x64-$(PREFIX)-native$(SUFFIX)
	$(CC) $(RFLAGS) $(SRC) $(MAKROS) $(LIBS) -o $(EXE)-x64-$(PREFIX)$(SUFFIX)
	$(CC) $(RFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) -o $(EXE)-x64-$(PREFIX)-popcnt$(SUFFIX)
	$(CC) $(RFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) $(AVX) -o $(EXE)-x64-$(PREFIX)-popcnt-avx$(SUFFIX)

emscripten:
	mkdir -p $(ROOT)$(FOLDER)
	emcc $(EFLAGS) $(SRC) $(MAKROS) $(LIBS) $(POPCNTFLAGS) -o $(ROOT)$(FOLDER)Koivisto.js
