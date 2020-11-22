cmake_minimum_required(VERSION 3.16)
project(Koivisto VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(Koivisto
        src_files/main.cpp
        src_files/Bitboard.h
        src_files/Bitboard.cpp
        src_files/Board.h
        src_files/Board.cpp
        src_files/Move.h
        src_files/Move.cpp
        src_files/TranspositionTable.h
        src_files/TranspositionTable.cpp
        src_files/Perft.h
        src_files/Perft.cpp
        src_files/Util.h
        src_files/Util.cpp
        src_files/search.cpp
        src_files/search.h
        src_files/uci.cpp
        src_files/uci.h
        src_files/eval.cpp
        src_files/eval.h
        src_files/MoveOrderer.cpp
        src_files/MoveOrderer.h
        src_files/History.cpp
        src_files/History.h
        src_files/Verification.h
        src_files/Tuning.cpp
        src_files/Tuning.h
        src_files/TimeManager.cpp
        src_files/TimeManager.h
        src_files/syzygy/stdendian.h
        src_files/syzygy/tbprobe.h
        src_files/syzygy/tbconfig.h
        src_files/syzygy/tbprobe.c
        src_files/syzygy/tbchess.c
        src_files/fecppnn/config.h
        src_files/fecppnn/Layer.h
        src_files/fecppnn/Data.h
        src_files/fecppnn/Data.cpp
        src_files/fecppnn/activations.h
        src_files/fecppnn/activations.cpp
        src_files/fecppnn/affine.h
        src_files/fecppnn/Network.h
        src_files/fecppnn/structure.h
        src_files/fecppnn/transform.h
        src_files/fecppnn/transform.cpp
        src_files/fecppnn/sample.h
        src_files/fecppnn/Network.cpp src_files/fecppnn/affine.cpp src_files/fecppnn/loss.h src_files/fecppnn/loss.cpp src_files/fecppnn/optimiser.cpp src_files/fecppnn/optimiser.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMINOR_VERSION=7")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAJOR_VERSION=4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")
set(CMAKE_EXE_LINKER_FLAGS " -static")

target_link_libraries(Koivisto -static-libgcc -static-libstdc++)
