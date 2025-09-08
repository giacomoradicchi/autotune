cmake_minimum_required(VERSION 3.15)
project(AutotuneProject C)

find_package(raylib REQUIRED)
find_package(FFTW3 REQUIRED)

set(SOURCES
    main.c
    autotune.c
)

add_executable(autotune ${SOURCES})
target_link_libraries(autotune PRIVATE raylib FFTW3::fftw3)
#uso C99
set_property(TARGET autotune PROPERTY C_STANDARD 99)