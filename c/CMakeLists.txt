cmake_minimum_required(VERSION 3.15)
project(AutotuneProject C)

find_package(SDL2 REQUIRED)
find_package(FFTW3 REQUIRED)

set(SOURCES
    main.c
    autotune.c
    draw.c
)

add_executable(autotune ${SOURCES})
if (WIN32)
    # SDL2main contiene il wrapper che chiama main()
    target_link_libraries(autotune PRIVATE SDL2::SDL2main SDL2::SDL2 FFTW3::fftw3)
else()
    target_link_libraries(autotune PRIVATE SDL2::SDL2 FFTW3::fftw3)
endif()

#uso C99
set_property(TARGET autotune PROPERTY C_STANDARD 99)