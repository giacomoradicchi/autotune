cmake_minimum_required(VERSION 3.15)
project(AutotuneProject C)

if (WIN32)
    find_package(SDL2 REQUIRED)
    find_package(FFTW3 REQUIRED)
else()

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)


    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3)
endif()

include_directories(
    ${SDL2_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIR}
)

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
    target_link_libraries(autotune PRIVATE ${SDL2_LIBRARIES} ${FFTW3_LIBRARY} m)
endif()

#uso C99
set_property(TARGET autotune PROPERTY C_STANDARD 99)