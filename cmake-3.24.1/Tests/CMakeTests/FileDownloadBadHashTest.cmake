if(NOT "/home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/CMakeTests" MATCHES "^/")
  set(slash /)
endif()
set(url "file://${slash}/home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
