include("${CMAKE_CURRENT_LIST_DIR}/PackageSetup.cmake")

# Configuring pkgconfig

# We need two different instances of OpenCL.pc
# One for installing (cmake --install), which contains CMAKE_INSTALL_PREFIX as prefix
# And another for the Debian development package, which contains CPACK_PACKAGING_INSTALL_PREFIX as prefix

join_paths(OPENCL_INCLUDEDIR_PC "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")
join_paths(OPENCL_LIBDIR_PC "\${exec_prefix}" "${CMAKE_INSTALL_LIBDIR}")

set(pkg_config_location ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
set(PKGCONFIG_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Configure and install OpenCL.pc for installing the project
configure_file(
  OpenCL.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig_install/OpenCL.pc
  @ONLY)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig_install/OpenCL.pc
  DESTINATION ${pkg_config_location}
  COMPONENT pkgconfig_install)

# Configure and install OpenCL.pc for the Debian package
set(PKGCONFIG_PREFIX "${CPACK_PACKAGING_INSTALL_PREFIX}")
configure_file(
  OpenCL.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig_package/OpenCL.pc
  @ONLY)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig_package/OpenCL.pc
  DESTINATION ${pkg_config_location}
  COMPONENT dev
  EXCLUDE_FROM_ALL)

set(CPACK_DEBIAN_PACKAGE_DEBUG ON)

include(CPack)
