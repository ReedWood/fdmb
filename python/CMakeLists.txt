# Copyright (C) 2012  Matthias Vogelgesang [1]
# Copyright (C) 2015  Wolfgang Mader <Wolfgang.Mader@fdm.uni-freiburg.de>
# [1] http://bloerg.net/2012/11/10/cmake-and-distutils.html
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


find_program(PYTHON "python")

if(PYTHON)
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/fdmb/__init__.py")
  set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

  configure_file(${SETUP_PY_IN} ${SETUP_PY})

  add_custom_command(OUTPUT ${OUTPUT}
                     COMMAND ${PYTHON} ${SETUP_PY} build
                     COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                     DEPENDS ${DEPS})

  add_custom_target(target ALL DEPENDS ${OUTPUT})

  if(DEFINED ENV{BUILDING_FOR_AUR})
    message(STATUS "Building AUR packages")
  else()
    message(STATUS "Install to prefix")
    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX})")
  endif()
endif()