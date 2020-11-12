set(TARGET asl_turtlebot)

set(SOURCES 
  SquirtleBotCommander.cpp
)

set(HEADERS
  SquirtleBotCommander.h
)

set( MOC_HEADERS
  SquirtleBotCommander.h
)

set( UIC_SOURCES 
  SquirtleBotCommander.ui
)

qt5_wrap_ui(UIC_FILES ${UIC_SOURCES} )
qt5_wrap_cpp(MOC_FILES ${MOC_HEADERS})
include_directories(
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)
set( QT_LINK_LIBRARIES Qt5::Widgets )

include_directories( 
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(${TARGET}
  ${SOURCES}
  ${HEADERS}
  ${MOC_HEADERS}
  ${MOC_FILES}
  ${UIC_FILES}
)

target_link_libraries(${TARGET}
  ${QT_LINK_LIBRARIES}
  ${catkin_LIBRARIES}
)

install(TARGETS ${TARGET}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
