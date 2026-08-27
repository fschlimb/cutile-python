foreach(_required GIT_EXECUTABLE SOURCE_DIR REVISION)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

if(NOT EXISTS "${SOURCE_DIR}/.git")
    if(EXISTS "${SOURCE_DIR}")
        file(GLOB _source_contents "${SOURCE_DIR}/*" "${SOURCE_DIR}/.[!.]*")
        if(_source_contents)
            message(FATAL_ERROR
                "LLVM source directory exists but is not a Git checkout: ${SOURCE_DIR}")
        endif()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --filter=blob:none --no-checkout
                https://github.com/llvm/llvm-project.git "${SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE _head_result
    OUTPUT_VARIABLE _head_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(_head_result EQUAL 0 AND _head_revision STREQUAL REVISION)
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" cat-file -e "${REVISION}^{commit}"
    RESULT_VARIABLE _revision_exists
    ERROR_QUIET)
if(NOT _revision_exists EQUAL 0)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" fetch --filter=blob:none
                origin "${REVISION}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" checkout --detach "${REVISION}"
    COMMAND_ERROR_IS_FATAL ANY)