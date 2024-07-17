
#include "utils.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

void __fatal(const char* message, ...) {
  va_list arg_list = {};
  va_start(arg_list, message);
  vprintf(message, arg_list);
  va_end(arg_list);

  std::abort();
}

void __checkFailed(const char* expr, int line, const char* source) {
  __fatal("\nCHECK failed [%s] at %s:%d\n\n", expr, source, line);
}

void __checkFailed(const char* expr,
                   int line,
                   const char* source,
                   const char* message,
                   ...) {
  printf("\nFATAL: ");

  va_list arg_list = {};
  va_start(arg_list, message);
  vprintf(message, arg_list);
  va_end(arg_list);

  printf("\n");

  __checkFailed(expr, line, source);
}
