#include <iostream>
#ifndef HELPER_H
#define HELPER_H
#define PRINT_ERROR                                                   \
    std::cout << "error happens in line " << __LINE__ << ", in file " \
              << __FILE__ << std::endl

#define PRINT_INFO \
    //std::cout << "done here in line " << __LINE__ << ", in file " << __FILE__ \
              << std::endl

#define PRINT_ERROR_MESSAGE(message)                                      \
    std::cout << "error in line " << __LINE__ << ", in file " << __FILE__ \
              << ", error message is " << message << std::endl;
#endif
