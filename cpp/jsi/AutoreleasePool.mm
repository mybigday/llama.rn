#include "AutoreleasePool.h"
#import <Foundation/Foundation.h>

#ifdef __APPLE__
void rnllama_run_in_autorelease_pool(const std::function<void()>& fn) {
    @autoreleasepool {
        fn();
    }
}
#endif
