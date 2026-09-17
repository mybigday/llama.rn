#pragma once
#ifdef __APPLE__
#include <functional>
// Runs fn() inside an explicit @autoreleasepool so the Objective-C objects
// Metal creates while freeing a context's buffers are drained synchronously.
// The ThreadPool workers are std::threads that never exit, so their implicit
// pool never drains on its own.
void rnllama_run_in_autorelease_pool(const std::function<void()>& fn);
#endif
