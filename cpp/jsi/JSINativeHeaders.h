#pragma once

#if defined(RNLLAMA_USE_FRAMEWORK_HEADERS)
#include <rnllama/rn-llama.h>
#include <rnllama/rn-completion.h>
#include <rnllama/rn-slot.h>
#include <rnllama/rn-slot-manager.h>
#include <rnllama/chat.h>
#include <rnllama/gguf.h>
#include <rnllama/ggml-backend.h>
#include <rnllama/common.h>
#include <rnllama/json-schema-to-grammar.h>
#include <rnllama/nlohmann/json.hpp>
#else
#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-slot.h"
#include "rn-slot-manager.h"
#include "chat.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"

#endif
