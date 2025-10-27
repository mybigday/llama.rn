package com.rnllama;

import androidx.annotation.NonNull;
import android.util.Log;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.Arguments;

import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

public class RNLlama implements LifecycleEventListener {
  public static final String NAME = "RNLlama";

  private final ReactApplicationContext reactContext;
  private final ExecutorService executorService;
  private final Handler mainHandler;

  public RNLlama(ReactApplicationContext reactContext) {
    reactContext.addLifecycleEventListener(this);
    this.reactContext = reactContext;
    this.executorService = Executors.newCachedThreadPool();
    this.mainHandler = new Handler(Looper.getMainLooper());
  }

  private final HashMap<Future<?>, String> tasks = new HashMap<>();

  private final HashMap<Integer, LlamaContext> contexts = new HashMap<>();

  public void toggleNativeLog(boolean enabled, Promise promise) {
    executorService.execute(() -> {
      try {
        LlamaContext.toggleNativeLog(reactContext, enabled);
        mainHandler.post(() -> promise.resolve(true));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  private int llamaContextLimit = -1;

  public void setContextLimit(double limit, Promise promise) {
    llamaContextLimit = (int) limit;
    promise.resolve(null);
  }

  public void modelInfo(final String model, final ReadableArray skip, final Promise promise) {
    executorService.execute(() -> {
      try {
        String[] skipArray = new String[skip.size()];
        for (int i = 0; i < skip.size(); i++) {
          skipArray[i] = skip.getString(i);
        }
        WritableMap result = LlamaContext.modelInfo(model, skipArray);
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void getBackendDevicesInfo(final Promise promise) {
    executorService.execute(() -> {
      try {
        if (LlamaContext.isArchNotSupported()) {
          throw new IllegalStateException("Only 64-bit architectures are supported");
        }
        String result = LlamaContext.getBackendDevicesInfo();
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void initContext(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context != null) {
          throw new Exception("Context already exists");
        }
        if (llamaContextLimit > -1 && contexts.size() >= llamaContextLimit) {
          throw new Exception("Context limit reached");
        }
        LlamaContext llamaContext = new LlamaContext(contextId, reactContext, params);
        if (llamaContext.getContext() == 0) {
          throw new Exception("Failed to initialize context");
        }
        contexts.put(contextId, llamaContext);
        WritableMap result = Arguments.createMap();
        result.putBoolean("gpu", llamaContext.isGpuEnabled());
        result.putString("reasonNoGPU", llamaContext.getReasonNoGpu());
        String gpuDevice = llamaContext.getGpuDevice();
        if (gpuDevice != null && !gpuDevice.isEmpty()) {
          result.putString("gpuDevice", gpuDevice);
        }
        result.putMap("model", llamaContext.getModelDetails());
        result.putString("androidLib", llamaContext.getLoadedLibrary());
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals("initContext"));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals("initContext"));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, "initContext");
    }
  }

  public void getFormattedChat(double id, final String messages, final String chatTemplate, final ReadableMap params, Promise promise) {
    final int contextId = (int) id;
    String taskName = "getFormattedChat-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Object result;
        if (params.hasKey("jinja") && params.getBoolean("jinja")) {
          ReadableMap resultMap = context.getFormattedChatWithJinja(
            messages,
            chatTemplate == null ? "" : chatTemplate,
            params
          );
          if (resultMap.hasKey("_error")) {
            throw new Exception(resultMap.getString("_error"));
          }
          result = resultMap;
        } else {
          result = context.getFormattedChat(messages, chatTemplate);
        }
        Object finalResult = result;
        mainHandler.post(() -> {
          promise.resolve(finalResult);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void loadSession(double id, final String path, Promise promise) {
    final int contextId = (int) id;
    String taskName = "loadSession-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        WritableMap result = context.loadSession(path);
        if (result != null && result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void saveSession(double id, final String path, double size, Promise promise) {
    final int contextId = (int) id;
    String taskName = "saveSession-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        WritableMap result = context.saveSession(path, (int) size);
        if (result != null && result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        mainHandler.post(() -> {
          // Return the tokens_saved count to maintain backward compatibility
          if (result != null && result.hasKey("tokens_saved")) {
            promise.resolve(result.getInt("tokens_saved"));
          } else {
            promise.resolve(0);
          }
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void completion(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "completion-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (context.isPredicting()) {
          throw new Exception("Context is busy");
        }
        WritableMap result = context.completion(params);
        if (result != null && result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void stopCompletion(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "stopCompletion-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        context.stopCompletion();
        // Wait for completion task to finish
        synchronized (tasks) {
          for (Future<?> task : tasks.keySet()) {
            if (tasks.get(task).equals("completion-" + contextId)) {
              try {
                task.get();
              } catch (Exception ignored) {
              }
              break;
            }
          }
        }
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void tokenize(double id, final String text, final ReadableArray media_paths, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "tokenize-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        WritableMap result = context.tokenize(text, media_paths);
        if (result != null && result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "detokenize-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        String result = context.detokenize(tokens);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void embedding(double id, final String text, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "embedding-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        WritableMap result = context.getEmbedding(text, params);
        if (result != null && result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void rerank(double id, final String query, final ReadableArray documents, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "rerank-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        WritableMap result = context.getRerank(query, documents, params);
        if (result == null) {
          throw new Exception("Rerank returned null result");
        }
        if (result.hasKey("error")) {
          throw new Exception(result.getString("error"));
        }
        // Extract the array from the result map
        WritableArray finalResult;
        if (result.hasKey("result")) {
          ReadableArray readableResult = result.getArray("result");
          // Convert ReadableArray to WritableArray by copying
          WritableArray writableResult = Arguments.createArray();
          for (int i = 0; i < readableResult.size(); i++) {
            writableResult.pushMap(readableResult.getMap(i));
          }
          finalResult = writableResult;
        } else {
          // Fallback to empty array if no result key (shouldn't happen)
          finalResult = Arguments.createArray();
        }
        mainHandler.post(() -> {
          promise.resolve(finalResult);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void bench(double id, final double pp, final double tg, final double pl, final double nr, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "bench-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        String result = context.bench((int) pp, (int) tg, (int) pl, (int) nr);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void applyLoraAdapters(double id, final ReadableArray loraAdapters, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "applyLoraAdapters-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (context.isPredicting()) {
          throw new Exception("Context is busy");
        }
        context.applyLoraAdapters(loraAdapters);
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void removeLoraAdapters(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "removeLoraAdapters-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (context.isPredicting()) {
          throw new Exception("Context is busy");
        }
        context.removeLoraAdapters();
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void getLoadedLoraAdapters(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "getLoadedLoraAdapters-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        ReadableArray result = context.getLoadedLoraAdapters();
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void initMultimodal(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "initMultimodal-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (context.isPredicting()) {
          throw new Exception("Context is busy");
        }
        Boolean result = context.initMultimodal(params);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void isMultimodalEnabled(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "isMultimodalEnabled-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Boolean result = context.isMultimodalEnabled();
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void getMultimodalSupport(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "getMultimodalSupport-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (!context.isMultimodalEnabled()) {
          throw new Exception("Multimodal is not enabled");
        }
        WritableMap result = context.getMultimodalSupport();
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  @ReactMethod
  public void releaseMultimodal(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "releaseMultimodal-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        context.releaseMultimodal();
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void initVocoder(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "initVocoder-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (context.isPredicting()) {
          throw new Exception("Context is busy");
        }
        Boolean result = context.initVocoder(params);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void releaseVocoder(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "releaseVocoder-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        context.releaseVocoder();
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void isVocoderEnabled(double id, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "isVocoderEnabled-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Boolean result = context.isVocoderEnabled();
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void getFormattedAudioCompletion(double id, final String speakerJsonStr, final String textToSpeak, Promise promise) {
    final int contextId = (int) id;
    String taskName = "getFormattedAudioCompletion-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (!context.isVocoderEnabled()) {
          throw new Exception("Vocoder is not enabled");
        }
        Object result = context.getFormattedAudioCompletion(speakerJsonStr, textToSpeak);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void getAudioCompletionGuideTokens(double id, final String textToSpeak, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "getAudioCompletionGuideTokens-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (!context.isVocoderEnabled()) {
          throw new Exception("Vocoder is not enabled");
        }
        WritableArray result = context.getAudioCompletionGuideTokens(textToSpeak);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void decodeAudioTokens(double id, final ReadableArray tokens, final Promise promise) {
    final int contextId = (int) id;
    String taskName = "decodeAudioTokens-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        if (!context.isVocoderEnabled()) {
          throw new Exception("Vocoder is not enabled");
        }
        WritableArray result = context.decodeAudioTokens(tokens);
        mainHandler.post(() -> {
          promise.resolve(result);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void releaseContext(double id, Promise promise) {
    final int contextId = (int) id;
    String taskName = "releaseContext-" + contextId;
    Future<?> future = executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context " + id + " not found");
        }
        context.interruptLoad();
        context.stopCompletion();
        synchronized (tasks) {
          for (Future<?> task : tasks.keySet()) {
            if (tasks.get(task).equals("completion-" + contextId)) {
              try {
                task.get();
              } catch (Exception ignored) {
              }
              break;
            }
          }
        }
        context.release();
        contexts.remove(contextId);
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  public void releaseAllContexts(Promise promise) {
    String taskName = "releaseAllContexts";
    Future<?> future = executorService.submit(() -> {
      try {
        onHostDestroy();
        mainHandler.post(() -> {
          promise.resolve(null);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      } catch (Exception e) {
        mainHandler.post(() -> {
          promise.reject(e);
          synchronized (tasks) {
            tasks.values().removeIf(name -> name.equals(taskName));
          }
        });
      }
    });
    synchronized (tasks) {
      tasks.put(future, taskName);
    }
  }

  // Parallel decoding methods
  public void enableParallelMode(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }

        boolean enabled = params.hasKey("enabled") ? params.getBoolean("enabled") : true;
        int nParallel = params.hasKey("n_parallel") ? params.getInt("n_parallel") : 2;
        int nBatch = params.hasKey("n_batch") ? params.getInt("n_batch") : 512;

        Boolean result;
        if (enabled) {
          result = context.doEnableParallelMode(nParallel, nBatch);
        } else {
          context.doDisableParallelMode();
          result = true;
        }
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void queueCompletion(double id, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Integer requestId = context.queueCompletion(params);
        WritableMap result = Arguments.createMap();
        result.putInt("requestId", requestId);
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void cancelRequest(double id, double requestIdDouble, final Promise promise) {
    final int contextId = (int) id;
    final int requestId = (int) requestIdDouble;
    executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        context.cancelRequest(requestId);
        mainHandler.post(() -> promise.resolve(null));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void queueEmbedding(double id, final String text, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Integer requestId = context.queueEmbedding(text, params);
        WritableMap result = Arguments.createMap();
        result.putInt("requestId", requestId);
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  public void queueRerank(double id, final String query, final ReadableArray documents, final ReadableMap params, final Promise promise) {
    final int contextId = (int) id;
    executorService.submit(() -> {
      try {
        LlamaContext context = contexts.get(contextId);
        if (context == null) {
          throw new Exception("Context not found");
        }
        Integer requestId = context.queueRerank(query, documents, params);
        WritableMap result = Arguments.createMap();
        result.putInt("requestId", requestId);
        mainHandler.post(() -> promise.resolve(result));
      } catch (Exception e) {
        mainHandler.post(() -> promise.reject(e));
      }
    });
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
    for (LlamaContext context : contexts.values()) {
      context.stopCompletion();
    }
    synchronized (tasks) {
      for (Future<?> task : tasks.keySet()) {
        try {
          task.get();
        } catch (Exception e) {
          Log.e(NAME, "Failed to wait for task", e);
        }
      }
      tasks.clear();
    }
    for (LlamaContext context : contexts.values()) {
      context.release();
    }
    contexts.clear();
    executorService.shutdown();
    try {
      if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
        executorService.shutdownNow();
      }
    } catch (InterruptedException e) {
      executorService.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }
}
